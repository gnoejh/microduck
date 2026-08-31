"""M2 tests: balance control, pushes, IMU model, estimation."""

import mujoco
import numpy as np
import pytest

from microduck_sim.control import (
    constant,
    keyframe_ctrl,
    make_balance_controller,
    make_push,
    quat_to_pitch_roll,
    run_control_loop,
)
from microduck_sim.estimation import ComplementaryFilter, accel_pitch
from microduck_sim.model import load_model, reset_to_keyframe
from microduck_sim.sensors import NoisyIMU, read_sensor


@pytest.fixture()
def duck():
    model = load_model("full")
    data = mujoco.MjData(model)
    reset_to_keyframe(model, data, "STAND")
    return model, data


def stands_for(model, data, ctrl_fn, duration, on_tick=None):
    log = run_control_loop(model, data, ctrl_fn, duration, on_tick=on_tick)
    return not np.any(log["qpos"][:, 2] < 0.10)


def test_quat_to_pitch_roll_identity_and_pure_pitch():
    assert quat_to_pitch_roll(np.array([1, 0, 0, 0])) == (0.0, 0.0)
    angle = 0.3
    q = np.array([np.cos(angle / 2), 0, np.sin(angle / 2), 0])  # pitch about y
    pitch, roll = quat_to_pitch_roll(q)
    assert abs(pitch - angle) < 1e-6 and abs(roll) < 1e-6


def test_balance_controller_stands_10s(duck):
    model, data = duck
    fn = make_balance_controller(model, data, keyframe_ctrl(model, "STAND"))
    assert stands_for(model, data, fn, 10.0)


def test_wrong_sign_falls_fast(duck):
    model, data = duck
    fn = make_balance_controller(
        model, data, keyframe_ctrl(model, "STAND"), kp_pitch=-1.0, kd_pitch=-0.1
    )
    log = run_control_loop(model, data, fn, 2.0)
    below = log["qpos"][:, 2] < 0.10
    assert below.any() and log["t"][np.argmax(below)] < 0.8  # worse than no feedback


def test_balance_survives_1N_push_not_2N(duck):
    model, data = duck
    fn = make_balance_controller(model, data, keyframe_ctrl(model, "STAND"))
    push = make_push(model, data, force=1.0, axis=0)
    assert stands_for(model, data, fn, 5.0, on_tick=push)
    data.xfrc_applied[:] = 0

    reset_to_keyframe(model, data, "STAND")
    fn = make_balance_controller(model, data, keyframe_ctrl(model, "STAND"))
    push = make_push(model, data, force=2.0, axis=0)
    assert not stands_for(model, data, fn, 5.0, on_tick=push)
    data.xfrc_applied[:] = 0


def test_imu_invalid_during_free_fall_then_valid(duck):
    """At reset the duck floats a hair above the floor: free fall ⇒ accel ≈ 0
    (specific force!), so the sample must be flagged invalid until touchdown."""
    model, data = duck
    imu = NoisyIMU(model, data)
    assert not imu.read().valid
    for _ in range(50):  # 100 ms: lands and loads the feet
        mujoco.mj_step(model, data)
    sample = imu.read()
    assert sample.valid
    assert 5.0 < np.linalg.norm(sample.accel) < 20.0  # gravity visible again


def test_noisy_imu_deterministic(duck):
    model, data = duck
    mujoco.mj_step(model, data)
    a = NoisyIMU(model, data, gyro_noise=0.02, accel_noise=0.4, seed=7).read()
    b = NoisyIMU(model, data, gyro_noise=0.02, accel_noise=0.4, seed=7).read()
    assert np.allclose(a.gyro, b.gyro) and np.allclose(a.accel, b.accel)


def test_complementary_filter_tracks_true_pitch(duck):
    model, data = duck
    imu = NoisyIMU(model, data, gyro_noise=0.02, accel_noise=0.4, gyro_bias=0.05)
    filt = ComplementaryFilter()
    errs = []

    def on_tick(m, d):
        est = filt.update(imu.read())
        errs.append(abs(est - quat_to_pitch_roll(d.qpos[3:7])[0]))

    balance = make_balance_controller(model, data, keyframe_ctrl(model, "STAND"))
    run_control_loop(model, data, balance, 5.0, on_tick=on_tick)
    assert np.mean(errs[25:]) < 0.1  # after settling


def test_balance_on_estimated_pitch_stands(duck):
    """The Lab 4 finale: stand on noisy IMU + filter only, no ground truth."""
    model, data = duck
    imu = NoisyIMU(model, data, gyro_noise=0.02, accel_noise=0.4, gyro_bias=0.05)
    filt = ComplementaryFilter()
    latest = {"sample": None}

    def pitch_fn():
        latest["sample"] = imu.read()
        return filt.update(latest["sample"])

    def rate_fn():
        return float(latest["sample"].gyro[1])

    fn = make_balance_controller(
        model, data, keyframe_ctrl(model, "STAND"), pitch_fn=pitch_fn, rate_fn=rate_fn
    )
    assert stands_for(model, data, fn, 10.0)


def test_filter_handles_invalid_start():
    filt = ComplementaryFilter()
    from microduck_sim.sensors import IMUSample

    bad = IMUSample(gyro=np.zeros(3), accel=np.zeros(3), valid=False)
    assert filt.update(bad) == 0.0  # must NOT be atan2(0,0) = π
    good = IMUSample(gyro=np.zeros(3), accel=np.array([0, 0, 9.81]), valid=True)
    assert abs(filt.update(good)) < 1e-6
