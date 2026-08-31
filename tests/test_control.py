"""M1 tests: control-loop and trajectory helpers."""

import numpy as np
import pytest

from microduck_sim.control import (
    chain,
    constant,
    goto,
    keyframe_ctrl,
    lerp,
    run_control_loop,
    sine,
)
from microduck_sim.model import load


@pytest.fixture(scope="module")
def stand():
    return load("walk")


def test_keyframe_ctrl_matches_model(stand):
    model, _ = stand
    ctrl = keyframe_ctrl(model, "STAND")
    assert ctrl.shape == (model.nu,)
    assert not np.allclose(ctrl, 0)
    with pytest.raises(ValueError):
        keyframe_ctrl(model, "NOPE")


def test_lerp_endpoints_and_clipping():
    a, b = np.zeros(3), np.ones(3)
    assert np.allclose(lerp(a, b, 0.0), a)
    assert np.allclose(lerp(a, b, 1.0), b)
    assert np.allclose(lerp(a, b, -5.0), a)
    assert np.allclose(lerp(a, b, 5.0), b)


def test_goto_is_smooth_and_reaches_target():
    fn = goto(np.zeros(2), np.ones(2), duration=1.0)
    assert np.allclose(fn(0.0), 0.0)
    assert np.allclose(fn(1.0), 1.0)
    assert np.allclose(fn(99.0), 1.0)  # holds after the end
    # min-jerk: midpoint is halfway, and starts slowly
    assert np.allclose(fn(0.5), 0.5)
    assert np.all(fn(0.05) < 0.01)


def test_sine_oscillates_around_center():
    fn = sine(center=np.ones(2), amplitude=0.5 * np.ones(2), frequency=2.0)
    assert np.allclose(fn(0.0), 1.0)
    assert np.allclose(fn(0.125), 1.5)  # quarter period of 2 Hz
    assert np.allclose(fn(0.25), 1.0, atol=1e-9)


def test_chain_uses_local_time_and_holds_last():
    seg1 = goto(np.zeros(1), np.ones(1), 1.0)
    seg2 = constant(np.full(1, 5.0))
    fn = chain((1.0, seg1), (1.0, seg2))
    assert np.allclose(fn(0.0), 0.0)
    assert np.allclose(fn(1.0), 1.0)
    assert np.allclose(fn(1.5), 5.0)
    assert np.allclose(fn(42.0), 5.0)


def test_run_control_loop_timing_and_log(stand):
    model, data = stand
    from microduck_sim.model import reset_to_keyframe

    reset_to_keyframe(model, data, "STAND")
    hold = constant(keyframe_ctrl(model, "STAND"))
    log = run_control_loop(model, data, hold, duration=0.5)
    assert log["t"].shape == (25,)  # 0.5 s at 50 Hz
    assert log["qpos"].shape == (25, model.nq)
    assert log["ctrl"].shape == (25, model.nu)
    assert log["site_xpos"].shape == (25, model.nsite, 3)
    # 50 Hz tick spacing
    assert np.allclose(np.diff(log["t"]), 0.02, atol=1e-9)
    # still standing after 0.5 s
    assert log["qpos"][-1, 2] > 0.10


def test_head_nod_moves_head_and_stays_up_short_term(stand):
    """Head oscillation on top of STAND: head tracks the sine, duck upright ≥0.6 s.

    (Anything much longer topples — open-loop standing is only marginally
    stable, which Lab 3 fixes with feedback.)
    """
    model, data = stand
    from microduck_sim.model import actuator_names, joint_names, reset_to_keyframe

    reset_to_keyframe(model, data, "STAND")
    base = keyframe_ctrl(model, "STAND")
    amp = np.zeros(model.nu)
    amp[actuator_names(model).index("head_pitch")] = 0.3
    log = run_control_loop(model, data, sine(base, amp, frequency=1.0), duration=0.6)
    assert log["qpos"][-1, 2] > 0.10
    # head_pitch qpos index: freejoint takes 7 slots, hinges follow joint order
    head_qpos = log["qpos"][:, 6 + joint_names(model).index("head_pitch")]
    assert head_qpos.max() - head_qpos.min() > 0.3
