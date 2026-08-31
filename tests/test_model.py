"""M0 integrity tests: every vendored scene loads and behaves sanely on CPU."""

import mujoco
import numpy as np
import pytest

from microduck_sim.model import (
    KEYFRAME_NAMES,
    POLICIES_DIR,
    SCENE_VARIANTS,
    actuator_names,
    joint_names,
    load,
    load_model,
    reset_to_keyframe,
)


@pytest.mark.parametrize("variant", sorted(SCENE_VARIANTS))
def test_scene_loads(variant):
    model = load_model(variant)
    assert model.nu > 0
    assert model.nbody > 1


@pytest.mark.parametrize("variant", sorted(set(SCENE_VARIANTS) - {"ball"}))
def test_keyframes_present(variant):
    model = load_model(variant)
    names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(model.nkey)
    }
    assert set(KEYFRAME_NAMES) <= names


def test_walk_model_structure():
    model = load_model("walk")
    # freejoint (7 qpos) + one hinge per actuated joint
    assert model.nu == 14
    assert model.nq == 7 + model.nu
    acts = actuator_names(model)
    for side in ("left", "right"):
        for joint in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"):
            assert f"{side}_{joint}" in acts
    assert "trunk_base_freejoint" in joint_names(model)


def test_imu_sensors_exist():
    model = load_model("walk")
    sensors = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        for i in range(model.nsensor)
    }
    assert {"orientation", "angular-velocity", "imu_accel"} <= sensors


def test_stand_holds_short_term_open_loop():
    """From STAND with keyframe ctrl held, the duck stays up for 0.5 s.

    Open-loop standing tips over after ~0.9 s (measured) — that instability is
    real and is the motivation for feedback control in Lab 3, so we only assert
    short-term stability here.
    """
    model, data = load(variant="walk")
    steps = int(0.5 / model.opt.timestep)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))
    assert data.qpos[2] > 0.10  # trunk height at STAND is 0.12


def test_reset_to_unknown_keyframe_raises():
    model = load_model("walk")
    data = mujoco.MjData(model)
    with pytest.raises(ValueError, match="not found"):
        reset_to_keyframe(model, data, "NOPE")


def test_policies_vendored():
    onnx_files = list(POLICIES_DIR.glob("*.onnx"))
    assert len(onnx_files) == 9
    for f in onnx_files:
        header = f.read_bytes()[:64]
        assert b"git-lfs" not in header, f"{f.name} is an LFS pointer, not a model"
