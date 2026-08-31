"""M3 tests: the 61-dim obs contract and ONNX policy playback (the linchpin)."""

import mujoco
import numpy as np
import pytest

from microduck_sim.model import POLICIES_DIR
from microduck_sim.obs import (
    CMD_SIZE,
    DEFAULT_POSE,
    JOINT_NAMES,
    NUM_JOINTS,
    OBS_SIZE,
    ObsBuilder,
)
from microduck_sim.policy import BEHAVIORS, OnnxPolicy, PolicyRunner


def test_all_vendored_policies_load_and_have_the_contract():
    for name in BEHAVIORS:
        p = OnnxPolicy(name)
        inp = p.session.get_inputs()[0]
        assert inp.shape[-1] == OBS_SIZE, name
        action = p.act(np.zeros(OBS_SIZE, dtype=np.float32))
        assert action.shape == (NUM_JOINTS,) and np.all(np.isfinite(action)), name


def test_obs_layout_upright_stand():
    r = PolicyRunner("stand")
    obs = r.obs_builder.build(r.data, np.zeros(NUM_JOINTS), np.zeros(CMD_SIZE))
    assert obs.shape == (OBS_SIZE,) and obs.dtype == np.float32
    # at STAND: joints at default pose → joint_pos block ≈ 0
    assert np.allclose(obs[6:20], 0.0, atol=1e-6)
    # upright: projected gravity ≈ [0, 0, -1]
    assert abs(obs[5] + 1.0) < 0.05 and np.all(np.abs(obs[3:5]) < 0.05)
    # zero velocity, zero last action, zero command
    assert np.allclose(obs[20:], 0.0, atol=1e-6)


def test_projected_gravity_flat_when_on_side():
    r = PolicyRunner("stand")
    # lie the duck on its back: rotate 90° about y
    r.data.qpos[3:7] = [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]
    mujoco.mj_forward(r.model, r.data)
    g = r.obs_builder.projected_gravity(r.data)
    assert abs(g[2]) < 0.05  # gravity now points along the trunk axis
    assert not r.upright


def test_stand_policy_balances_5s():
    r = PolicyRunner("stand")
    r.run(5.0)
    assert r.upright and r.data.qpos[2] > 0.10


def test_walk_policy_walks_forward():
    r = PolicyRunner("walk")
    r.command[0] = 0.25
    start = r.data.qpos[:2].copy()
    r.run(10.0)
    assert r.upright
    assert r.data.qpos[0] - start[0] > 0.5  # covers >0.5 m forward


def test_sitstand_policy_sits_on_flag():
    r = PolicyRunner("sitstand")
    r.command[0] = 1.0
    r.run(4.0)
    assert r.data.qpos[2] < 0.08  # seated height
    r.command[0] = 0.0
    r.run(4.0)
    assert r.data.qpos[2] > 0.10  # stood back up — what open-loop never managed


def test_getup_policy_recovers_from_prone():
    r = PolicyRunner("walk")
    r.command[0] = 0.25
    trunk = mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    for _ in range(100):
        r.step()
    r.data.xfrc_applied[trunk, 0] = 8.0
    for _ in range(10):
        r.step()
    r.data.xfrc_applied[trunk, 0] = 0.0
    for _ in range(50):
        r.step()
    assert not r.upright  # knocked flat

    r.policy = OnnxPolicy("stand")
    r.command[:] = 0.0
    r.last_action[:] = 0.0
    for _ in range(300):
        r.step()
        if r.obs_builder.projected_gravity(r.data)[2] < -0.85:
            break
    assert r.obs_builder.projected_gravity(r.data)[2] < -0.85  # back upright


def test_policy_runner_uses_training_timestep():
    r = PolicyRunner("walk")
    assert r.model.opt.timestep == pytest.approx(0.005)
