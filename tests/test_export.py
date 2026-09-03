"""Exported ONNX files must be self-describing and Pollen-compatible."""

import onnx
import pytest

from microduck_sim.export import ONNX_OPSET, contract_metadata, read_onnx_metadata
from microduck_sim.model import ASSETS_DIR
from microduck_sim.obs import DEFAULT_POSE, JOINT_NAMES
from microduck_sim.policy import OnnxPolicy

CHECKPOINTS = [ASSETS_DIR / "checkpoints" / f for f in ("stand_ppo.onnx", "walk_ppo.onnx")]


@pytest.mark.parametrize("path", CHECKPOINTS, ids=lambda p: p.name)
def test_checkpoint_opset_and_metadata(path):
    model = onnx.load(str(path))
    assert max(o.version for o in model.opset_import if o.domain == "") <= ONNX_OPSET
    meta = read_onnx_metadata(path)
    assert meta["joint_names"].split(",") == JOINT_NAMES
    pose = [float(x) for x in meta["default_joint_pos"].split(",")]
    assert all(abs(a - b) < 1e-3 for a, b in zip(pose, DEFAULT_POSE))
    assert meta["normalizer"] == "none" and meta["action_scale"] == "1.0"
    assert OnnxPolicy(path).metadata["observation_names"].startswith("base_ang_vel")


def test_contract_metadata_matches_pollen_keys():
    pollen = read_onnx_metadata(ASSETS_DIR / "policies" / "BEST_alpha_walking.onnx")
    ours = contract_metadata()
    for key in ("joint_names", "default_joint_pos", "observation_names", "command_names", "action_scale"):
        assert key in ours and key in pollen
    assert ours["observation_names"] == pollen["observation_names"]
    assert ours["default_joint_pos"] == pollen["default_joint_pos"]


def test_identity_pose_policy_exists():
    path = ASSETS_DIR / "checkpoints" / "identity_pose.onnx"
    policy = OnnxPolicy(path)
    assert policy.session.get_inputs()[0].shape[-1] == 14
