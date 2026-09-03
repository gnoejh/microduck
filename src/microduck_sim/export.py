"""Export a Stable-Baselines3 PPO policy to a contract-compatible ONNX file.

The exported file has the same interface as Pollen's shipped policies —
input ``obs`` [1, 61] → output ``actions`` [1, 14] (deterministic mean action,
clamped to the env's action limit), opset 18 like theirs — so it can be:
- replayed by `microduck_sim.policy.OnnxPolicy` / `PolicyRunner`,
- loaded by the browser playground (onnxruntime-web, see `web.py`),
- published to the Hugging Face Hub like the community checkpoints (Lab 8).

It also carries Pollen-style ``metadata_props`` (joint names, default pose,
observation/command names, action scale) plus a few course keys, so a file is
self-describing wherever it ends up. Note: our policies are trained on RAW
observations (no VecNormalize), so unlike Pollen's there is no normalizer inside
the graph — ``normalizer="none"`` says so.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .obs import DEFAULT_POSE, JOINT_NAMES

ONNX_OPSET = 18  # Pollen parity; onnxruntime-web in the playground is happiest here

#: Pollen's metadata keys, in their comma-separated string format.
OBSERVATION_NAMES = (
    "base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command"
)


def contract_metadata(
    *, command_names: str = "twist", action_scale: float = 1.0, **extra: str
) -> dict[str, str]:
    """The metadata block written into every course-exported ONNX."""
    meta = {
        "joint_names": ",".join(JOINT_NAMES),
        "joint_stiffness": ",".join("1.000" for _ in JOINT_NAMES),
        "joint_damping": ",".join("0.000" for _ in JOINT_NAMES),
        "default_joint_pos": ",".join(f"{x:.3f}" for x in DEFAULT_POSE),
        "command_names": command_names,
        "observation_names": OBSERVATION_NAMES,
        "action_scale": f"{action_scale}",
        "normalizer": "none",
        "source": "microduck course (SB3 PPO, CPU)",
    }
    meta.update({k: str(v) for k, v in extra.items()})
    return meta


def write_onnx_metadata(path: str | Path, metadata: dict[str, str]) -> None:
    """Attach/overwrite ``metadata_props`` on an existing ONNX file."""
    import onnx

    model = onnx.load(str(path))
    existing = {p.key: p for p in model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = value
        else:
            model.metadata_props.add(key=key, value=value)
    onnx.save(model, str(path))


def read_onnx_metadata(path: str | Path) -> dict[str, str]:
    """The ``metadata_props`` of an ONNX file as a dict (empty if none)."""
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    return {p.key: p.value for p in model.metadata_props}


def export_sb3_to_onnx(
    model_path: str | Path,
    out_path: str | Path,
    action_limit: float = 0.6,
    *,
    opset_version: int = ONNX_OPSET,
    metadata: dict[str, str] | None = None,
    command_names: str = "twist",
) -> Path:
    """Export a saved SB3 PPO checkpoint (.zip) to ONNX. Returns the path."""
    import torch
    from stable_baselines3 import PPO

    model = PPO.load(str(model_path), device="cpu")
    policy = model.policy
    policy.eval()

    class ActorOnly(torch.nn.Module):
        def __init__(self, policy, limit):
            super().__init__()
            self.policy = policy
            self.limit = limit

        def forward(self, obs):
            features = self.policy.extract_features(
                obs, self.policy.pi_features_extractor
            )
            latent_pi = self.policy.mlp_extractor.forward_actor(features)
            actions = self.policy.action_net(latent_pi)
            return torch.clamp(actions, -self.limit, self.limit)

    wrapper = ActorOnly(policy, action_limit)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, int(np.prod(policy.observation_space.shape)))
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        input_names=["obs"],
        output_names=["actions"],
        opset_version=opset_version,
        dynamo=False,
    )
    meta = contract_metadata(
        command_names=command_names,
        clip_actions=f"{action_limit}",
        checkpoint=Path(model_path).name,
    )
    if metadata:
        meta.update({k: str(v) for k, v in metadata.items()})
    write_onnx_metadata(out_path, meta)
    return out_path
