"""Export a Stable-Baselines3 PPO policy to a contract-compatible ONNX file.

The exported file has the same interface as Pollen's shipped policies —
input ``obs`` [1, 61] → output ``actions`` [1, 14] (deterministic mean action,
clamped to the env's action limit) — so it can be:
- replayed by `microduck_sim.policy.OnnxPolicy` / `PolicyRunner`,
- published to the Hugging Face Hub like the community checkpoints (Lab 8).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def export_sb3_to_onnx(
    model_path: str | Path,
    out_path: str | Path,
    action_limit: float = 0.6,
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
        dynamo=False,
    )
    return out_path
