"""Run Pollen's pretrained ONNX policies in the CPU simulator (Lab 5).

Reproduces the loop from ``microduck_rl/scripts/infer_policy.py`` (and the
browser playground): physics at timestep 0.005 s, one inference every 4 steps
(= 50 Hz), ``ctrl = DEFAULT_POSE + action * ACTION_SCALE``.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from microduck_sim.model import POLICIES_DIR, load_model, reset_to_keyframe
from microduck_sim.obs import (
    ACTION_SCALE,
    CMD_SIZE,
    DEFAULT_POSE,
    NUM_JOINTS,
    ObsBuilder,
)

POLICY_TIMESTEP = 0.005
DECIMATION = 4  # 4 × 0.005 s = one 50 Hz control tick

# Behavior name → vendored ONNX file (from the official browser simulator).
BEHAVIORS = {
    "walk": "BEST_alpha_walking.onnx",
    "stand": "BEST_alpha_stand.onnx",
    "sitstand": "BEST_alpha_sitstand.onnx",
    "roll": "roulade.onnx",
    "kick_left": "ball_kick_left.onnx",
    "kick_right": "ball_kick_right.onnx",
    "ground_pick": "alpha_ground_pick.onnx",
    "roller_drive": "BEST_roller.onnx",
    "roller_crouch": "BEST_roller_crouch.onnx",
}


class OnnxPolicy:
    """One pretrained behavior: obs (61) in → action (14) out."""

    def __init__(self, name_or_path: str | Path):
        import onnxruntime as ort

        path = Path(name_or_path)
        if not path.suffix:  # behavior name
            path = POLICIES_DIR / BEHAVIORS[str(name_or_path)]
        self.path = path
        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def act(self, obs: np.ndarray) -> np.ndarray:
        out = self.session.run(
            [self.output_name], {self.input_name: obs.reshape(1, -1).astype(np.float32)}
        )
        return np.asarray(out[0]).ravel()[:NUM_JOINTS]


class PolicyRunner:
    """50 Hz policy loop on a Microduck scene, one tick per ``step()``.

    ``command`` is the live 13-dim command buffer (twist in [0:3], head in
    [3:7]) — mutate it between ticks to steer, exactly like the gamepad does.
    """

    def __init__(
        self,
        policy: OnnxPolicy | str,
        variant: str = "full",
        keyframe: str = "STAND",
    ):
        self.policy = policy if isinstance(policy, OnnxPolicy) else OnnxPolicy(policy)
        self.model = load_model(variant)
        self.model.opt.timestep = POLICY_TIMESTEP  # match the training setup
        self.data = mujoco.MjData(self.model)
        self.obs_builder = ObsBuilder(self.model)
        self.command = np.zeros(CMD_SIZE)
        self.last_action = np.zeros(NUM_JOINTS)
        self.reset(keyframe)

    def reset(self, keyframe: str = "STAND") -> None:
        reset_to_keyframe(self.model, self.data, keyframe)
        self.last_action[:] = 0.0

    def step(self) -> dict:
        """One control tick: build obs, infer, apply, 4 physics substeps."""
        obs = self.obs_builder.build(self.data, self.last_action, self.command)
        action = self.policy.act(obs)
        self.last_action = action.copy()
        self.data.ctrl[:NUM_JOINTS] = DEFAULT_POSE + action * ACTION_SCALE
        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)
        return {
            "obs": obs,
            "action": action,
            "qpos": self.data.qpos.copy(),
            "time": self.data.time,
        }

    def run(self, duration: float) -> dict[str, np.ndarray]:
        """Run for `duration` seconds; return stacked logs."""
        logs = [self.step() for _ in range(int(round(duration * 50)))]
        return {k: np.asarray([row[k] for row in logs]) for k in logs[0]}

    @property
    def upright(self) -> bool:
        """Same criterion as the playground: projected gravity z < -0.5."""
        return self.obs_builder.projected_gravity(self.data)[2] < -0.5
