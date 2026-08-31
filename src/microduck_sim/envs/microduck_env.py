"""Microduck training environments (Labs 6–7).

Design notes (teaching-oriented):
- Same interfaces as the pretrained policies: obs 61 (ObsBuilder), 14 actions
  applied as ``ctrl = DEFAULT_POSE + action`` at 50 Hz, physics dt 0.005 s ×
  decimation 4 — the "walk" scene variant, exactly like upstream training.
- Rewards are sums of NAMED terms, reported per-step in ``info["reward_terms"]``
  so students can plot and ablate them (Lab 7).
- Falls terminate (projected gravity z > −0.5 or trunk below 0.06 m).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from microduck_sim.model import load_model, reset_to_keyframe
from microduck_sim.obs import CMD_SIZE, DEFAULT_POSE, NUM_JOINTS, OBS_SIZE, ObsBuilder
from microduck_sim.policy import DECIMATION, POLICY_TIMESTEP


class MicroduckEnv(gym.Env):
    """Base: standing task. Subclass and override `_command` / `_reward`."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    # actions are ctrl offsets [rad] from DEFAULT_POSE, clipped to ±0.6
    ACTION_LIMIT = 0.6

    def __init__(
        self,
        render_mode: str | None = None,
        variant: str = "walk",
        reward_weights: dict[str, float] | None = None,
    ):
        # per-term multipliers on top of the built-in weights, e.g.
        # gym.make("MicroduckWalk-v0", reward_weights={"upright": 0.2}) — the
        # Lab 7 ablation knob. Unknown term names raise at first step.
        self.reward_weights = dict(reward_weights or {})
        self.model = load_model(variant)
        self.model.opt.timestep = POLICY_TIMESTEP
        self.data = mujoco.MjData(self.model)
        self.obs_builder = ObsBuilder(self.model)
        self.render_mode = render_mode
        self._renderer = None

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_SIZE,), np.float32)
        self.action_space = gym.spaces.Box(
            -self.ACTION_LIMIT, self.ACTION_LIMIT, (NUM_JOINTS,), np.float32
        )
        self.command = np.zeros(CMD_SIZE)
        self.last_action = np.zeros(NUM_JOINTS)
        self.max_episode_steps = 500  # 10 s

    # -- task hooks --------------------------------------------------------
    def _sample_command(self) -> None:
        self.command[:] = 0.0

    def _reward(self, action: np.ndarray) -> dict[str, float]:
        """Named reward terms for STANDING (Lab 6's starting point)."""
        gz = self.obs_builder.projected_gravity(self.data)[2]
        height = float(self.data.qpos[2])
        joint_vel = self.data.qvel[6:]
        return {
            "upright": 1.0 * float(-gz),  # 1 when vertical, 0 on the side
            "height": 1.0 * float(np.exp(-(((height - 0.118) / 0.03) ** 2))),
            "still": 0.3 * float(np.exp(-0.1 * float(joint_vel @ joint_vel))),
            "action_rate": -0.05
            * float(np.sum(np.abs(action - self.last_action))),
        }

    # -- gym API -----------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        reset_to_keyframe(self.model, self.data, "STAND")
        # small random perturbation so the policy can't memorize one state
        self.data.qpos[7:] += self.np_random.uniform(-0.03, 0.03, NUM_JOINTS)
        self.data.qvel[:] += self.np_random.uniform(-0.05, 0.05, self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        self.last_action[:] = 0.0
        self._steps = 0
        self._sample_command()
        obs = self.obs_builder.build(self.data, self.last_action, self.command)
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -self.ACTION_LIMIT, self.ACTION_LIMIT).astype(np.float64)
        self.data.ctrl[:NUM_JOINTS] = DEFAULT_POSE + action
        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)
        terms = self._reward(action)
        for name, w in self.reward_weights.items():
            terms[name] *= w  # KeyError = typo'd term name, on purpose
        self.last_action = action.copy()
        self._steps += 1

        obs = self.obs_builder.build(self.data, self.last_action, self.command)
        gz = self.obs_builder.projected_gravity(self.data)[2]
        fallen = bool(gz > -0.5 or self.data.qpos[2] < 0.06)
        terminated = fallen
        truncated = self._steps >= self.max_episode_steps
        reward = float(sum(terms.values())) - (5.0 if fallen else 0.0)
        info: dict[str, Any] = {"reward_terms": terms, "fallen": fallen}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=360, width=480)
        self._renderer.update_scene(self.data, camera=-1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class MicroduckStandEnv(MicroduckEnv):
    """MicroduckStand-v0: don't fall over. The Lab 6 task."""


class MicroduckWalkEnv(MicroduckEnv):
    """MicroduckWalk-v0: track a commanded forward velocity (Lab 7).

    The command (obs slots 48:51) carries [vx, 0, 0]; vx is resampled per
    episode from [0, 0.25] m/s (10% of episodes: exactly 0 — standing must
    stay in the repertoire).
    """

    def _sample_command(self) -> None:
        self.command[:] = 0.0
        if self.np_random.random() > 0.1:
            self.command[0] = self.np_random.uniform(0.05, 0.25)

    def _reward(self, action: np.ndarray) -> dict[str, float]:
        gz = self.obs_builder.projected_gravity(self.data)[2]
        vx_world = float(self.data.qvel[0])
        vx_cmd = float(self.command[0])
        joint_vel = self.data.qvel[6:]
        return {
            "upright": 0.5 * float(-gz),
            "track_vx": 2.0 * float(np.exp(-(((vx_world - vx_cmd) / 0.1) ** 2))),
            "energy": -0.01 * float(joint_vel @ joint_vel) * 0.1,
            "action_rate": -0.05
            * float(np.sum(np.abs(action - self.last_action))),
        }
