"""Standardized policy evaluation (Lab 8).

With no hardware, evaluation IS our deployment: a policy's grade is how it
behaves across seeds, disturbances, and *model variants* (sim2sim transfer —
domain randomization's exam). All functions accept anything `OnnxPolicy`
accepts: a behavior name, or a path to any contract-compatible ONNX
(shipped, community, or student-trained).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from microduck_sim.policy import OnnxPolicy, PolicyRunner


def evaluate_walking(
    policy: str | Path | OnnxPolicy,
    vx: float = 0.25,  # the shipped walker has a command deadband below ~0.25!
    episodes: int = 5,
    duration: float = 10.0,
    variant: str = "full",
) -> dict[str, float]:
    """Walking scorecard: survival, achieved speed, tracking error.

    Episodes differ by a small random initial joint perturbation (seeded).
    """
    policy = policy if isinstance(policy, OnnxPolicy) else OnnxPolicy(policy)
    survived, speeds = [], []
    for seed in range(episodes):
        runner = PolicyRunner(policy, variant=variant)
        rng = np.random.default_rng(seed)
        runner.data.qpos[7:] += rng.uniform(-0.03, 0.03, runner.model.nq - 7)
        mujoco.mj_forward(runner.model, runner.data)
        runner.command[0] = vx
        x0 = float(runner.data.qpos[0])
        runner.run(duration)
        survived.append(runner.upright)
        speeds.append((float(runner.data.qpos[0]) - x0) / duration)
    speeds = np.array(speeds)
    return {
        "survival_rate": float(np.mean(survived)),
        "mean_speed": float(speeds.mean()),
        "tracking_error": float(np.abs(speeds - vx).mean()),
    }


def max_push(
    policy: str | Path | OnnxPolicy,
    axis: int = 0,
    resolution: float = 0.2,
    limit: float = 6.0,
    variant: str = "full",
) -> float:
    """Largest 0.1 s trunk push (N) the standing policy survives (10 s runs)."""
    policy = policy if isinstance(policy, OnnxPolicy) else OnnxPolicy(policy)
    force = 0.0
    while force + resolution <= limit:
        runner = PolicyRunner(policy, variant=variant)
        trunk = mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        for tick in range(500):
            if tick == 100:
                runner.data.xfrc_applied[trunk, axis] = force + resolution
            if tick == 105:
                runner.data.xfrc_applied[trunk, axis] = 0.0
            runner.step()
            if not runner.upright:
                return force
        force += resolution
    return force


def variant_sweep(
    policy: str | Path | OnnxPolicy,
    vx: float = 0.25,  # the shipped walker has a command deadband below ~0.25!
    variants: tuple[str, ...] = ("full", "walk", "backlash"),
    episodes: int = 3,
) -> dict[str, dict[str, float]]:
    """Sim2sim robustness: the same walking scorecard across model variants.

    A robust (domain-randomized) policy degrades gracefully across variants;
    an overfit one falls apart on the ±1° backlash model.
    """
    return {
        v: evaluate_walking(policy, vx=vx, episodes=episodes, variant=v)
        for v in variants
    }
