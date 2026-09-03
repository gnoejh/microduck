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


# --------------------------------------------------------------------------- Lab 8 submissions
# A submission is a folder assets/community/<handle>/ with policy.onnx,
# manifest.json (schema below) and README.md. CI validates it, then the playground
# rebuild puts the policy in the browser for everyone.

MANIFEST_SCHEMA_VERSION = 2


def make_manifest(
    name: str,
    *,
    eval: dict,
    known_limits: str,
    scene: str = "walk",
    twist: str = "velocity",
    training: dict | None = None,
    kind: str = "perpetual",
    entry_pose: str = "standing",
    command: dict | None = None,
    clip_actions: float = 0.6,
) -> dict:
    """The manifest.json every Lab 8 release carries (same schema as the
    community checkpoints on the Hub, plus `scene`/`twist`/`clip_actions`,
    which the playground build reads)."""
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "model_api": 1,
        "name": name,
        "kind": kind,
        "obs_len": 61,
        "action_len": 14,
        "action_scale": 1.0,
        "clip_actions": clip_actions,
        "entry_pose": entry_pose,
        "scene": scene,
        "twist": twist,
        "command": command or {
            "twist": ["vx m/s in [0, 0.25]", "unused (0)", "unused (0)"],
            "head": "unused (zeros)",
            "idle": [0, 0, 0],
        },
        "robot": {"model": "microduck", "servos": "xl330", "control_hz": 50},
        "training": training or {},
        "eval": {**eval, "known_limits": known_limits},
    }


def validate_submission(folder: str | Path, *, quick: bool = True) -> dict:
    """Check a submission folder; raises AssertionError with a clear message.

    Returns the manifest. With ``quick=False`` also runs a 5 s stand check (or
    the walking scorecard for ``twist == "velocity"``) and asserts survival.
    """
    folder = Path(folder)
    onnx_path, manifest_path = folder / "policy.onnx", folder / "manifest.json"
    assert onnx_path.exists(), f"{folder}: policy.onnx missing"
    assert manifest_path.exists(), f"{folder}: manifest.json missing"
    assert (folder / "README.md").exists(), f"{folder}: README.md missing"
    assert onnx_path.stat().st_size < 5_000_000, f"{folder}: policy.onnx > 5 MB"

    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("schema_version", "name", "obs_len", "action_len", "eval"):
        assert key in manifest, f"{folder}: manifest lacks {key!r}"
    assert manifest["obs_len"] == 61 and manifest["action_len"] == 14, f"{folder}: wrong contract dims"
    assert "known_limits" in manifest["eval"], f"{folder}: eval.known_limits missing (be honest!)"
    assert manifest.get("scene", "walk") in {"walk"}, f"{folder}: only the 'walk' scene accepts submissions"

    policy = OnnxPolicy(onnx_path)
    inp, out = policy.session.get_inputs()[0], policy.session.get_outputs()[0]
    assert list(inp.shape)[-1] == 61, f"{folder}: ONNX input must be [1, 61], got {inp.shape}"
    assert list(out.shape)[-1] == 14, f"{folder}: ONNX output must be [1, 14], got {out.shape}"

    if not quick:
        if manifest.get("twist", "velocity") == "velocity" and manifest.get("kind") == "perpetual":
            score = evaluate_walking(policy, episodes=2, duration=5.0)
            assert score["survival_rate"] > 0, f"{folder}: falls immediately when walking"
        else:
            runner = PolicyRunner(policy)
            runner.run(5.0)
            assert runner.upright, f"{folder}: falls within 5 s while standing"
    return manifest
