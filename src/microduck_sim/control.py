"""Control-loop and trajectory helpers (Labs 2–3).

The real Microduck runs its policy at 50 Hz while MuJoCo integrates physics at
500 Hz (timestep 0.002 s), so one control tick = 10 physics substeps. Every
helper here follows that contract.
"""

from __future__ import annotations

from typing import Callable

import mujoco
import numpy as np

CONTROL_HZ = 50

CtrlFn = Callable[[float], np.ndarray]


def keyframe_ctrl(model: mujoco.MjModel, name: str) -> np.ndarray:
    """Return a copy of the ctrl vector stored in a named keyframe."""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if key_id < 0:
        raise ValueError(f"Keyframe {name!r} not found")
    return model.key_ctrl[key_id].copy()


def lerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Linear interpolation between two ctrl vectors; alpha is clipped to [0, 1]."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - alpha) * np.asarray(a) + alpha * np.asarray(b)


def constant(ctrl: np.ndarray) -> CtrlFn:
    """A ctrl function that holds one vector forever."""
    ctrl = np.asarray(ctrl, dtype=float).copy()
    return lambda t: ctrl


def goto(start: np.ndarray, target: np.ndarray, duration: float) -> CtrlFn:
    """A ctrl function moving smoothly from start to target in `duration` seconds.

    Uses a minimum-jerk profile (smooth start and stop) rather than plain lerp,
    which is what real robot joint ramps use.
    """
    start = np.asarray(start, dtype=float).copy()
    target = np.asarray(target, dtype=float).copy()

    def fn(t: float) -> np.ndarray:
        s = np.clip(t / duration, 0.0, 1.0)
        alpha = 10 * s**3 - 15 * s**4 + 6 * s**5  # min-jerk
        return lerp(start, target, alpha)

    return fn


def sine(
    center: np.ndarray,
    amplitude: np.ndarray,
    frequency: float,
    phase: float = 0.0,
) -> CtrlFn:
    """A ctrl function oscillating around `center` (amplitude may be per-joint)."""
    center = np.asarray(center, dtype=float).copy()
    amplitude = np.asarray(amplitude, dtype=float).copy()

    def fn(t: float) -> np.ndarray:
        return center + amplitude * np.sin(2 * np.pi * frequency * t + phase)

    return fn


def chain(*segments: tuple[float, CtrlFn]) -> CtrlFn:
    """Sequence (duration, ctrl_fn) segments into one ctrl function.

    Each segment's fn receives its LOCAL time (starting at 0). After the last
    segment ends, its final value is held.
    """
    if not segments:
        raise ValueError("chain() needs at least one segment")

    def fn(t: float) -> np.ndarray:
        remaining = t
        for duration, seg_fn in segments:
            if remaining <= duration:
                return seg_fn(remaining)
            remaining -= duration
        last_duration, last_fn = segments[-1]
        return last_fn(last_duration)

    return fn


def run_control_loop(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl_fn: CtrlFn,
    duration: float,
    control_hz: int = CONTROL_HZ,
    on_tick: Callable[[mujoco.MjModel, mujoco.MjData], None] | None = None,
) -> dict[str, np.ndarray]:
    """Run `ctrl_fn` at `control_hz` for `duration` seconds; return a log.

    ctrl_fn(t) -> ctrl vector, with t the elapsed time since this call started.
    Returns {"t", "qpos", "qvel", "ctrl", "site_xpos"} as stacked arrays
    (one row per control tick).
    """
    substeps = round(1.0 / (control_hz * model.opt.timestep))
    if substeps < 1:
        raise ValueError(
            f"control_hz={control_hz} is faster than the physics timestep allows"
        )
    n_ticks = int(round(duration * control_hz))
    t0 = data.time
    log: dict[str, list] = {"t": [], "qpos": [], "qvel": [], "ctrl": [], "site_xpos": []}
    for _ in range(n_ticks):
        data.ctrl[:] = ctrl_fn(data.time - t0)
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        log["t"].append(data.time - t0)
        log["qpos"].append(data.qpos.copy())
        log["qvel"].append(data.qvel.copy())
        log["ctrl"].append(data.ctrl.copy())
        log["site_xpos"].append(data.site_xpos.copy())
        if on_tick is not None:
            on_tick(model, data)
    return {k: np.asarray(v) for k, v in log.items()}
