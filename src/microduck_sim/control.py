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


def quat_to_pitch_roll(quat: np.ndarray) -> tuple[float, float]:
    """Trunk pitch and roll [rad] from a wxyz quaternion (e.g. ``data.qpos[3:7]``)."""
    w, x, y, z = quat
    pitch = float(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
    roll = float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    return pitch, roll


def make_balance_controller(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    base_ctrl: np.ndarray,
    kp_pitch: float = 1.0,
    kd_pitch: float = 0.1,
    pitch_fn: Callable[[], float] | None = None,
    rate_fn: Callable[[], float] | None = None,
) -> CtrlFn:
    """Ankle-strategy standing balance (built by students in Lab 3).

    Reads trunk pitch + pitch rate each tick and offsets the ankle targets:
    ``left_ankle += kp*pitch + kd*rate``, ``right_ankle -=`` the same (the
    ankles mirror). Defaults read ground truth from `data`; pass `pitch_fn` /
    `rate_fn` to use estimated state instead (Lab 4).

    Verified gains: kp=1.0, kd=0.1 stands indefinitely (baseline falls at
    ~0.8 s); flipped signs make it fall FASTER than no feedback.
    """
    from microduck_sim.model import actuator_names

    acts = actuator_names(model)
    left, right = acts.index("left_ankle"), acts.index("right_ankle")
    base_ctrl = np.asarray(base_ctrl, dtype=float).copy()

    def fn(t: float) -> np.ndarray:
        pitch = pitch_fn() if pitch_fn else quat_to_pitch_roll(data.qpos[3:7])[0]
        rate = rate_fn() if rate_fn else float(data.qvel[4])
        correction = kp_pitch * pitch + kd_pitch * rate
        ctrl = base_ctrl.copy()
        ctrl[left] += correction
        ctrl[right] -= correction
        return ctrl

    return fn


def make_push(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    force: float,
    axis: int = 0,
    t_start: float = 2.0,
    duration: float = 0.1,
    body: str = "trunk_base",
) -> Callable[[mujoco.MjModel, mujoco.MjData], None]:
    """An ``on_tick`` callback applying a horizontal push to a body.

    axis 0 = forward (+x), 1 = sideways (+y). Times are relative to the tick
    when the returned callback first fires. Remember to zero
    ``data.xfrc_applied`` after the run if you reuse `data`.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if body_id < 0:
        raise ValueError(f"body {body!r} not found")
    start = {"t0": None}  # latched on the first tick, so resets in between are safe

    def on_tick(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        if start["t0"] is None:
            start["t0"] = d.time
        active = t_start <= (d.time - start["t0"]) < t_start + duration
        d.xfrc_applied[body_id, axis] = force if active else 0.0

    return on_tick


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
