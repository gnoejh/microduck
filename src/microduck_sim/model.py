"""Model loading utilities for the Microduck (Lab 1).

The MJCF files in ``assets/mjcf`` are vendored from Pollen Robotics'
``microduck_rl`` (see ``assets/ATTRIBUTION.md``). All scene variants define the
keyframes INIT / STAND / SIT / FOLD.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

# Repo-root-relative paths; valid because the package is installed editable
# (the file stays under src/). If this ever moves to a wheel, switch to
# importlib.resources.
ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
MJCF_DIR = ASSETS_DIR / "mjcf"
POLICIES_DIR = ASSETS_DIR / "policies"

# Scene entry points (each includes the matching robot_*.xml).
SCENE_VARIANTS = {
    "walk": "scene_walk.xml",  # 14 leg/neck/head actuators — default for labs
    "full": "scene.xml",  # all collisions + beak, for manipulation tasks
    "rollers": "scene_rollers.xml",  # wheeled roller-skating variant
    "walk_backlash": "scene_walk_backlash.xml",  # ±1° gear play (robustness labs)
    "backlash": "scene_backlash.xml",
    "ball": "scene_ball.xml",  # full robot + a ball to kick
}

KEYFRAME_NAMES = ("INIT", "STAND", "SIT", "FOLD")


def load_model(variant: str = "walk") -> mujoco.MjModel:
    """Load a Microduck scene by variant name (see SCENE_VARIANTS)."""
    try:
        filename = SCENE_VARIANTS[variant]
    except KeyError:
        raise ValueError(
            f"Unknown variant {variant!r}; choose from {sorted(SCENE_VARIANTS)}"
        ) from None
    return mujoco.MjModel.from_xml_path(str(MJCF_DIR / filename))


def load(variant: str = "walk") -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load a scene and a fresh MjData, reset to the STAND keyframe."""
    model = load_model(variant)
    data = mujoco.MjData(model)
    reset_to_keyframe(model, data, "STAND")
    return model, data


def reset_to_keyframe(
    model: mujoco.MjModel, data: mujoco.MjData, name: str = "STAND"
) -> None:
    """Reset the simulation state (qpos/qvel/ctrl) to a named keyframe."""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if key_id < 0:
        available = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
            for i in range(model.nkey)
        ]
        raise ValueError(f"Keyframe {name!r} not found; model has {available}")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)


def joint_names(model: mujoco.MjModel) -> list[str]:
    """Names of all joints (index = joint id)."""
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]


def actuator_names(model: mujoco.MjModel) -> list[str]:
    """Names of all actuators (index = actuator id = ctrl index)."""
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]
