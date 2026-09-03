"""Browser playground helpers (Colab-first course platform, M6).

Two halves, deliberately separated:

1. **Viewing** — dependency-free. `show_playground()` embeds the course's static
   mjswan site (built by CI, hosted on GitHub Pages) as an iframe in any notebook,
   Colab included. Nothing here imports mujoco, mjswan or torch at module level.

2. **Building** — the port of the Microduck scene/policy wiring from
   `mjswan_playground` (Apache-2.0). Only `playground/build.py` calls these; they
   import mjswan + mjlab lazily and are meant to run in the isolated
   `playground/` uv project (mujoco 3.10 override), never in the student env.

The observation contract is the same 61-dim vector `obs.py` builds - the slot
order below is derived from those constants so there is one source of truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from .model import ASSETS_DIR, MJCF_DIR, POLICIES_DIR, SCENE_VARIANTS
from .obs import CMD_SIZE, JOINT_NAMES, NUM_JOINTS

# --------------------------------------------------------------------------- viewing

PLAYGROUND_URL = "https://gnoejh.github.io/microduck/"

#: Scene ids in the built playground -> what they are (for docs/tests).
SCENES = {
    "walk": "full-collision duck; walking-layout policies (twist + head sliders)",
    "stand_pose": "shipped stand policy with head + trunk-pose sliders",
    "sit_stand": "shipped sit/stand policy (Sit checkbox)",
    "ground_pick": "shipped ground-pick (4 s phase clock)",
    "roulade": "shipped forward roll",
    "ball_kick_right": "shipped kick, ball in front of the right foot",
    "ball_kick_left": "shipped kick, ball in front of the left foot",
    "roller_skate": "roller variant, throttle + heading sliders",
    "roller_crouch": "roller variant, crouch phase clock",
    "backlash": "walk scene on the +/-1 deg backlash model (Lab 7 dither demo)",
    "pose": "no controller: 14 joint sliders, seated (Lab 2)",
}


def playground_url(
    scene: str = "walk",
    policy: str | None = None,
    *,
    panel: bool = True,
    ref: bool = False,
    base: str | None = None,
) -> str:
    """URL of one scene (and optionally one policy) in the course playground."""
    params: dict[str, str] = {"scene": scene}
    if policy:
        params["policy"] = policy
    if not panel:
        params["panel"] = "0"
    if not ref:
        params["ref"] = "0"
    return (base or PLAYGROUND_URL).rstrip("/") + "/?" + urlencode(params)


def show_playground(
    scene: str = "walk",
    policy: str | None = None,
    *,
    height: int = 600,
    panel: bool = True,
    base: str | None = None,
):
    """Embed the course playground in the notebook (works in Colab and locally)."""
    from IPython.display import IFrame

    return IFrame(
        playground_url(scene, policy, panel=panel, base=base), width="100%", height=height
    )


# --------------------------------------------------------------------------- building
# Everything below is a port of mjswan_playground/microduck/main.py (Apache-2.0),
# reduced to what the course needs and pointed at our vendored assets.

ENTITY = "robot"
ROOT_JOINT = "trunk_base_freejoint"
TRACKED_BODY = "trunk_base"
BALL_JOINT = "ball_free"

CONTROL_DT = 0.02  # 0.005 timestep x decimation 4 = 50 Hz (policy.py)
TIMESTEP = 0.005
SOLVER_ITERATIONS = 10
SOLVER_LS_ITERATIONS = 20

STAND_KEY = "STAND"
STAND_HEIGHT = 0.125  # 5 mm above the keyframe, like upstream's infer script
ROLLERS_STAND_HEIGHT = 0.1385
WHEEL_FRICTIONLOSS = 0.003
BALL_OFFSET_X = 0.08
BALL_OFFSET_ABS_Y = 0.042
BALL_RADIUS = 0.035
FELL_OVER_ANGLE_DEG = 70.0

TWIST_RANGES = ((-0.4, 0.4), (-0.3, 0.3), (-1.0, 1.0))
ROLLER_THROTTLE_RANGE = (-0.5, 0.6)
ROLLER_HEADING_RANGE = (-1.0, 1.0)
HEAD_RANGES = ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))
HEAD_LABELS = ("Neck Pitch", "Head Pitch", "Head Yaw", "Head Roll")
BODY_Z_RANGE = (-0.04, 0.030)
BODY_ANGLE_RANGE = (-math.radians(15), math.radians(15))
POSE_RANGE = (-1.5, 1.5)  # identity "pose policy" slider range (rad, offset from the key)

#: Command block = twist(3) + head(4) + body(6) - must add up to obs.CMD_SIZE.
assert 3 + 4 + 6 == CMD_SIZE


@dataclass(frozen=True)
class PolicySpec:
    """One ONNX policy to attach to a scene."""

    name: str
    onnx: Path
    twist: str = "velocity"  # velocity | heading | posture | phase | zero
    head: bool = False
    body: bool = False
    fell_over: bool = True
    clip_actions: float | None = None
    default: bool = False


@dataclass(frozen=True)
class SceneSpec:
    """One scene: which MJCF variant, how it is posed, and its policies."""

    id: str
    name: str
    variant: str
    policies: tuple[PolicySpec, ...]
    root_height: float = STAND_HEIGHT
    wheels: bool = False
    ball_y: float | None = None
    trace_variant: str | None = None  # the ball scene traces against the ball-less spec
    key: str = STAND_KEY  # the keyframe the scene resets to / actions offset from


def shipped(name: str, file: str, **kw) -> PolicySpec:
    return PolicySpec(name, POLICIES_DIR / file, **kw)


def course_policy(
    name: str, path: str | Path, *, twist: str = "velocity", head: bool = True, **kw
) -> PolicySpec:
    """A policy exported by `export.py` (raw obs, +/-0.6 clip) or by microduck_rl."""
    return PolicySpec(name, Path(path), twist=twist, head=head, **kw)


def _mjcf(variant: str) -> Path:
    return MJCF_DIR / SCENE_VARIANTS[variant]


def _require_build_deps() -> None:
    try:
        import mjlab  # noqa: F401
        import mjswan  # noqa: F401
    except ImportError as exc:  # pragma: no cover - student env
        raise ImportError(
            "Building the playground needs mjswan + mjlab. Run it inside the isolated "
            "build project:  uv run --project playground python playground/build.py"
        ) from exc


def servo_joints(model) -> list[str]:
    """The 14 servos in actuator order (drops the rollers' passive wheel hinges)."""
    import mujoco

    names = []
    for a in range(model.nu):
        if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        names.append(model.joint(int(model.actuator_trnid[a, 0])).name)
    assert names == JOINT_NAMES, names
    return names


def key_pose(variant: str, key: str = STAND_KEY) -> tuple[dict[str, float], float]:
    """(joint positions by name, root height) of a keyframe - upstream's DEFAULT_POSE."""
    import mujoco

    model = mujoco.MjSpec.from_file(str(_mjcf(variant))).compile()
    k = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key)
    if k < 0:
        raise ValueError(f"{variant} has no {key!r} keyframe")
    qpos = model.key_qpos[k]
    pose = {
        model.joint(j).name: float(qpos[model.jnt_qposadr[j]])
        for j in range(model.njnt)
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
    }
    return pose, float(qpos[2])


def _free_joint_adr(model, name: str) -> slice:
    import mujoco

    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if j < 0:
        raise ValueError(f"no joint {name!r}")
    s = int(model.jnt_qposadr[j])
    return slice(s, s + 7)


def scene_spec(
    variant: str,
    pose: dict[str, float],
    *,
    root_height: float,
    wheels: bool = False,
    ball_y: float | None = None,
    sim_options: bool = True,
    key_name: str = STAND_KEY,
):
    """The scene as the browser compiles it: upstream's solver settings, wheel
    friction, and exactly ONE keyframe (the browser and the tracing env reset to
    the first key, and `joint_pos_rel` bakes its pose in)."""
    import mujoco
    import numpy as np

    spec = mujoco.MjSpec.from_file(str(_mjcf(variant)))
    if sim_options:
        spec.option.timestep = TIMESTEP
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.iterations = SOLVER_ITERATIONS
        spec.option.ls_iterations = SOLVER_LS_ITERATIONS
    if wheels:
        for joint in spec.joints:
            if joint.name.startswith("passive_"):
                joint.frictionloss = WHEEL_FRICTIONLOSS

    model = spec.compile()
    qpos = np.array(model.qpos0, dtype=float)
    qpos[_free_joint_adr(model, ROOT_JOINT)] = [0.0, 0.0, root_height, 1.0, 0.0, 0.0, 0.0]
    for j in range(model.njnt):
        name = model.joint(j).name
        if name in pose:
            qpos[model.jnt_qposadr[j]] = pose[name]
    if ball_y is not None:
        qpos[_free_joint_adr(model, BALL_JOINT)] = [
            BALL_OFFSET_X, ball_y, BALL_RADIUS, 1.0, 0.0, 0.0, 0.0,
        ]
    ctrl = [pose[n] for n in servo_joints(model)]
    for key in list(spec.keys):
        spec.delete(key)
    spec.add_key(name=key_name, qpos=qpos.tolist(), ctrl=ctrl)
    return spec


def _sliders(specs):
    import mjswan

    return mjswan.ui_command(
        [
            mjswan.SliderConfig(
                name=name,
                label=label,
                range=rng,
                default=0.0,
                step=round((rng[1] - rng[0]) / 40.0, 4),
            )
            for name, label, rng in specs
        ]
    )


def twist_commands(kind: str) -> dict[str, Any]:
    import mjswan
    from mjswan_playground.microduck import terms

    if kind == "velocity":
        return {
            "twist": _sliders(
                [
                    ("lin_vel_x", "Forward (m/s)", TWIST_RANGES[0]),
                    ("lin_vel_y", "Sideways (m/s)", TWIST_RANGES[1]),
                    ("ang_vel_z", "Turn (rad/s)", TWIST_RANGES[2]),
                ]
            )
        }
    if kind == "heading":
        return {
            "throttle": _sliders([("lin_vel_x", "Forward (m/s)", ROLLER_THROTTLE_RANGE)]),
            "heading": _sliders([("heading_error", "Error (rad)", ROLLER_HEADING_RANGE)]),
        }
    if kind == "posture":
        return {
            "posture": mjswan.ui_command(
                [mjswan.CheckboxConfig(name="sit", label="Sit", default=False)]
            )
        }
    if kind == "phase":
        return {"twist": terms.GroundPickPhaseCommandCfg(control_dt=CONTROL_DT)}
    if kind != "zero":
        raise ValueError(f"unknown twist kind {kind!r}")
    return {}


def _twist_terms(kind: str) -> dict[str, Any]:
    from mjlab.envs.mdp import observations as obs_fns
    from mjswan.managers.observation_manager import ObservationTermCfg as T
    from mjswan_playground.microduck import terms

    gen = obs_fns.generated_commands
    if kind in ("velocity", "phase"):
        return {"twist": T(func=gen, params={"command_name": "twist"})}
    if kind == "heading":
        return {
            "throttle": T(func=gen, params={"command_name": "throttle"}),
            "lin_vel_y": T(func=terms.zeros, params={"dim": 1}),
            "heading": T(func=gen, params={"command_name": "heading"}),
        }
    if kind == "posture":
        return {
            "posture": T(func=gen, params={"command_name": "posture"}),
            "twist_pad": T(func=terms.zeros, params={"dim": 2}),
        }
    return {"twist": T(func=terms.zeros, params={"dim": 3})}


def pose_commands(head: bool, body: bool) -> dict[str, Any]:
    cmds: dict[str, Any] = {}
    if head:
        cmds["head_pose"] = _sliders(
            list(zip(("neck_pitch", "head_pitch", "head_yaw", "head_roll"), HEAD_LABELS, HEAD_RANGES))
        )
    if body:
        cmds["body_pose"] = _sliders(
            [
                ("z", "Trunk Height (m)", BODY_Z_RANGE),
                ("roll", "Trunk Roll (rad)", BODY_ANGLE_RANGE),
                ("pitch", "Trunk Pitch (rad)", BODY_ANGLE_RANGE),
            ]
        )
    return cmds


def _pose_terms(head: bool, body: bool) -> dict[str, Any]:
    from mjlab.envs.mdp import observations as obs_fns
    from mjswan.managers.observation_manager import ObservationTermCfg as T
    from mjswan_playground.microduck import terms

    gen = obs_fns.generated_commands
    out: dict[str, Any] = {}
    if head:
        out["head_command"] = T(func=gen, params={"command_name": "head_pose"})
    else:
        out["head_command"] = T(func=terms.zeros, params={"dim": 4})
    if body:
        out["body_xy"] = T(func=terms.zeros, params={"dim": 2})
        out["body_command"] = T(func=gen, params={"command_name": "body_pose"})
        out["body_yaw"] = T(func=terms.zeros, params={"dim": 1})
    else:
        out["body_command"] = T(func=terms.zeros, params={"dim": 6})
    return out


def microduck_obs_config(twist: str, head: bool, body: bool, joints):
    """The 61 values, in `obs.py` order: gyro 3, gravity 3, qpos 14, qvel 14,
    last action 14, command 13 = twist(3) + head(4) + body(6)."""
    from mjlab.envs.mdp import observations as obs_fns
    from mjswan.managers.observation_manager import ObservationGroupCfg
    from mjswan.managers.observation_manager import ObservationTermCfg as T

    group = {
        "base_ang_vel": T(func=obs_fns.base_ang_vel),
        "projected_gravity": T(func=obs_fns.projected_gravity),
        "joint_pos": T(func=obs_fns.joint_pos_rel, params={"asset_cfg": joints}),
        "joint_vel": T(func=obs_fns.joint_vel_rel, params={"asset_cfg": joints}),
        "actions": T(func=obs_fns.last_action),
    }
    group.update(_twist_terms(twist))
    group.update(_pose_terms(head, body))
    return ObservationGroupCfg(terms=group)


def trace_commands(twist: str, head: bool, body: bool) -> dict[str, Any]:
    from mjswan_playground.microduck import terms

    widths = {
        "velocity": {"twist": 3},
        "phase": {"twist": 3},
        "heading": {"throttle": 1, "heading": 1},
        "posture": {"posture": 1},
    }
    stand_ins = {n: terms.CommandValues(w) for n, w in widths.get(twist, {}).items()}
    if head:
        stand_ins["head_pose"] = terms.CommandValues(4)
    if body:
        stand_ins["body_pose"] = terms.CommandValues(3)
    return stand_ins


def add_scene(project, scene: SceneSpec, *, viewer: bool = True):
    """Add one course scene with all its policies to an mjswan project.

    All policies on a scene must share one command layout (mjswan 0.9.3 writes a
    single fused observation graph per scene). Enforced here: every policy must
    match the scene's first policy.
    """
    _require_build_deps()
    import mjswan
    import onnx
    from mjlab.envs.mdp import terminations as term_fns
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjswan.envs.mdp.actions import JointPositionActionCfg
    from mjswan.managers.termination_manager import TerminationTermCfg
    from mjswan.trace_env import build_single_entity_trace_env

    pose, _ = key_pose("full" if scene.variant == "ball" else scene.variant, scene.key)
    spec = scene_spec(
        scene.variant, pose, root_height=scene.root_height, wheels=scene.wheels,
        ball_y=scene.ball_y, key_name=scene.key,
    )
    joint_names = servo_joints(spec.compile())
    joints = SceneEntityCfg(name=ENTITY, joint_names=tuple(joint_names), preserve_order=True)

    handle = project.add_scene(name=scene.name, spec=spec, control_dt=CONTROL_DT)
    if viewer:
        handle.set_viewer(
            mjswan.ViewerConfig(
                origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
                body_name=TRACKED_BODY, distance=0.8, elevation=-12.0, azimuth=40.0,
            )
        )

    first = scene.policies[0]
    layout = (first.twist, first.head, first.body)
    for p in scene.policies:
        if (p.twist, p.head, p.body) != layout:
            raise ValueError(
                f"policy {p.name!r} has a different command layout than scene {scene.id!r}"
            )

    def trace_spec_fn():
        tv = scene.trace_variant or scene.variant
        return scene_spec(
            tv, pose, root_height=scene.root_height, wheels=scene.wheels,
            ball_y=None if scene.trace_variant else scene.ball_y,
            sim_options=False, key_name=scene.key,
        )

    handle.set_trace_env(
        build_single_entity_trace_env(
            trace_spec_fn, entity_name=ENTITY, commands=trace_commands(*layout)
        )
    )

    for p in scene.policies:
        terminations = {}
        if p.fell_over:
            terminations["fell_over"] = TerminationTermCfg(
                func=term_fns.bad_orientation,
                params={"limit_angle": math.radians(FELL_OVER_ANGLE_DEG)},
            )
        handle.add_policy(
            name=p.name,
            policy=onnx.load(str(p.onnx)),
            commands={**twist_commands(p.twist), **pose_commands(p.head, p.body)},
            observations=microduck_obs_config(p.twist, p.head, p.body, joints),
            actions={
                "joint_pos": JointPositionActionCfg(
                    entity_name="", actuator_names=(".*",), scale=1.0, use_default_offset=True
                )
            },
            terminations=terminations,
            policy_joint_names=joint_names,
            default_joint_pos=[pose[n] for n in joint_names],
            clip_actions=p.clip_actions,
            default=p.default,
        )
    return handle


def add_pose_scene(
    project, *, variant: str = "full", key: str = "SIT", onnx_path: Path | None = None
):
    """The Lab-2 scene: no controller, 14 sliders -> joint targets (offsets from `key`).

    The 'policy' is the identity ONNX from playground/make_pose_policy.py: it reads
    ONLY the 14-wide 'pose' command and returns it as the action.
    """
    _require_build_deps()
    import mjswan
    import onnx
    from mjlab.envs.mdp import observations as obs_fns
    from mjswan.envs.mdp.actions import JointPositionActionCfg
    from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjswan.trace_env import build_single_entity_trace_env
    from mjswan_playground.microduck import terms

    pose, z = key_pose(variant, key)
    spec = scene_spec(variant, pose, root_height=z + 0.005, key_name=key)
    joint_names = servo_joints(spec.compile())
    handle = project.add_scene(name="Pose", spec=spec, control_dt=CONTROL_DT)
    handle.set_viewer(
        mjswan.ViewerConfig(
            origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
            body_name=TRACKED_BODY, distance=0.8, elevation=-12.0, azimuth=40.0,
        )
    )
    handle.set_trace_env(
        build_single_entity_trace_env(
            lambda: scene_spec(variant, pose, root_height=z + 0.005, sim_options=False, key_name=key),
            entity_name=ENTITY,
            commands={"pose": terms.CommandValues(NUM_JOINTS)},
        )
    )
    handle.add_policy(
        name="Identity",
        policy=onnx.load(str(onnx_path or ASSETS_DIR / "checkpoints" / "identity_pose.onnx")),
        commands={"pose": _sliders([(n, n.replace("_", " "), POSE_RANGE) for n in joint_names])},
        observations=ObservationGroupCfg(
            terms={
                "pose": ObservationTermCfg(
                    func=obs_fns.generated_commands, params={"command_name": "pose"}
                )
            }
        ),
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="", actuator_names=(".*",), scale=1.0, use_default_offset=True
            )
        },
        terminations={},
        policy_joint_names=joint_names,
        default_joint_pos=[pose[n] for n in joint_names],
        default=True,
    )
    return handle
