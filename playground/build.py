"""Build the course's browser playground (static mjswan site).

Run inside the isolated build project (mujoco 3.10 override, mjswan + mjlab):

    uv run --project playground python playground/build.py --out playground/dist
    uv run --project playground python playground/build.py --scenes walk,pose --serve
    uv run --project playground python playground/build.py --policy runs/my_stand.onnx:My duck

Scenes (ids = mjswan's slug of the name):
    walk          full-collision duck, walking layout: shipped walk + course checkpoints
                  + every assets/community/* submission
    stand_pose, sit_stand, ground_pick, roulade, ball_kick_left/right,
    roller_skate, roller_crouch     the other shipped behaviours (upstream layouts)
    backlash      the walk scene on the +/-1 deg backlash model (Lab 7 dither demo)
    pose          no controller, 14 joint sliders, seated (Lab 2)

Windows: run with PYTHONUTF8=1 (mjswan writes generated files without an encoding).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "assets"
CHECKPOINTS = ASSETS / "checkpoints"
COMMUNITY = ASSETS / "community"


def shipped_scenes():
    from microduck_sim.web import (
        BALL_OFFSET_ABS_Y, ROLLERS_STAND_HEIGHT, PolicySpec, SceneSpec, shipped,
    )

    return {
        "stand_pose": SceneSpec("stand_pose", "Stand & Pose", "full", (
            shipped("Stand & Pose", "BEST_alpha_stand.onnx", twist="zero", head=True, body=True, fell_over=False, default=True),)),
        "sit_stand": SceneSpec("sit_stand", "Sit / Stand", "full", (
            shipped("Sit / Stand", "BEST_alpha_sitstand.onnx", twist="posture", head=True, fell_over=False, default=True),)),
        "ground_pick": SceneSpec("ground_pick", "Ground Pick", "full", (
            shipped("Ground Pick", "alpha_ground_pick.onnx", twist="phase", default=True),)),
        "roulade": SceneSpec("roulade", "Roulade", "full", (
            shipped("Roulade", "roulade.onnx", twist="zero", fell_over=False, default=True),)),
        "ball_kick_right": SceneSpec("ball_kick_right", "Ball Kick (right)", "ball", (
            shipped("Ball Kick (right)", "ball_kick_right.onnx", twist="zero", default=True),),
            ball_y=-BALL_OFFSET_ABS_Y, trace_variant="full"),
        "ball_kick_left": SceneSpec("ball_kick_left", "Ball Kick (left)", "ball", (
            shipped("Ball Kick (left)", "ball_kick_left.onnx", twist="zero", default=True),),
            ball_y=BALL_OFFSET_ABS_Y, trace_variant="full"),
        "roller_skate": SceneSpec("roller_skate", "Roller Skate", "rollers", (
            shipped("Roller Skate", "BEST_roller.onnx", twist="heading", default=True),),
            root_height=ROLLERS_STAND_HEIGHT, wheels=True),
        "roller_crouch": SceneSpec("roller_crouch", "Roller Crouch", "rollers", (
            shipped("Roller Crouch", "BEST_roller_crouch.onnx", twist="phase", default=True),),
            root_height=ROLLERS_STAND_HEIGHT, wheels=True),
    }


def walking_policies(extra: list[tuple[str, Path]], community: bool):
    """Everything that speaks the walking layout (twist + head sliders)."""
    from microduck_sim.web import course_policy, shipped

    policies = [
        shipped("Walk", "BEST_alpha_walking.onnx", twist="velocity", head=True, default=True),
        shipped("Stand", "BEST_alpha_stand.onnx", twist="velocity", head=True, fell_over=False),
    ]
    for name, file in (("Stand PPO (course, CPU)", "stand_ppo.onnx"), ("Walk PPO (course, CPU)", "walk_ppo.onnx"),
                       ("Walk mjlab (course, T4)", "mjlab_walk.onnx")):
        if (CHECKPOINTS / file).exists():
            policies.append(course_policy(name, CHECKPOINTS / file, clip_actions=0.6))
    if community and COMMUNITY.exists():
        for sub in sorted(COMMUNITY.iterdir()):
            onnx_path, manifest = sub / "policy.onnx", sub / "manifest.json"
            if not onnx_path.exists():
                continue
            meta = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
            policies.append(course_policy(meta.get("name", sub.name), onnx_path,
                                          clip_actions=meta.get("clip_actions", 0.6)))
    for name, path in extra:
        policies.append(course_policy(name, path, clip_actions=0.6))
    return tuple(policies)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).parent / "dist"))
    ap.add_argument("--scenes", default="all", help="comma-separated scene ids, or 'all'")
    ap.add_argument("--policy", action="append", default=[], metavar="PATH[:NAME]",
                    help="extra walking-layout ONNX to add to the walk scene (repeatable)")
    ap.add_argument("--no-community", action="store_true")
    ap.add_argument("--base-path", default="/", help="URL base for subdirectory hosting, e.g. /microduck/")
    ap.add_argument("--serve", action="store_true", help="serve the build after writing it")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    if os.name == "nt" and os.environ.get("PYTHONUTF8") != "1":
        sys.exit("On Windows run with PYTHONUTF8=1 (mjswan writes generated files in the locale encoding).")

    import mjswan
    from microduck_sim.web import SceneSpec, add_pose_scene, add_scene

    extra = []
    for item in args.policy:
        path, _, name = item.partition(":")
        extra.append((name or Path(path).stem, Path(path).resolve()))

    walking = walking_policies(extra, community=not args.no_community)
    scenes = {
        "walk": SceneSpec("walk", "Walk", "full", walking),
        "backlash": SceneSpec("backlash", "Backlash", "backlash", walking[:1]),
        **shipped_scenes(),
    }
    wanted = list(scenes) + ["pose"] if args.scenes == "all" else args.scenes.split(",")

    builder = mjswan.Builder(base_path=args.base_path)
    project = builder.add_project(name="Microduck Course")
    for sid in wanted:
        if sid == "pose":
            add_pose_scene(project)
        else:
            add_scene(project, scenes[sid])

    out = Path(args.out).resolve()
    app = builder.build(output_dir=out)
    attribution = ASSETS / "ATTRIBUTION.md"
    if attribution.exists():
        shutil.copy(attribution, out / "ATTRIBUTION.md")
    print(f"built {len(wanted)} scene(s) -> {out}")
    if args.serve:
        app.launch(port=args.port, open_browser=False)


if __name__ == "__main__":
    main()
