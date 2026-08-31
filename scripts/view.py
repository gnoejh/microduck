"""Open the Microduck in the interactive MuJoCo viewer.

Usage:
    uv run scripts/view.py [--variant walk|full|rollers|walk_backlash|backlash|ball]

Keys (with the viewer window focused):
    SPACE  cycle keyframe pose (INIT -> STAND -> SIT -> FOLD)
    R      reset to current keyframe
(plus all standard MuJoCo viewer controls: drag to orbit, scroll to zoom,
double-click to select, Ctrl+right-drag to apply a push force)
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer

from microduck_sim.model import KEYFRAME_NAMES, SCENE_VARIANTS, load_model, reset_to_keyframe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", default="walk", choices=sorted(SCENE_VARIANTS), help="scene to load"
    )
    args = parser.parse_args()

    model = load_model(args.variant)
    data = mujoco.MjData(model)

    state = {"key_index": 1}  # start at STAND
    reset_to_keyframe(model, data, KEYFRAME_NAMES[state["key_index"]])

    def on_key(keycode: int) -> None:
        if keycode == ord(" "):
            state["key_index"] = (state["key_index"] + 1) % len(KEYFRAME_NAMES)
        elif keycode not in (ord("R"), ord("r")):
            return
        name = KEYFRAME_NAMES[state["key_index"]]
        reset_to_keyframe(model, data, name)
        print(f"pose: {name}")

    print(f"variant={args.variant}  nq={model.nq}  nu={model.nu}  "
          f"timestep={model.opt.timestep}s")
    print("SPACE = cycle pose, R = reset pose")

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            # real-time pacing
            leftover = model.opt.timestep - (time.time() - step_start)
            if leftover > 0:
                time.sleep(leftover)


if __name__ == "__main__":
    main()
