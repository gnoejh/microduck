"""Drive a pretrained Microduck policy in the interactive viewer.

Usage:
    uv run scripts/play_policy.py [--behavior walk|stand|sitstand|...]

Keys (viewer focused):
    ↑ / ↓     forward / backward velocity command
    ← / →     turn left / right
    S         sit-stand toggle (sitstand behavior)
    0         zero all commands
    R         reset to STAND
(Ctrl + right-drag on the trunk to shove the duck.)
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer

from microduck_sim.obs import VEL_ANG, VEL_BACK, VEL_FWD
from microduck_sim.policy import BEHAVIORS, PolicyRunner

KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior", default="walk", choices=sorted(BEHAVIORS))
    args = parser.parse_args()

    runner = PolicyRunner(args.behavior)
    cmd = runner.command

    def on_key(keycode: int) -> None:
        if keycode == KEY_UP:
            cmd[0] = VEL_FWD if cmd[0] <= 0 else cmd[0]
        elif keycode == KEY_DOWN:
            cmd[0] = VEL_BACK if cmd[0] >= 0 else cmd[0]
        elif keycode == KEY_LEFT:
            cmd[2] = VEL_ANG
        elif keycode == KEY_RIGHT:
            cmd[2] = -VEL_ANG
        elif keycode in (ord("S"), ord("s")):
            cmd[0] = 0.0 if cmd[0] == 1.0 else 1.0  # sitstand posture flag
        elif keycode == ord("0"):
            cmd[:] = 0.0
        elif keycode in (ord("R"), ord("r")):
            runner.reset()
        print(f"command: vx={cmd[0]:+.2f} wz={cmd[2]:+.2f}")

    print(f"behavior={args.behavior}  (50 Hz policy, timestep 5 ms, decimation 4)")
    print(__doc__.split("Keys")[1])

    with mujoco.viewer.launch_passive(
        runner.model, runner.data, key_callback=on_key
    ) as viewer:
        while viewer.is_running():
            t0 = time.time()
            runner.step()
            viewer.sync()
            leftover = 0.02 - (time.time() - t0)
            if leftover > 0:
                time.sleep(leftover)


if __name__ == "__main__":
    main()
