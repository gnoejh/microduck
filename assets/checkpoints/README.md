# Committed fallback checkpoints

Instructor-trained SB3 PPO checkpoints so labs never block on a failed student
training run. Regenerate with `scripts/train_sb3.py` (settings below); each has
its `*_progress.csv` learning curve alongside, used by the lecture notebooks.

| files | task | recipe |
|-------|------|--------|
| `stand_ppo.zip`, `stand_ppo.onnx`, `stand_progress.csv` | MicroduckStand-v0 | 1M steps, 8 envs, defaults (`--log-std-init -1`) |
| `walk_ppo.zip`, `walk_ppo.onnx`, `walk_progress.csv` | MicroduckWalk-v0 | 2M steps, 8 envs, defaults |

The `.onnx` files follow the shipped-policy contract (61 obs → 14 actions) and
run in `PolicyRunner` / `scripts/play_policy.py` like any official behavior.
