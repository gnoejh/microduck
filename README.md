# Microduck Simulation Lab 🦆

A **simulation-only** university lab course built around the
[Pollen Robotics Microduck](https://pollen-robotics.com/microduck/) biped —
8 sessions from "what is a joint" to "train, evaluate and publish your own
neural-network policy", entirely on a **laptop CPU**. No robot required, ever.

The course replays the **actual policies shipped with the real robot** (61-dim
observation contract, 50 Hz) in plain MuJoCo, and everything students train
speaks the same contract — interchangeable with the official and community
checkpoints on the Hugging Face Hub.

## Setup (students)

Requirements: Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/). Windows,
macOS and Linux all work.

```bash
git clone https://github.com/gnoejh/microduck.git
cd microduck
uv sync                      # creates .venv with everything pinned
uv run pytest -q             # ~50 green tests = your machine is ready
uv run scripts/view.py       # meet the duck (SPACE cycles poses)
uv run scripts/play_policy.py  # drive the shipped walking policy (arrow keys)
```

Then open the labs: `uv run jupyter lab labs/`

## The 8 labs

| # | directory | topic |
|---|-----------|-------|
| 1 | `labs/lab01_meet_the_duck` | MuJoCo basics, poses, the open-loop fall |
| 2 | `labs/lab02_kinematics` | trajectories, FK, rhythm (and honest failures) |
| 3 | `labs/lab03_feedback_control` | PD balance from scratch, push limits |
| 4 | `labs/lab04_sensors_estimation` | IMU, noise, complementary filter, balancing blind |
| 5 | `labs/lab05_trained_policy` | run the robot's real neural networks |
| 6 | `labs/lab06_first_rl` | Gymnasium, rewards, PPO — train standing |
| 7 | `labs/lab07_reward_shaping` | reward audits, the walking task, deadbands |
| 8 | `labs/lab08_evaluation` | protocols, sim2sim robustness, publish to the Hub |

Each lab: `lecture.ipynb` (doubles as the slides) + `lab.ipynb` (your
worksheet, with self-checking asserts) + `solution.ipynb` (instructors).
See `labs/README.md` for the authoring workflow.

## GPU track (optional)

`stage2/` — Google Colab notebook (free T4) and RTX-4060 notes for the
official [`microduck_rl`](https://github.com/pollen-robotics/microduck_rl)
pipeline (mjlab/MuJoCo Warp, 1024–4096 parallel ducks). Optional for
students; the second half-semester (NVIDIA Isaac Sim) builds on the same ideas.

## Layout

- `src/microduck_sim/` — the course library, built up lab by lab
- `assets/mjcf/` — robot models (vendored from Pollen, CC BY-SA-NC — see `assets/ATTRIBUTION.md`)
- `assets/policies/` — the 9 shipped ONNX behaviors
- `assets/checkpoints/` — instructor-trained SB3 fallbacks
- `scripts/` — `view.py`, `play_policy.py`, `train_sb3.py`
- `CLAUDE.md` — development plan + progress log
