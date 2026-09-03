# Microduck Simulation Lab 🦆

A **simulation-only** university lab course built around the
[Pollen Robotics Microduck](https://pollen-robotics.com/microduck/) biped —
8 sessions from "what is a joint" to "train, evaluate and publish your own
neural-network policy". No robot required, ever; no GPU required either.

The course replays the **actual policies shipped with the real robot** (61-dim
observation contract, 50 Hz) in plain MuJoCo, and everything students train
speaks the same contract — interchangeable with the official and community
checkpoints, in Python and in the browser.

## Run it in Colab (students)

Every notebook has an *Open in Colab* badge; its first cell installs the course
package. Interactive 3D comes from the **course playground** — MuJoCo compiled to
WebAssembly plus the trained networks, running in your browser tab, embedded in the
notebooks:

- playground: <https://gnoejh.github.io/microduck/> (built by CI from this repo)
- start here: [Lab 1 lecture](https://colab.research.google.com/github/gnoejh/microduck/blob/main/labs/lab01_meet_the_duck/lecture.ipynb)

Training cells are budgeted for Colab's free CPU runtime (≤30 min, with
instructor-trained fallbacks in `assets/checkpoints/`). Lab 7's optional GPU track
uses a free T4 (`stage2/colab_gpu_track.ipynb`).

## Run it on a laptop (optional)

Requirements: Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/). Windows, macOS
and Linux all work.

```bash
git clone https://github.com/gnoejh/microduck.git
cd microduck
uv sync                      # creates .venv with everything pinned
uv run pytest -q             # green tests = your machine is ready
uv run jupyter lab labs/     # the same notebooks; the playground embeds still work
```

Laptop-only extras: `uv run scripts/view.py` (native MuJoCo viewer, SPACE cycles
poses) and `uv run scripts/play_policy.py` (drive the shipped walker with arrow keys).

## The 8 labs

| # | directory | topic |
|---|-----------|-------|
| 1 | `labs/lab01_meet_the_duck` | MuJoCo basics, poses, the open-loop fall |
| 2 | `labs/lab02_kinematics` | trajectories, FK, rhythm (and honest failures) |
| 3 | `labs/lab03_feedback_control` | PD balance from scratch, push limits |
| 4 | `labs/lab04_sensors_estimation` | IMU, noise, complementary filter, balancing blind |
| 5 | `labs/lab05_trained_policy` | run the robot's real neural networks |
| 6 | `labs/lab06_first_rl` | Gymnasium, rewards, PPO — train standing |
| 7 | `labs/lab07_reward_shaping` | reward audits, the walking task, deadbands, GPU track |
| 8 | `labs/lab08_evaluation` | protocols, sim2sim robustness, publish to the playground |

Each lab: `lecture.ipynb` (doubles as the slides) + `lab.ipynb` (your
worksheet, with self-checking asserts) + `solution.ipynb` (instructors).
See `labs/README.md` for the authoring workflow.

## Publishing a policy (Lab 8)

A pull request adding `assets/community/<your-handle>/` (`policy.onnx`,
`manifest.json`, `README.md`) is the deployment: CI validates it
(`validate_submission`), and the playground rebuilds with your duck in it.
See `assets/community/README.md`.

## Layout

- `src/microduck_sim/` — the course library, built up lab by lab
  (`web.py` = playground helpers, `colab.py` = Colab bootstrap)
- `assets/mjcf/` — robot models (vendored from Pollen, CC BY-SA-NC — see `assets/ATTRIBUTION.md`)
- `assets/policies/` — the 9 shipped ONNX behaviors
- `assets/checkpoints/` — instructor-trained fallbacks
- `assets/community/` — Lab 8 submissions (shown in the playground)
- `playground/` — builds the browser playground (isolated uv project: mjswan + mjlab;
  instructor/CI only, see `playground/build.py`)
- `scripts/` — `view.py`, `play_policy.py`, `train_sb3.py`
- `stage2/` — GPU track notebook + instructor GPU notes
- `CLAUDE.md` — development plan + progress log
