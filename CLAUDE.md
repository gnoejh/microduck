# Microduck Simulation Lab — Teaching Environment (Simulation-Only)

A **simulation-only** teaching environment for the Pollen Robotics **Microduck** biped
(25 cm, 800 g, 15 Dynamixel XL330 servos, 50 Hz policy loop). We own no hardware and
never will for this course — simulation IS the platform.

**Course context:** first half of a semester (8 lab sessions) for **undergraduates**
(Python basics, first robotics/RL exposure), covering kinematics & control, state
estimation, and reinforcement learning. The second half of the semester is **NVIDIA
Isaac Sim** (planned separately later); Labs 7–8 seed the bridge concepts (parallel
sims, domain randomization, GPU training).

**Hard constraints:**
- Students work on their own **laptops, CPU only** (Win/macOS/Linux). Everything
  student-facing must run there. GPU work (instructor RTX 4060, free Colab T4) is an
  optional advanced track.
- **Progressive derivation:** code, slides, and labs build up session by session; each
  lab graduates a module into the shared `microduck_sim` library that later labs import.
- **Jupyter-based slides:** each lab's `lecture.ipynb` doubles as the presentation.
- **Update this file** (checkboxes + dated progress log entry) after every work session.

Approved full plan: `C:\Users\hjeong\.claude\plans\before-going-on-we-goofy-graham.md`

## Upstream references

- Product: https://pollen-robotics.com/microduck/ · announcement blog
- Robot firmware/SDK (Rust, RK3566): https://github.com/pollen-robotics/microduck (Apache-2.0)
- RL training: https://github.com/pollen-robotics/microduck_rl (Apache-2.0 code; models CC BY-SA-NC)
- HF Space **pollen-robotics/microduck-simulator** ("RL Playground"): MuJoCo-WASM +
  onnxruntime-web in the browser, ships 6 pretrained ONNX policies, keyboard/gamepad,
  legged+roller variants. Zero-install demo (Lab 1 warm-up) and our ONNX/MJCF source.
- Community policy checkpoints on HF (search "microduck"): e.g. RemiFabre/microduck-flamingo-cycle,
  HannesVonEssen/microduck-running — used in Lab 8 comparisons.

Key technical facts (verified 2026-08-31):
- Upstream training = **mjlab (MuJoCo Warp) + PPO, CUDA required** → our student track is
  plain MuJoCo (CPU) + Gymnasium, our own code. The two tracks share the policy format.
- MJCF from Onshape: `robot_walk.xml`, `robot_allcollisions.xml`,
  `robot_allcollisions_rollers.xml`, backlash variants (±1°); STAND/SIT/FOLD keyframes.
- BAM actuator model (XL330: voltage, back-EMF, friction); DR over voltage/delay/friction.
- **61-dim observation contract** (48 proprio + commands), normalization baked into ONNX
  → pretrained/official policies replay in our CPU sim (Lab 5 linchpin; validate early).
- No hardware ⇒ no sim-to-real: "deployment" = publish to HF Hub / load in browser
  playground; "reality gap" = sim2sim evaluation across DR/backlash/friction variants.

## The 8-lab arc (each lab: lecture.ipynb + lab.ipynb + solution.ipynb)

| # | Lab | Teaches | Graduates into library |
|---|-----|---------|------------------------|
| 1 | Meet the duck | Browser playground warm-up; uv install; MJCF anatomy; viewer; keyframes | `model.py` |
| 2 | Kinematics & joint control | Joint space, position targets, FK via sites, leg trajectories | `control.py` (trajectories) |
| 3 | Feedback control | PD from scratch, gain tuning, standing balance, pushes | `control.py` (PD @ 50 Hz) |
| 4 | Sensors & estimation | IMU, noise, complementary filter vs ground truth (Kalman = challenge) | `sensors.py`, `estimation.py` |
| 5 | Running a trained policy | 61-dim contract, ONNX walk policy in our sim, obs/action inspection | `obs.py`, `policy.py` |
| 6 | First RL training | Gymnasium anatomy, rewards, SB3 PPO standing task ≤30 min CPU | `envs/MicroduckStand-v0` |
| 7 | Reward shaping & walking | Reward ablations, velocity task, DR; optional Colab GPU track (microduck_rl) | `envs/MicroduckWalk-v0`, `stage2/` |
| 8 | Evaluation & mini-project | Push tests, DR sweeps, backlash = sim2sim robustness; compare vs official/community policies; publish to HF Hub; outlook → Isaac half | `evaluate.py` |

Principles: lab N imports the *library* version of earlier modules (falling behind never
blocks); every notebook fits a 2–3 h session on laptop CPU; training cells ≤30 min with
pretrained fallback checkpoints provided.

## Milestones

### M0 — Foundation ✅ (2026-08-31)
- [x] M0.1 Deps in pyproject (mujoco, gymnasium, numpy, matplotlib, onnxruntime, stable-baselines3, torch-cpu, jupyter, imageio); deleted `main.py`; `src/microduck_sim/` package (hatchling, editable)
- [x] M0.2 Vendored 16 MJCF + 47 STL meshes from microduck_rl into `assets/mjcf/`; **9** ONNX policies (not 6!) from the HF Space into `assets/policies/` (Space uses git-LFS and git-lfs isn't installed → fetched via HF resolve URLs); Space's plain-MuJoCo MJCF + kinematics.json kept in `assets/space/` for the Lab 5 port; `assets/ATTRIBUTION.md` written
- [x] M0.3 `scripts/view.py`: viewer with SPACE=cycle INIT/STAND/SIT/FOLD, R=reset (visual check pending on user's machine)
- [x] M0.4 `tests/test_model.py`: 16 tests green (all 6 scene variants load, keyframes, walk structure, IMU sensors, short-term stand, policies not LFS pointers)
- [x] M0.5 First real commit

### M1 — Labs 1–2 (`model.py`, trajectory helpers, notebooks)
### M2 — Labs 3–4 (PD control, sensors/estimation, push helper, notebooks)
### M3 — Lab 5 ⚠ linchpin (`obs.py` 61-dim, `policy.py` ONNX runner, `scripts/play_policy.py`;
  port obs/action wiring from the HF Space's JS glue — it already replays these policies in plain MuJoCo)
### M4 — Labs 6–7 (Gym envs, CPU-tuned SB3 PPO, `stage2/colab_train.ipynb`, fallback checkpoints)
### M5 — Lab 8 + polish (`evaluate.py`, HF publish walkthrough, setup guide, CI, notebook-execution tests)

## Risks
- Lab 5 fidelity gap (BAM actuators vs our position actuators) — mitigate via the Space's
  own MJCF+glue; tackle in M3, early enough to adjust Labs 5–7.
- If full-duck CPU PPO exceeds the 30-min budget → reduced-obs / fixed-base balance variant.

## Progress log
- **2026-08-31** — Kickoff. Researched robot + `microduck_rl` (GPU-only upstream ⇒ own CPU
  track). Surveyed HF: found official browser simulator Space with 6 ONNX policies (adopted
  as Lab 1 warm-up + policy source) and community checkpoints. Constraints finalized with
  user: simulation-only (no hardware ever), laptop-CPU students, 8 undergrad labs
  (control + estimation + RL), Jupyter-based slides, Isaac Sim occupies the other half
  semester. Plan approved; starting M0.
- **2026-08-31 (later)** — **M0 done.** Facts learned: walk model = 14 position actuators
  (nu=14, nq=21; beak excluded), free joint is `trunk_base_freejoint`; `scene_ball.xml` has
  NO keyframes; only 6 geoms collide (feet) — trunk passes through floor when fallen;
  **open-loop STAND tips over after ~0.9 s** (measured) → perfect motivation demo for
  Lab 3 feedback control. 16 tests green. Next: M1 (Labs 1–2 notebooks + model/control
  library polish).
