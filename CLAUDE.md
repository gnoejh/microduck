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

### M1 — Labs 1–2 ✅ (2026-08-31)
- [x] `control.py` (50 Hz loop contract: constant/goto min-jerk/sine/chain/run_control_loop + logging) and `viz.py` (offscreen render/record/show/show_grid/show_video — all headless-safe)
- [x] Lab 1 "Meet the duck": lecture + worksheet + solution (MJCF anatomy, keyframes, head-camera view, the open-loop fall, first ctrl command, fall-time experiment)
- [x] Lab 2 "Kinematics & joint control": lecture + worksheet + solution (walk-vs-full variants, pose vectors, teleport-vs-min-jerk quantified, FK beak-path, seated dance, longest-stand challenge)
- [x] jupytext workflow: `.py` percent files = source of truth → `.ipynb` committed WITHOUT outputs; lecture+solution notebooks validated by headless execution (see `labs/README.md`)
- [x] 23 tests green
### M2 — Labs 3–4 ✅ (2026-08-31)
- [x] control.py: `quat_to_pitch_roll`, `make_balance_controller` (ankle strategy, injectable pitch/rate for Lab 4), `make_push` (t0 latched on first tick — creating before reset was a silent no-op bug, fixed)
- [x] sensors.py: `read_sensor`, `NoisyIMU` (Gaussian noise + gyro bias, seeded; `valid` flag when |accel| ≤ 1)
- [x] estimation.py: `accel_pitch`, `ComplementaryFilter` (invalid-sample guard)
- [x] Lab 3 "Feedback control": gravity droop vs servo kp, quaternion→pitch, ankle-strategy PD (0.82 s → indefinite), sign sabotage, push limits + hip-strategy challenge
- [x] Lab 4 "Sensors & estimation": specific force (free-fall zero!), noise/bias modeling, estimator ladder (accel / gyro / complementary), the atan2(0,0)=π init faceplant, balance-blind finale
- [x] 32 tests green; all 4 new lecture/solution notebooks validated by headless execution
### M3 — Lab 5 ✅ LINCHPIN CLEARED (2026-08-31)
- [x] Decoded the contract from the Space's game.js + constants.js: obs = [gyro(3),
  projected_gravity(3), qpos−DEFAULT_POSE(14), qvel(14), last_action(14), cmd(13)];
  ctrl = DEFAULT_POSE + action·1.0; **timestep 0.005, decimation 4** (Space injects
  `<option timestep>` into robot_allcollisions.xml); DEFAULT_POSE ≡ our STAND ctrl;
  cmd: [vx,vy,wz, 4 head slots, zero-pad]; sitstand: cmd[0]=1 sits; one-shots use
  [cosφ,sinφ]; fall: gz>−0.5, upright: gz<−0.85; VEL limits 0.25/−0.2/1.0
- [x] obs.py (`ObsBuilder`), policy.py (`OnnxPolicy`, `PolicyRunner`), viz `record_policy`
- [x] **Verified: shipped policies WORK in our CPU sim** — walk 1.06 m/10 s + steers,
  stand balances, sitstand does the Lab-2-impossible transition both ways, get-up
  recovers from prone in 1.1 s after an 8 N knockdown; **78× realtime** single env
- [x] `scripts/play_policy.py` (arrow-key driving); Lab 5 lecture/worksheet/solution
  (contract tour, square drive, gait FFT ≈ 2.5 Hz, overdrive extrapolation, full
  fall→get-up→walk pipeline); 40 tests green
- Gotcha for later labs: left/right joints have MIRRORED sign conventions — raw L/R
  hip actions correlate +0.65 while being physically antiphase; also pitch ≡ asin(gx)
  exactly (gx = 2(wy−zx))
- **2026-08-31 (M4+M5)** — Course complete: all 8 labs written, all 16 lecture/solution
  notebooks execute headless, 49 fast tests + 16 opt-in notebook tests, CI, README.
  Verified highlights: CPU PPO **solves standing in ~7 min** (1M steps, 2500 fps, 8 envs;
  log_std_init=-1 essential); 2M-step walk run **learned to stand instead of walk**
  (0.015 m/s, 100% survival) — predicted by the reward audit (non-walker earns 90%),
  now Lab 7's core lesson; reward-fix ladder verified (weights 1.11→1.30 < eval dist
  1.49 < progress term 3.15); **shipped-walker deadband** (walks only at vx=0.25; backlash
  chatter dithers it away); push tournament: our stand 0.6 N < Lab 3 PD 1.1 N < shipped
  ≥5.8 N; **community repro**: flamingo-cycle checkpoint runs in our sim, lifted-foot
  0.093 m vs manifest's 0.09, known-limits verified → manifest = Lab 8 grading rubric;
  SB3→ONNX export matches predict to 1e-5 and replays in PolicyRunner. Windows gotcha:
  write_text needs encoding="utf-8" (cp949 locale). Course ready for dry-run teaching.
### M4 — Labs 6–7 (in progress 2026-08-31)
- [x] `envs/`: MicroduckStand-v0 + MicroduckWalk-v0, SAME contract as shipped policies
  (61 obs incl. commands, 14 actions = ctrl offsets ±0.6, 50 Hz, dt 0.005×4); named
  reward terms in info; `reward_weights` kwarg for ablations (+ `--rw` in train script)
- [x] `scripts/train_sb3.py` (CPU-tuned PPO; **log_std_init=-1 critical** — SB3's σ=1.0
  saturates the ±0.6 action box and cripples learning), `export.py` (SB3→contract-ONNX,
  verified vs predict), `evaluate.py` (scorecard/max_push/variant_sweep)
- [x] **Stand solved on CPU**: 1M steps ≈ 7 min laptop (2500 fps, 8 envs), 10/10 full
  episodes; ONNX export replays in PolicyRunner; committed to assets/checkpoints/
- [x] Lab 6 notebooks (lecture executes; solution has 2×300k self-training cells)
- [x] Lab 7 notebooks: reward-audit ladder ALL VERIFIED: default reward pays a
  non-walker 90% of a walker (ratio 1.11) → weights ceiling 1.30 → fixed-cmd eval
  1.49 → +progress term 3.15. **Shipped walker has a command deadband**: stands still
  at vx≤0.2, walks only at 0.25 (the gamepad's exact value) — and the backlash
  variant's chatter DITHERS it out of the deadband (walks at 0.2 there!)
- [x] walk_v1 2M-step run done (~13 min CPU) → assets/checkpoints/. **The agent took
  the loophole**: 100% survival, 0.015 m/s — it learned to STAND, exactly as the T1
  audit predicted. Adopted as Lab 7's centerpiece ("audit first; the reward is a
  contract"). All lecture/solution notebooks for Labs 6–7 executed headless ✅
- [x] stage2/: colab_train.ipynb (SB3 track + official microduck_rl on T4; upstream
  section marked not-yet-GPU-verified), rtx4060.md (instructor notes + TODO verify)

### M5 — Lab 8 + polish (in progress 2026-08-31)
- [x] `evaluate.py`; Lab 8 notebooks written (execution pending walk artifacts)
- [x] **Push tournament measured**: our CPU stand_ppo 0.6 N < Lab 3 hand-PD 1.1 N <
  shipped stand ≥5.8 N — "robustness you didn't train for doesn't exist"
- [x] **Community repro verified**: RemiFabre/microduck-flamingo-cycle (policy.onnx +
  manifest.json schema v2) runs in our sim; one-foot stand lifted-foot height 0.093 m
  vs manifest's 0.09; backward-push known_limits verified. Manifest = Lab 8's grading rubric.
- [x] lab08 notebooks executed (found+fixed a real cp949 Windows-locale bug in
  write_text — kept as a teaching note); root README/setup guide; GitHub Actions CI
  (fast suite on push; full notebook dry-run on manual dispatch, MUJOCO_GL=osmesa);
  opt-in `pytest -m notebooks` runs all 16 lecture/solution notebooks (~25 min)
- Remaining course TODOs (not blocking): verify stage2 GPU sections on RTX 4060/Colab;
  instructor decides solution-distribution policy; mini-project grading rubric detail

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
- **2026-08-31 (M1)** — Labs 1–2 built and validated. Empirical findings that reshaped
  Lab 2 (rule: *demo it or don't claim it*): in the walk model a seated duck sinks through
  the floor (feet-only collision) → Lab 2 uses the `full` variant and T1 teaches variants;
  SIT is a stable base (motions demoed seated); **open-loop stand-up from SIT always
  face-plants** (even min-jerk, even via crouch) → honest cliffhanger for Lab 3 + Lab 5's
  `BEST_alpha_sitstand.onnx`; teleport vs min-jerk head turn: 5.4 vs 2.3 rad/s peak;
  longest-open-loop-stand challenge is winnable (baseline 0.82 s → slight crouch 1.18 s);
  sine chains need integer periods per segment or poses jump. Next: M2 (Labs 3–4).
- **2026-08-31 (M2)** — Labs 3–4 built, all claims sim-verified. Key numbers (full model):
  ankle-strategy PD kp=1.0/kd=0.1 stands indefinitely; wrong sign falls at 0.4 s (faster
  than no feedback); max push ankle-only **1.1 N forward / 4.1 N lateral** (support-polygon
  asymmetry), hip strategy (+0.4·corr mirrored like ankles) → 1.4 N; servo droop:
  |err| 0.47→0.009 rad for kp 0.2→8. IMU: duck spawns airborne → first ~30 ms are free
  fall → accel ≈ 0 → naive filter init atan2(0,0)=π = instant faceplant (validity guard
  required); complementary α=0.98 stands blind w/ noise+bias and survives 1 N (accel-only
  falls at 0.8 N — dynamic acceleration corrupts it; shown cleanly via 4 N sinusoidal
  shake: 0.005→0.086 rad err with CLEAN sensors). Next: M3 (Lab 5, the ONNX linchpin).
