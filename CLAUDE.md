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
- **Colab-first, laptop optional** (decided 2026-09-04): every notebook opens in Google
  Colab (badge + setup cell) and still runs on a laptop checkout (Win/macOS/Linux, CPU).
  Interactive 3D = the browser **playground** (mjswan static site on GitHub Pages,
  embedded as an iframe); the native viewer scripts are laptop extras. GPU work =
  Lab 7's track on Colab T4 (upstream microduck_rl), instructor RTX 4060 / paid tier.
- **Progressive derivation:** code, slides, and labs build up session by session; each
  lab graduates a module into the shared `microduck_sim` library that later labs import.
- **Jupyter-based slides:** each lab's `lecture.ipynb` doubles as the presentation.
- **Update this file** (checkboxes + dated progress log entry) after every work session.

Approved full plan: `C:\Users\hjeong\.claude\plans\before-going-on-we-goofy-graham.md`
M6 (Colab-native platform) plan: `C:\Users\hjeong\.claude\plans\i-must-think-carefully-zippy-clarke.md`

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
| 1 | Meet the duck | Our browser playground warm-up (embedded); Colab setup; MJCF anatomy; keyframes | `model.py` |
| 2 | Kinematics & joint control | Joint space, position targets, FK via sites, leg trajectories | `control.py` (trajectories) |
| 3 | Feedback control | PD from scratch, gain tuning, standing balance, pushes | `control.py` (PD @ 50 Hz) |
| 4 | Sensors & estimation | IMU, noise, complementary filter vs ground truth (Kalman = challenge) | `sensors.py`, `estimation.py` |
| 5 | Running a trained policy | 61-dim contract, ONNX walk policy in our sim, obs/action inspection | `obs.py`, `policy.py` |
| 6 | First RL training | Gymnasium anatomy, rewards, SB3 PPO standing task ≤30 min CPU | `envs/MicroduckStand-v0` |
| 7 | Reward shaping & walking | Reward ablations, velocity task, DR; optional Colab GPU track (microduck_rl) | `envs/MicroduckWalk-v0`, `stage2/` |
| 8 | Evaluation & mini-project | Push tests, DR sweeps, backlash = sim2sim robustness; compare vs official/community policies; publish = PR into `assets/community/` → playground (HF Hub optional); outlook → Isaac half | `evaluate.py` (+ manifest/validation) |

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
- **2026-08-31 (publish)** — Pushed to https://github.com/gnoejh/microduck (M0–M5, all
  8 labs). Next session candidates: dry-run Lab 1 on the user's machine (viewer +
  notebooks), verify stage2 GPU sections on RTX 4060/Colab, solution-distribution policy.
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

### M6 — Colab-native platform (in progress 2026-09-04)
Decision: Colab-first; browser viewer = **mjswan** (MuJoCo-WASM + onnxruntime-web,
Apache-2.0); GPU track = upstream microduck_rl on Colab T4. Plan file above.
- [x] S0 isolated build env `playground/pyproject.toml` (mjswan 0.9.3 + mjlab 1.5.3 +
  torch-cpu; uv `override-dependencies` mujoco==3.10.0, numpy<2.5 — mjlab needs both);
  upstream `msp build microduck` builds all 9 scenes in **28 s** with the wheel's
  prebuilt frontend (160 MB dist). Gotchas: the wheel ships `template/dist` WITHOUT the
  cache marker → mjswan tries a Node rebuild; the Node build needs **PYTHONUTF8=1** on
  Windows (cp949 again: generated stubs contain "§"); Git Bash mangles `--base-path
  /microduck/` → use `MSYS_NO_PATHCONV=1`
- [x] S1 our library + all 6 MJCF variants pass `test_model`/`test_policy` on mujoco 3.10
- [x] S2 `web.py` port: our walk scene's fused obs graph + fell_over graph are
  **byte-identical to upstream's**, and the browser obs graph equals `ObsBuilder` to
  **2e-8** on a live state → the 61-dim contract holds in the browser. Visual drive
  check in a real browser: PENDING (instructor: `playground/build.py --serve`)
- [x] S2b `pose` scene (identity ONNX, 14 sliders, seated) builds; Lab 1/2 embed it
- [x] S3 course checkpoints re-exported at **opset 18** with Pollen-style metadata_props
  (`export.py`); match SB3 predict to 3e-7; replay upright; mjswan ignores metadata
  and applies no normalizer (our raw-obs policies need nothing special)
- [x] S4 (local half) `--base-path /microduck/` build emits `/microduck/assets/...`;
  `.github/workflows/playground.yml` builds + deploys to GitHub Pages (needs the
  one-time Pages=Actions setting). Colab-iframe check: PENDING first deploy
- [ ] S5 microduck_rl on Colab T4 (num-envs sweep, minutes-to-upright, export →
  PolicyRunner + playground) — needs a Colab session; recipe in `stage2/colab_gpu_track`
- [ ] S6 all 16 notebooks on a Colab CPU runtime (setup cell, MUJOCO_GL=egl, budgets)
- [ ] S7 in-Colab self-build of a student ONNX (`uv sync --project playground` cold time)
- [x] Library: `web.py` (show_playground + build port), `colab.py` (setup/persist, SETUP_CELL),
  `export.py` (opset 18 + metadata + read_onnx_metadata), `policy.py` (metadata,
  `community/<handle>`), `evaluate.py` (`make_manifest`, `validate_submission`),
  `train_sb3.py` (**SubprocVecEnv** default — it was DummyVecEnv while the text claimed
  "8 processes"), deps ipython + huggingface_hub
- [x] Playground: `playground/build.py` (11 scenes; walk scene hosts every walking-layout
  policy incl. `assets/community/*`), `make_pose_policy.py`, example submission
  `assets/community/course-walk-cpu`, `assets/community/README.md`
- [x] Notebooks: Colab badge + shared setup cell in all 24 + stage2; playground embeds
  in Labs 1, 2, 3 (closing), 5, 6, 8; Lab 7 GPU block = concrete T4 recipe +
  three-way scorecard table; Lab 8 deployment = PR; `stage2/colab_gpu_track.ipynb`
  (real clone URL, Drive via `colab.persist`), `stage2/gpu_notes.md` (measurement table)
- [x] Tests: `test_web`, `test_export`, `test_community` (fast suite); CI `playground.yml`
- Not ported to the browser on purpose: Lab 3/4 Python loops, pushes, eval protocols
  (Python = measurement instrument, browser = experience instrument)

## Risks
- Lab 5 fidelity gap (BAM actuators vs our position actuators) — mitigate via the Space's
  own MJCF+glue; tackle in M3, early enough to adjust Labs 5–7.
- If full-duck CPU PPO exceeds the 30-min budget → reduced-obs / fixed-base balance variant.
- Colab: free T4 not guaranteed at peak; 2 vCPU CPU runtime slower than a laptop; mjswan
  moves fast (61 releases) → pins + `playground/tests`; CC BY-SA-NC meshes on a public
  Pages site → ATTRIBUTION.md copied into the build, non-commercial teaching use.

## Progress log
- **2026-09-04 (M6 kickoff)** — Platform decision with the instructor: **Colab-first,
  laptop optional**; interactive 3D via **mjswan** (browser MuJoCo-WASM + ONNX; its
  `mjswan_playground` already wired all 9 Microduck policies); GPU track = upstream
  microduck_rl on Colab T4. Built the isolated `playground/` env (mujoco 3.10 + numpy<2.5
  overrides), ported the scene/obs wiring into `web.py` and verified the browser's fused
  obs graph is byte-identical to upstream's and equals `ObsBuilder` to 2e-8; re-exported
  checkpoints at opset 18 with Pollen metadata; `colab.py` + setup cell + badge in all
  25 notebooks; playground embeds in Labs 1/2/3/5/6/8; Lab 8 deployment = PR into
  `assets/community/`; CI `playground.yml` for GitHub Pages; SubprocVecEnv fix. Windows
  gotchas: PYTHONUTF8=1 for the Node build, MSYS_NO_PATHCONV=1 for `--base-path`.
  Verified locally: 59 fast tests, playground build test, Labs 1/2/3/5/6/8 lectures
  execute headless; full 11-scene build 134 MB. Committed + pushed to main (this
  push triggers the first `playground.yml` run). CI after push: `tests` green on
  ubuntu + windows (after making `colab.setup` a no-op outside Colab — it raised on
  headless Linux); `playground` **build job green on Linux incl. the Node frontend
  build**, deploy job failed because Pages is not enabled yet. **Next:** Settings →
  Pages → Source: GitHub Actions, then re-run the workflow; visual drive check in a
  browser; S5/S6/S7 in Colab; fill `stage2/gpu_notes.md` table.
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
- **2026-09-01** — Concept review session (what is MuJoCo / what is ONNX) while browsing
  Lab 1 solution + robot_walk.xml; no code changes. Standing next steps unchanged:
  dry-run Lab 1 on the user's machine (viewer + notebooks), verify stage2 GPU sections
  on RTX 4060/Colab, decide solution-distribution policy.
- **2026-09-03** — Housekeeping session: browsed Lab 1 solution notebook in the IDE; no
  code changes, working tree was clean. Standing next steps unchanged: dry-run Lab 1 on
  the user's machine (viewer + notebooks), verify stage2 GPU sections on RTX 4060/Colab,
  decide solution-distribution policy.
