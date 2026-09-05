# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lab 8 — Evaluation, Robustness & Shipping 🚀
#
# **Microduck Simulation Course · session 8 of 8**
#
# We own no robot — so **evaluation is our deployment**. Today:
# 1. Protocols, not vibes: the scorecard
# 2. The push tournament (a humbling ranking)
# 3. Sim2sim: variant sweeps as the reality-gap exam
# 4. The community ecosystem: manifests, reproduction, publishing
# 5. Mini-projects & where this course goes next

# %% [markdown]
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/microduck/blob/main/labs/lab08_evaluation/lecture.ipynb)
# *Colab-first:* run the setup cell below once per runtime (it installs the course
# package and picks the headless renderer). On a laptop checkout it is a no-op.

# %%
# --- setup (Colab or laptop; safe to re-run) -----------------------------------
import os, subprocess, sys
if "google.colab" in sys.modules and not os.path.exists("/content/microduck/pyproject.toml"):
    subprocess.run(["git", "clone", "-q", "--depth", "1",
                    "https://github.com/gnoejh/microduck.git", "/content/microduck"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "/content/microduck"], check=True)
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")  # headless Linux (Colab, CI); Win/mac keep their default
from microduck_sim import colab; colab.setup("labs/lab08_evaluation")

# %%
import matplotlib.pyplot as plt
import numpy as np

from microduck_sim.evaluate import evaluate_walking, max_push, variant_sweep
from microduck_sim.policy import OnnxPolicy, PolicyRunner

# %% [markdown]
# ## Protocols, not vibes
#
# "It looks fine" is not a result. A protocol fixes: the task and command,
# the number of episodes and their seeds, the metrics, and the termination
# rule — then it's reproducible to the tick (deterministic sim + seeded
# resets). Ours (`microduck_sim.evaluate`):
#
# - `evaluate_walking`: survival rate, achieved speed, tracking error over
#   seeded perturbed episodes
# - `max_push`: largest 0.1 s shove survived while standing
# - `variant_sweep`: the same scorecard across *model variants* — our
#   substitute for a reality gap

# %%
print("shipped walker @ vx=0.25:", evaluate_walking("walk", episodes=3, duration=6.0))

# %% [markdown]
# ## The push tournament
#
# Three standing controllers from this course, same exam (numbers from the
# committed instructor runs — recompute live if you like):
#
# | controller | origin | max push |
# |---|---|---|
# | our SB3 `stand_ppo` | Lab 6, 1M steps, 7 min CPU | **0.6 N** |
# | hand-tuned ankle PD | Lab 3, ~15 lines | **1.1 N** |
# | shipped `stand` policy | GPU, domain-randomized, gets up after falls | **≥ 5.8 N** |
#
# Read it carefully, it's the honest state of robot learning in one table:
#
# - **RL is not magic**: our trained policy loses to a 15-line PD controller.
#   Why? Its training env *never contained a push* — you don't get robustness
#   you didn't train for.
# - **The shipped policy barely has a limit**: it was trained with
#   randomization AND it can stand back up — falling stopped being fatal.
#   Robustness came from the training *distribution*, not the algorithm.

# %%
print("live check, shipped stand policy:", max_push("stand", resolution=1.0, limit=4.0), "N+")

# %% [markdown]
# ## Sim2sim: the variant sweep
#
# We can't measure a sim-to-real gap, but we can measure sim-to-*sim*:
# the same policy across model variants (full collisions / feet-only /
# ±1° gear backlash). Recall Lab 7's discovery — on the backlash model the
# walker's low-command deadband *disappears* (chatter as dither):

# %%
sweep = variant_sweep("walk", vx=0.25, episodes=2)
for variant, result in sweep.items():
    print(f"{variant:10s} survival {result['survival_rate']:.0%}  "
          f"speed {result['mean_speed']:+.3f}  err {result['tracking_error']:.3f}")

# %% [markdown]
# A policy that keeps its score across variants earned its robustness; one
# that collapses was overfit to one physics. This is exactly the test the
# Isaac-half of the semester will run at scale, with thousands of randomized
# worlds.
#
# ## The ecosystem: policies as artifacts
#
# The community publishes Microduck behaviors on the Hugging Face Hub as
# `policy.onnx` + **`manifest.json`** — obs/action contract, command
# semantics, *and a documented eval battery*. Case study,
# `RemiFabre/microduck-flamingo-cycle` (stand on one foot on command):
#
# ```json
# "command": { "twist": ["flag: 0 = two feet, 1 = one foot",
#                        "side: +1 = right foot down", "unused"] },
# "eval":    { "lifted_foot_height_m": 0.09,
#              "known_limits": "falls on backward pushes >= 0.18 m/s ..." }
# ```
#
# We downloaded it and *reproduced its documented eval in our sim*: commanded
# [1, +1] → left foot settles at **0.093 m** (manifest says 0.09), and it
# returns to two feet on [0, 0]. Different simulator, same contract, same
# behavior — that's what a good manifest buys.
#
# ### ...but the contract alone guarantees nothing
#
# We ran **seven** Hub checkpoints in our sim. All Apache-2.0, all declaring
# 61 obs → 14 actions at 50 Hz. Four kept their behavior; three fell within
# 0.5–3 s at *every* command we tried. The split follows one manifest field:
#
# | the author evaluated in (`eval.sim_proxy`) | transferred to our sim |
# |---|---|
# | "CPU MuJoCo with the BAM XL330 servo model" | **4 / 4** |
# | "mjlab / MuJoCo Warp", or not declared at all | **0 / 3** |
#
# The contract is a *wire format*, not a promise about physics. A policy is also
# a function of the actuator model it grew up with: Pollen's BAM model gives the
# XL330s backlash, back-EMF, current saturation and delay, where we use plain
# position actuators. A 2.2 m/s running gait leans hard on that force response,
# and it does not survive the crossing. `happy-hop` shows the middle case —
# upright and recognizable, but a 7 mm hop instead of a real one.
#
# That is the whole sim2sim lesson in one table, and it is why the manifest asks
# what you evaluated on. A stranger cannot re-run your training; they *can* ask
# whether your simulator resembles theirs. Worksheet T4 has you reproduce it.
#
# Bonus finding from the same seven: the `rough-walk` siblings have **no command
# deadband** — 0.071 m/s at vx = 0.2, where the shipped walker gives 0.001 (Lab
# 7's finding). Two policies, same contract, different command semantics in
# practice. Mini-project 2 is exactly this problem.
#
# ## Publishing your own (the worksheet finale)
#
# Your Lab 6/7 exports are already contract-compatible. Shipping one =
# `policy.onnx` + your `manifest.json` (declare commands honestly, document
# the eval battery you actually ran, list known limits) + a README —
# `make_manifest` writes the schema, `validate_submission` checks it:
#
# ```python
# from microduck_sim.evaluate import make_manifest, validate_submission
# manifest = make_manifest("my-duck", eval=my_scorecard, known_limits="...")
# validate_submission("my_release", quick=False)   # what CI will run on your PR
# ```
#
# **Deployment = a pull request** adding `assets/community/<your-handle>/`.
# CI validates it, the playground rebuilds, and your duck is in everyone's
# browser: `show_playground("walk", policy="<your-handle>")`. The example
# submission is the instructor's own (honest!) walk_ppo:

# %%
from microduck_sim.web import show_playground

show_playground("walk", policy="course_walk_cpu", height=480)

# %% [markdown]
# Optionally also push the folder to the Hugging Face Hub, where the community
# checkpoints live (`huggingface-cli login`, then `HfApi().upload_folder(...)`).
#
# Known limits are not weakness — the flamingo manifest lists exactly when it
# falls, and that's what makes it *usable* by strangers.
#
# ## Mini-projects (pick one, 2–3 weeks)
#
# 1. **Robust stand**: add random pushes to `MicroduckStand-v0` training;
#    beat the PD controller's 1.1 N (target: 2 N+).
# 2. **Fix the deadband**: retrain walking with commands sampled densely in
#    [0.05, 0.25]; deliver a walker with no deadband, manifest documenting
#    the tracking curve.
# 3. **A new trick**: sit-spin, bow, head-bang to a beat — one-shot policy
#    with phase encoding (see the crouch-glide's [cos φ, sin φ] convention).
# 4. **Estimator upgrade**: replace the complementary filter with a Kalman
#    filter (Lab 4 bonus), quantify balance improvement under pushes.
# 5. **Your idea** — pitch it with: task, reward sketch, eval protocol.
#
# Grading = your manifest: does the eval battery hold up when we run it?
#
# ## The whole course in one slide
#
# | lab | you built | it became |
# |-----|-----------|-----------|
# | 1–2 | poses, trajectories, FK | the action space |
# | 3 | feedback control | the baseline that RL must beat |
# | 4 | estimation from noisy IMU | the observation space |
# | 5 | the 61-dim contract | interoperability with the whole ecosystem |
# | 6–7 | envs, rewards, PPO | your own policies |
# | 8 | protocols & manifests | shippable, checkable artifacts |
#
# Next semester: the same loop — parallel sims, randomization, sim2sim — in
# NVIDIA Isaac Sim, at industrial scale. You already speak the language. 🦆
