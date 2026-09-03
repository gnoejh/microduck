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
# # Microduck on Google Colab — the GPU track 🚀
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/microduck/blob/main/stage2/colab_gpu_track.ipynb)
#
# Two ways to train beyond the lab session's budget, both optional (Labs 6–7 run
# fine on CPU):
#
# **Part A — our SB3 track, bigger budget.** Same code as Lab 6, more steps.
# Works on ANY runtime (CPU or GPU; the GPU adds nothing — the physics is CPU).
#
# **Part B — the official `microduck_rl` pipeline** (mjlab / MuJoCo Warp, PPO,
# thousands of parallel ducks). Needs a **GPU runtime**: Runtime → Change
# runtime type → T4 GPU. This is the pipeline that trained the policies you
# drove in Lab 5 — and a preview of the parallel-simulation world of the
# Isaac Sim half-semester.
#
# > ⚠ Colab's free tier can disconnect you: checkpoint to Drive early & often
# > (`colab.persist`). Free GPUs are not guaranteed at peak times — every result
# > below has an instructor-trained fallback in `assets/checkpoints/`.

# %%
# --- setup (Colab or laptop; safe to re-run) -----------------------------------
import os, subprocess, sys
if "google.colab" in sys.modules and not os.path.exists("/content/microduck/pyproject.toml"):
    subprocess.run(["git", "clone", "-q", "--depth", "1",
                    "https://github.com/gnoejh/microduck.git", "/content/microduck"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "/content/microduck"], check=True)
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")  # headless Linux (Colab, CI); Win/mac keep their default
from microduck_sim import colab; colab.setup("")

# %% [markdown]
# ## Part A — SB3 track on Colab (any runtime)
#
# Colab's free runtime has **2 vCPUs**, so use `--n-envs 2` (the script's default
# adapts to the CPU count). Expect roughly a third of a laptop's throughput.

# %%
# Train the standing task with a bigger budget than the lab session allowed (~20 min):
# subprocess.run([sys.executable, "scripts/train_sb3.py", "--env", "MicroduckStand-v0",
#                 "--steps", "1000000", "--out", "runs/colab_stand"], check=True)

# %%
# Or the walking task (Lab 7) with the reward fix of your choice:
# subprocess.run([sys.executable, "scripts/train_sb3.py", "--env", "MicroduckWalk-v0",
#                 "--steps", "2000000", "--rw", "progress=1.0", "--out", "runs/colab_walk"], check=True)

# %%
# Save your run to Drive so a disconnect can't eat it:
# colab.persist("runs", "microduck_runs")

# %% [markdown]
# Evaluate/export exactly as in Lab 6: `export_sb3_to_onnx(...)`, replay with
# `PolicyRunner`, or view it in the playground through a Lab 8 pull request.

# %% [markdown]
# ## Part B — official microduck_rl (GPU runtime required)
#
# The real thing: MuJoCo Warp stepping thousands of ducks in parallel on the
# GPU. Numbers below marked *measure* are placeholders until the instructor's
# T4 dry-run (see `stage2/gpu_notes.md`).

# %%
# Check the GPU first — this must print a T4/L4/A100, not an error:
# subprocess.run(["nvidia-smi"], check=True)

# %%
# subprocess.run("git clone -q https://github.com/pollen-robotics/microduck_rl /content/microduck_rl", shell=True, check=True)
# subprocess.run("pip install -q uv && cd /content/microduck_rl && uv sync", shell=True, check=True)

# %%
# A 20-minute taste on a T4: 1024 envs, ~300 PPO iterations. Upstream quotes
# 1–2 h for a usable gait at 4096 envs on bigger cards.
# subprocess.run("cd /content/microduck_rl && uv run train Mjlab-Velocity-Flat-MicroDuck "
#                "--env.scene.num-envs 1024 --agent.max-iterations 300", shell=True, check=True)

# %%
# Checkpoint to Drive periodically (run in a separate cell while training):
# colab.persist("/content/microduck_rl/logs", "microduck_rl_logs")

# %%
# Export the trained policy to ONNX (normalizer baked in) and run it in the
# COURSE simulator — same 61-dim contract end to end, that's the whole point:
# subprocess.run("cd /content/microduck_rl && uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck "
#                "--wandb-run-path <entity/project/run_id> --onnx-file /content/mjlab_walk.onnx", shell=True, check=True)
# from microduck_sim.evaluate import evaluate_walking
# print(evaluate_walking("/content/mjlab_walk.onnx"))

# %% [markdown]
# ### Why this scales
#
# Your laptop stepped 8 ducks in 8 processes; the GPU steps 1024–4096 in
# lockstep on one chip. Same MDP, same PPO idea, ~3 orders of magnitude more
# experience per hour. NVIDIA Isaac Sim (next half-semester) is built around
# exactly this trade — you already know the vocabulary.
