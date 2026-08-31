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
# # Microduck on Google Colab — GPU training track 🚀
#
# Two ways to train beyond your laptop, both optional (Labs 6–7 run fine on CPU):
#
# **Part A — our SB3 track, bigger budget.** Same code as Lab 6, more steps.
# Works on ANY Colab runtime (CPU or GPU).
#
# **Part B — the official `microduck_rl` pipeline** (mjlab / MuJoCo Warp, PPO,
# thousands of parallel ducks). Needs a **GPU runtime**: Runtime → Change
# runtime type → T4 GPU. This is the pipeline that trained the policies you
# drove in Lab 5 — and a preview of the parallel-simulation world of the
# Isaac Sim half-semester.
#
# > ⚠ Colab's free tier can disconnect you: checkpoint to Drive early & often.

# %% [markdown]
# ## Part A — SB3 track on Colab

# %%
# Clone the course repo and install (about a minute)
# !git clone https://github.com/YOUR-ORG/microduck-sim-course.git microduck
# %cd microduck
# !pip install -q uv && uv sync

# %%
# Train the standing task with a bigger budget than the lab session allowed:
# !uv run scripts/train_sb3.py --env MicroduckStand-v0 --steps 2000000 --out runs/colab_stand

# %%
# Or the walking task (Lab 7):
# !uv run scripts/train_sb3.py --env MicroduckWalk-v0 --steps 3000000 --out runs/colab_walk

# %%
# Save your run to Drive so a disconnect can't eat it:
# from google.colab import drive
# drive.mount("/content/drive")
# !cp -r runs /content/drive/MyDrive/microduck_runs

# %% [markdown]
# Evaluate/export exactly as in Lab 6: `export_sb3_to_onnx(...)`, then replay
# with `PolicyRunner` — locally on your laptop if you download the ONNX.

# %% [markdown]
# ## Part B — official microduck_rl (GPU runtime required)
#
# The real thing: MuJoCo Warp stepping thousands of ducks in parallel on the
# GPU. On a free T4 use a reduced env count.

# %%
# Check the GPU first — this must print a T4/L4/A100, not an error:
# !nvidia-smi

# %%
# !git clone https://github.com/pollen-robotics/microduck_rl
# %cd microduck_rl
# !pip install -q uv && uv sync

# %%
# Train velocity-tracking walking. 1024 envs fits a T4; expect a usable gait
# after 1–3 h (upstream quotes 1–2 h at 4096 envs on bigger cards).
# !uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 1024

# %%
# Checkpoint to Drive periodically (run in a separate cell while training):
# !cp -r logs /content/drive/MyDrive/microduck_rl_logs

# %%
# Export the trained policy to ONNX (see their scripts/export.py), download
# it, and run it in the COURSE simulator on your CPU laptop:
#
#   uv run scripts/play_policy.py --behavior path/to/your_policy.onnx
#
# Same 61-dim contract end to end — that's the whole point of the course.

# %% [markdown]
# ### Why this scales
#
# Your laptop stepped 8 ducks in parallel processes; the GPU steps 1024–4096
# in lockstep on one chip. Same MDP, same PPO idea, ~3 orders of magnitude
# more experience per hour. NVIDIA Isaac Sim (next half-semester) is built
# around exactly this trade — you already know the vocabulary.
