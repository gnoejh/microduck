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
# # Lab 6 — Your First RL Training · WORKSHEET
#
# Train a network to stand. T1–T5. (T3/T5 each cost ~2–3 min of CPU —
# start them, then think about the questions while they run.)

# %% [markdown]
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/microduck/blob/main/labs/lab06_first_rl/lab.ipynb)
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
from microduck_sim import colab; colab.setup("labs/lab06_first_rl")

# %%
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import microduck_sim.envs  # noqa: F401
from microduck_sim.obs import NUM_JOINTS

env = gym.make("MicroduckStand-v0")

# %% [markdown]
# ## T1 — Environment spelunking
#
# (a) Reset with seed 7; print the projected-gravity and joint-position
# slices of the observation (Lab 5's map: 3:6 and 6:20) and sanity-check
# their values. (b) From the same seed, take one zero-action step and one
# step with +0.3 on `left_knee` (action index 3); compare the reward terms
# (`info["reward_terms"]`) and explain the differences.

# %%
# TODO

# %% [markdown]
# *Your explanation:* ...

# %% [markdown]
# ## T2 — Baselines
#
# Write `rollout(policy_fn, seed) -> (total_reward, steps)`. Over 10 seeds,
# report mean±std reward and survival time for: zero action, random actions,
# random×0.1. Rank and explain.

# %%
def rollout(policy_fn, seed):
    # TODO
    ...


# TODO: the three baselines

# %% [markdown]
# *Your ranking & why:* ...

# %% [markdown]
# ## T3 — Train
#
# Run 300k steps of PPO (≈2–3 min):
#
# ```
# uv run scripts/train_sb3.py --steps 300000 --out runs/lab6_t3
# ```
#
# (from a terminal, or via `subprocess.run` as in the cell hint). Then load
# `runs/lab6_t3/progress.csv` and plot mean standing time
# (`rollout/ep_len_mean / 50`) against `time/total_timesteps`.

# %%
# TODO (hint: subprocess.run([sys.executable, "../../scripts/train_sb3.py", ...]))

# %%
df = pd.read_csv("runs/lab6_t3/progress.csv")
final_len = df["rollout/ep_len_mean"].iloc[-1]
first_len = df["rollout/ep_len_mean"].iloc[0]
assert final_len > 2 * first_len
print(f"T3 ok: {first_len/50:.2f} s → {final_len/50:.2f} s")

# %% [markdown]
# ## T4 — Evaluate, export, replay
#
# (a) Load the checkpoint (`PPO.load`) and evaluate 10 deterministic episodes
# with your `rollout`. (b) Export to ONNX (`export_sb3_to_onnx`) and replay
# it through `PolicyRunner` for 5 s — YOUR network in the Lab 5 machinery.
# Record a video if you like (`record_policy` + `show_video`).

# %%
# TODO
replay_upright = ...

# %%
assert replay_upright in (True, False)  # ran end-to-end
print("T4 done — trained, exported, replayed")

# %% [markdown]
# **Question:** your exported ONNX and `BEST_alpha_stand.onnx` are now
# interchangeable files. What, concretely, makes that true?
#
# *Your answer:* ...

# %% [markdown]
# ## T5 — Challenge: the exploration knob 🏆
#
# Retrain with `--log-std-init 0` (SB3's default σ = 1.0; ours is −1 ⇒
# σ≈0.37) at the same 300k budget, overlay both learning curves, and explain
# the gap. Leaderboard: highest final `ep_len_mean` at exactly 300k steps —
# any knob allowed except more steps (try `--n-envs`, learning rate,
# `net_arch`, reward weights...).

# %%
# TODO

# %% [markdown]
# *Your explanation of the gap:* ...
#
# ---
# You now own the full pipeline: env → reward → PPO → checkpoint → ONNX →
# runner. Next week we point it at walking, where the reward starts lying.
