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
# # Lab 6 — Your First RL Training · SOLUTION
#
# Note: T3's training cell takes ~2–3 minutes of CPU time by design.

# %% [markdown]
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/microduck/blob/main/labs/lab06_first_rl/solution.ipynb)
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
# (a) Reset with seed 7 and identify, by slicing the observation, the
# projected gravity and the joint-position blocks (Lab 5's map). (b) Step with
# a +0.3 offset on `left_knee` only (index 3) and describe what changes in
# the reward terms vs a zero action.

# %%
obs, _ = env.reset(seed=7)
print("projected gravity:", obs[3:6].round(3), "(≈ [0,0,−1]: upright)")
print("joint pos Δ:      ", obs[6:20].round(3), "(≈ 0 + reset noise)")

env.reset(seed=7)
_, r0, _, _, info0 = env.step(np.zeros(NUM_JOINTS, dtype=np.float32))
env.reset(seed=7)
a = np.zeros(NUM_JOINTS, dtype=np.float32); a[3] = 0.3
_, r1, _, _, info1 = env.step(a)
for k in info0["reward_terms"]:
    print(f"{k:12s} zero {info0['reward_terms'][k]:+.3f}   knee-kick {info1['reward_terms'][k]:+.3f}")
# The knee kick costs action_rate immediately and starts disturbing
# still/height on subsequent steps — the reward already 'feels' the twitch.

# %% [markdown]
# ## T2 — Baselines
#
# Measure mean±std of episode reward and survival time over 10 seeded
# episodes for (a) zero-action, (b) random actions, (c) random actions scaled
# by 0.1. Rank them and explain the ranking.

# %%
def rollout(policy_fn, seed):
    o, _ = env.reset(seed=seed)
    total, steps = 0.0, 0
    while True:
        o, r, term, trunc, _ = env.step(policy_fn(o))
        total += r; steps += 1
        if term or trunc:
            return total, steps


policies = {
    "zero": lambda o: np.zeros(NUM_JOINTS, dtype=np.float32),
    "random": lambda o: env.action_space.sample(),
    "random×0.1": lambda o: env.action_space.sample() * 0.1,
}
for name, fn in policies.items():
    st = np.array([rollout(fn, s) for s in range(10)])
    print(f"{name:11s} reward {st[:,0].mean():7.1f} ± {st[:,0].std():5.1f}   "
          f"survival {st[:,1].mean()/50:5.2f} s")
# zero > random×0.1 > random: full-scale noise shakes the duck over almost
# immediately; tiny noise is nearly open-loop; holding still is the best
# non-learning strategy — and still only ~1 s.

# %% [markdown]
# ## T3 — Train
#
# Train PPO for 300k steps (`scripts/train_sb3.py`, ~2–3 min). Then plot the
# learning curve (mean standing time vs env steps) from `progress.csv`.

# %%
import subprocess, sys
subprocess.run(
    [sys.executable, "../../scripts/train_sb3.py", "--steps", "300000",
     "--out", "runs/lab6_t3", "--n-envs", "8"],
    check=True, capture_output=True, text=True,
)

df = pd.read_csv("runs/lab6_t3/progress.csv")
plt.plot(df["time/total_timesteps"], df["rollout/ep_len_mean"] / 50)
plt.xlabel("env steps"); plt.ylabel("mean standing time [s]"); plt.show()
final_len = df["rollout/ep_len_mean"].iloc[-1]
first_len = df["rollout/ep_len_mean"].iloc[0]
print(f"standing time: {first_len/50:.2f} s → {final_len/50:.2f} s")
assert final_len > 2 * first_len, "should at least double within 300k steps"

# %% [markdown]
# ## T4 — Evaluate, export, replay
#
# Evaluate your checkpoint over 10 deterministic episodes (mean survival).
# Export it to ONNX and replay it with `PolicyRunner` — the Lab 5 machinery
# running YOUR network.

# %%
from stable_baselines3 import PPO

model = PPO.load("runs/lab6_t3/model.zip", device="cpu")
st = np.array([rollout(lambda o: model.predict(o, deterministic=True)[0], s)
               for s in range(10)])
print(f"trained: reward {st[:,0].mean():.1f}, survival {st[:,1].mean()/50:.2f} s")

from microduck_sim.export import export_sb3_to_onnx
from microduck_sim.policy import OnnxPolicy, PolicyRunner

path = export_sb3_to_onnx("runs/lab6_t3/model.zip", "runs/lab6_t3/model.onnx")
runner = PolicyRunner(OnnxPolicy(path), variant="walk")
runner.run(5.0)
print("ONNX replay upright:", runner.upright)

# %% [markdown]
# ## T5 — Challenge: the exploration knob 🏆
#
# Retrain 300k steps with `--log-std-init 0` (SB3's default, σ = 1.0) and
# overlay the two learning curves. Explain the gap in one sentence.
# Leaderboard: best final ep_len_mean at exactly 300k steps — any
# hyperparameter allowed except more steps.

# %%
subprocess.run(
    [sys.executable, "../../scripts/train_sb3.py", "--steps", "300000",
     "--out", "runs/lab6_t5", "--n-envs", "8", "--log-std-init", "0"],
    check=True, capture_output=True, text=True,
)
df5 = pd.read_csv("runs/lab6_t5/progress.csv")
plt.plot(df["time/total_timesteps"], df["rollout/ep_len_mean"] / 50, label="σ≈0.37 (tuned)")
plt.plot(df5["time/total_timesteps"], df5["rollout/ep_len_mean"] / 50, label="σ=1.0 (default)")
plt.xlabel("env steps"); plt.ylabel("standing time [s]"); plt.legend(); plt.show()
print(f"tuned {df['rollout/ep_len_mean'].iloc[-1]/50:.2f} s vs "
      f"default {df5['rollout/ep_len_mean'].iloc[-1]/50:.2f} s")
# σ=1.0 saturates the ±0.6 rad action box: exploration turns into full-power
# thrashing, so the agent rarely experiences quiet standing at all.
