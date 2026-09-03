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
# # Lab 7 — Reward Shaping & Walking · SOLUTION
#
# (T5's training cell costs ~4 minutes of CPU by design.)

# %% [markdown]
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/microduck/blob/main/labs/lab07_reward_shaping/solution.ipynb)
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
from microduck_sim import colab; colab.setup("labs/lab07_reward_shaping")

# %%
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import microduck_sim.envs  # noqa: F401
from microduck_sim.envs import MicroduckWalkEnv
from microduck_sim.obs import NUM_JOINTS
from microduck_sim.policy import OnnxPolicy


def score(env, policy_fn, episodes=4):
    totals = []
    for seed in range(episodes):
        obs, _ = env.reset(seed=seed)
        tot = 0.0
        while True:
            obs, r, term, trunc, _ = env.step(policy_fn(obs))
            tot += r
            if term or trunc:
                break
        totals.append(tot)
    return float(np.mean(totals))


def onnx_fn(policy):
    return lambda o: np.clip(policy.act(o), -0.6, 0.6).astype(np.float32)


walker, getup = OnnxPolicy("walk"), OnnxPolicy("stand")
zero_fn = lambda o: np.zeros(NUM_JOINTS, dtype=np.float32)

# %% [markdown]
# ## T1 — Audit the reward with policies you trust
#
# Score three known policies under the default `MicroduckWalk-v0` reward:
# the shipped walker, the shipped get-up (a non-walker), and zero-action.
# Compute the walker/get-up ratio. What does it tell you?

# %%
env = gym.make("MicroduckWalk-v0")
w = score(env, onnx_fn(walker))
s = score(env, onnx_fn(getup))
z = score(env, zero_fn)
print(f"walker {w:.0f}   get-up {s:.0f}   zero-action {z:.0f}   ratio walker/get-up {w/s:.2f}")
# Ratio ≈ 1.1: a policy that never walks earns ~90% of the walker's pay.
# The reward pays mostly for being upright and alive — the job (tracking) is
# a small bonus on top. An agent could rationally learn to mostly stand.

# %% [markdown]
# ## T2 — Can weights fix it?
#
# Use the `reward_weights` knob to down-weight `upright` (×0.2) and up-weight
# `track_vx` (×2). Re-audit. How far can the ratio go?

# %%
for weights in [{"upright": 0.2}, {"upright": 0.2, "track_vx": 2.0}]:
    env_w = gym.make("MicroduckWalk-v0", reward_weights=weights)
    w2, s2 = score(env_w, onnx_fn(walker)), score(env_w, onnx_fn(getup))
    print(f"{str(weights):42s} ratio {w2/s2:.2f}")
# Ceiling ≈ 1.3. Weights rescale the terms but every term still pays a
# stander almost as much as a walker — the STRUCTURE is wrong, not the
# coefficients.

# %% [markdown]
# ## T3 — Fix the structure
#
# Two structural changes, applied by subclassing:
# (a) audit at a command the walker can actually attempt (fixed vx = 0.25 —
# see T4 for why), (b) add a `progress` term that pays for *actual forward
# motion* whenever motion is commanded:
# `2.0 · clip(vx_world/0.25, −1, 1)` if cmd > 0.05 else 0.
# Target: walker/get-up ratio ≥ 2.

# %%
class FixedCmdWalk(MicroduckWalkEnv):
    def _sample_command(self):
        self.command[:] = 0.0
        self.command[0] = 0.25


class ProgressWalk(FixedCmdWalk):
    def _reward(self, action):
        terms = super()._reward(action)
        vx, cmd = float(self.data.qvel[0]), float(self.command[0])
        terms["progress"] = 2.0 * np.clip(vx / 0.25, -1, 1) * (1.0 if cmd > 0.05 else 0.0)
        return terms


for name, e in [("fixed cmd", FixedCmdWalk()), ("fixed cmd + progress", ProgressWalk())]:
    w3, s3 = score(e, onnx_fn(walker)), score(e, onnx_fn(getup))
    print(f"{name:22s} walker {w3:.0f} vs get-up {s3:.0f}  ratio {w3/s3:.2f}")
ratio = w3 / s3
assert ratio >= 2.0
# 1.11 → 1.30 → 1.49 → 3.15: weights < evaluation distribution < term design.
# The most powerful reward edits change WHAT is measured, not how much.

# %% [markdown]
# ## T4 — Map the walker's envelope (the deadband)
#
# Sweep commanded vx ∈ {0.05, 0.10, …, 0.30} on the shipped walker
# (`evaluate_walking`, 3 episodes, 6 s) and plot commanded vs achieved speed.
# Then repeat at vx = 0.20 on the `backlash` variant. Explain both findings.

# %%
from microduck_sim.evaluate import evaluate_walking

cmds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
achieved = [evaluate_walking("walk", vx=c, episodes=3, duration=6.0)["mean_speed"]
            for c in cmds]
plt.plot(cmds, achieved, "o-"); plt.plot(cmds, cmds, "--", c="gray", label="perfect tracking")
plt.xlabel("commanded vx [m/s]"); plt.ylabel("achieved vx [m/s]"); plt.legend(); plt.show()
print(dict(zip(cmds, np.round(achieved, 3))))

bl = evaluate_walking("walk", vx=0.20, episodes=3, duration=6.0, variant="backlash")
print(f"vx=0.20 on backlash variant: achieved {bl['mean_speed']:.3f} m/s")
# Finding 1 — a DEADBAND: below ~0.25 the walker just stands (the gamepad
# only ever sent 0.25, so intermediate commands are out-of-distribution).
# Finding 2 — on the backlash model the same 0.20 command WALKS: the ±1° gear
# chatter acts as dither, continuously perturbing the policy out of its
# stand-still attractor. Evaluation reveals the training distribution.

# %% [markdown]
# ## T5 — Challenge: train on your reward 🏆
#
# Train 500k steps on the default walk reward and once with your structural
# fix, then compare achieved speed at cmd = 0.25. Leaderboard: lowest
# tracking error at 500k steps. (GPU track: `stage2/colab_gpu_track.ipynb` runs
# the official pipeline that made the policies you just audited.)

# %%
import subprocess, sys

subprocess.run([sys.executable, "../../scripts/train_sb3.py",
                "--env", "MicroduckWalk-v0", "--steps", "500000",
                "--out", "runs/lab7_default"], check=True,
               capture_output=True, text=True)
subprocess.run([sys.executable, "../../scripts/train_sb3.py",
                "--env", "MicroduckWalk-v0", "--steps", "500000",
                "--rw", "upright=0.2", "--rw", "track_vx=2.0",
                "--out", "runs/lab7_shaped"], check=True,
               capture_output=True, text=True)

from stable_baselines3 import PPO

for name in ["lab7_default", "lab7_shaped"]:
    model = PPO.load(f"runs/{name}/model.zip", device="cpu")
    e = gym.make("MicroduckWalk-v0")
    base = e.unwrapped
    obs, _ = e.reset(seed=0)
    base.command[0] = 0.25
    x0 = float(base.data.qpos[0]); n = 0
    for _ in range(500):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = e.step(a)
        n += 1
        if term or trunc:
            break
    speed = (float(base.data.qpos[0]) - x0) / (n / 50)
    print(f"{name}: survived {n/50:.1f} s, achieved {speed:+.3f} m/s at cmd 0.25")
# Expect modest speeds either way at this budget — the honest CPU story.
# The point of the comparison is the DIRECTION of the change, not elegance.
