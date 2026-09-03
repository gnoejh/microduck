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
# # Lab 6 — Your First RL Training 🎓
#
# **Microduck Simulation Course · session 6 of 8**
#
# Lab 5's networks came pre-trained. Today you train your own:
# 1. The RL loop: agent ⇄ environment, reward as the only teacher
# 2. Gymnasium anatomy: our `MicroduckStand-v0`
# 3. Reward design (where the actual engineering lives)
# 4. PPO, and one exploration knob that makes-or-breaks it
# 5. Train, evaluate, export — your policy in the Lab 5 machinery

# %% [markdown]
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/microduck/blob/main/labs/lab06_first_rl/lecture.ipynb)
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

import microduck_sim.envs  # registers MicroduckStand-v0 / MicroduckWalk-v0
from microduck_sim.obs import NUM_JOINTS
from microduck_sim.viz import show, show_video

# %% [markdown]
# ## The RL loop
#
# ```
#        obs (61)                    action (14)
#   env ──────────► policy network ──────────► env.step()
#    ▲                                              │
#    └──────── reward (one number per tick) ◄───────┘
# ```
#
# Nobody tells the network *how* to stand. It tells it only *how well it's
# doing*, 50 times a second, and PPO adjusts the network to collect more of
# that number. Everything you know about the duck — contract, 50 Hz, STAND
# pose — carries over: the env speaks Lab 5's exact interface.
#
# ## The environment

# %%
env = gym.make("MicroduckStand-v0")
print("obs space:   ", env.observation_space)
print("action space:", env.action_space, "← ctrl offsets from DEFAULT_POSE, rad")

obs, _ = env.reset(seed=0)
obs2, reward, terminated, truncated, info = env.step(np.zeros(NUM_JOINTS, dtype=np.float32))
print("reward:", round(reward, 3))
print("terms: ", {k: round(v, 3) for k, v in info["reward_terms"].items()})

# %% [markdown]
# ## The reward is the spec
#
# ```python
# upright     = 1.0 · (−gravity_z)          # 1 vertical … 0 on the side
# height      = 1.0 · exp(−((z−0.118)/0.03)²)   # trunk at standing height
# still       = 0.3 · exp(−0.1·|joint_vel|²)    # calm joints
# action_rate = −0.05 · Σ|aₜ − aₜ₋₁|            # don't twitch
# # falling additionally costs −5 and ends the episode
# ```
#
# Read it as a job description. The agent will exploit any gap between what
# you wrote and what you meant — reward design IS the engineering (Lab 7 is
# entirely about this).
#
# Baselines before learning — what do zero-action and random policies score?

# %%
def rollout(policy_fn, seed):
    o, _ = env.reset(seed=seed)
    total, steps = 0.0, 0
    while True:
        o, r, term, trunc, _ = env.step(policy_fn(o))
        total, steps = total + r, steps + 1
        if term or trunc:
            return total, steps


zero = lambda o: np.zeros(NUM_JOINTS, dtype=np.float32)
rand = lambda o: env.action_space.sample()
for name, fn in [("zero-action (hold STAND)", zero), ("random", rand)]:
    stats = [rollout(fn, s) for s in range(10)]
    print(f"{name:26s} reward {np.mean([s[0] for s in stats]):7.1f}   "
          f"survives {np.mean([s[1] for s in stats]) / 50:.2f} s")

# %% [markdown]
# Zero-action is Lab 1's open-loop stand: ~0.8 s. Random is worse. The bar to
# beat is low — but beating it *well* means standing the full 10 s episode.
#
# ## PPO in one slide
#
# **Proximal Policy Optimization**: collect experience with the current
# (stochastic) policy → estimate which actions did better than expected →
# nudge the network toward those, *clipping* the change so one bad batch
# can't wreck it → repeat. The same algorithm Pollen used — theirs just ran
# 4096 ducks in parallel; we run 8 (one process per env, `SubprocVecEnv` —
# on Colab's 2 vCPUs use `--n-envs 2`, the default adapts).
#
# **The knob that bit us:** SB3 starts with exploration noise σ = 1.0 rad.
# Our actions cap at ±0.6 rad — pure noise saturation, and the poor duck gets
# violently shaken all episode. Setting `log_std_init=-1` (σ≈0.37) measurably
# accelerated learning in our runs. Exploration scale must fit the action
# scale — remember that on any robot you ever train.
#
# ## Train it
#
# `scripts/train_sb3.py` wraps SB3 PPO with laptop-tuned defaults. In class
# we run ~3 minutes' worth; the instructor's 1M-step run (≈7 min on a laptop)
# is committed for the impatient:

# %%
# Live-training cell (~2 min). Uncomment to run in class:
# !uv run scripts/train_sb3.py --steps 200000 --out runs/lab6_live

df = pd.read_csv("../../assets/checkpoints/stand_progress.csv")
plt.plot(df["time/total_timesteps"], df["rollout/ep_len_mean"] / 50.0)
plt.axhline(10.0, ls="--", c="gray", label="full episode (10 s)")
plt.xlabel("env steps"); plt.ylabel("mean standing time [s]")
plt.title("PPO learns to stand (instructor run, 1M steps)"); plt.legend(); plt.show()

# %% [markdown]
# ## Watch what it learned

# %%
from stable_baselines3 import PPO

model = PPO.load("../../assets/checkpoints/stand_ppo.zip", device="cpu")
frames = []
o, _ = env.reset(seed=123)
renderer_env = env.unwrapped
for i in range(500):
    action, _ = model.predict(o, deterministic=True)
    o, r, term, trunc, _ = env.step(action)
    if i % 2 == 0:
        frames.append(renderer_env.render())
    if term or trunc:
        break
show_video(frames)
print(f"survived {i / 50:.1f} s of a 10 s episode")

# %% [markdown]
# ## Export: your network in the Lab 5 machinery
#
# One call turns the SB3 checkpoint into a contract-compatible ONNX — the
# same format Pollen ships, loadable by `PolicyRunner`, publishable to the
# Hugging Face Hub (Lab 8):

# %%
from microduck_sim.export import export_sb3_to_onnx
from microduck_sim.policy import OnnxPolicy, PolicyRunner

path = export_sb3_to_onnx("../../assets/checkpoints/stand_ppo.zip",
                          "../../runs/my_stand.onnx")
runner = PolicyRunner(OnnxPolicy(path))
runner.run(5.0)
print(f"ONNX replay in the Lab 5 runner: upright after 5 s = {runner.upright}")
print("contract metadata:", OnnxPolicy(path).metadata["observation_names"])

# %% [markdown]
# The exported file is the same format the browser runs. The instructor's copy of
# this exact checkpoint is already in the playground — drag the duck and feel how
# little push it tolerates (Lab 8 measures it: 0.6 N). Yours gets there through a
# pull request in Lab 8.

# %%
from microduck_sim.web import show_playground

show_playground("walk", policy="stand_ppo_course_cpu", height=480)

# %% [markdown]
# ## Recap
#
# - RL = reward as the only teacher; the env is the exam, the reward the rubric
# - `MicroduckStand-v0` speaks the shipped-policy contract (61/14/50 Hz)
# - exploration noise must fit the action scale (σ 1.0 → 0.37 mattered)
# - train → evaluate → export ONNX → replay/publish: the full workflow
#
# **Worksheet: T1–T5.** Next week: walking — where reward design gets real.
