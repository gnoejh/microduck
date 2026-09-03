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
# # Lab 7 — Reward Shaping & Walking 🚶
#
# **Microduck Simulation Course · session 7 of 8**
#
# Standing rewarded *a state*. Walking must reward *a behavior* — and the
# agent will take every shortcut you leave open:
# 1. The walking task and its reward
# 2. What our CPU budget actually learns (honestly)
# 3. Reward shaping: the art of saying what you mean
# 4. Domain randomization: why Lab 5's policies transfer
# 5. The GPU track: same recipe, three orders of magnitude more ducks

# %% [markdown]
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/microduck/blob/main/labs/lab07_reward_shaping/lecture.ipynb)
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
from microduck_sim.obs import NUM_JOINTS
from microduck_sim.policy import PolicyRunner
from microduck_sim.viz import show_video

# %% [markdown]
# ## The walking task
#
# `MicroduckWalk-v0`: each episode samples a forward-velocity command
# vx ∈ [0, 0.25] m/s (10% of episodes: exactly 0 — standing stays in the
# job). The command rides in obs slots 48:51, exactly where the shipped
# walker reads it. The reward:
#
# ```python
# track_vx    = 2.0 · exp(−((vx_world − vx_cmd)/0.1)²)   # the actual job
# upright     = 0.5 · (−gravity_z)                        # stay a biped
# energy      = −0.001 · |joint_vel|²                     # don't buzz
# action_rate = −0.05 · Σ|aₜ − aₜ₋₁|                      # don't twitch
# ```
#
# Note what is *not* rewarded: gait symmetry, foot clearance, step rhythm.
# Upstream's recipe shapes those too — we'll feel their absence shortly.
#
# ## A reward can be audited before training
#
# We hold a policy that objectively walks: Lab 5's `BEST_alpha_walking`.
# Drive it manually through OUR env reward and see what it earns — if a
# "good" policy scores badly, the reward is wrong, not the policy:

# %%
env = gym.make("MicroduckWalk-v0")
base = env.unwrapped


def score_runner_policy(behavior, episodes=5):
    """Run a shipped ONNX policy inside the env, scoring it with OUR reward."""
    from microduck_sim.policy import OnnxPolicy

    policy = OnnxPolicy(behavior)
    totals = []
    for seed in range(episodes):
        obs, _ = env.reset(seed=seed)
        total = 0.0
        while True:
            action = np.clip(policy.act(obs), -0.6, 0.6)
            obs, r, term, trunc, _ = env.step(action.astype(np.float32))
            total += r
            if term or trunc:
                break
        totals.append(total)
    return np.mean(totals)


print(f"shipped walker under our reward:      {score_runner_policy('walk'):7.1f}")
print(f"shipped GET-UP (never walks!) scores: {score_runner_policy('stand'):7.1f}")

# %% [markdown]
# Read that again: a policy that **ignores the velocity command entirely and
# just stands** earns ~85% of the walker's reward. Our contract under-prices
# the job — upright+survival pays almost as well as working. An agent
# trained on this may rationally choose to... mostly stand. Keep that in
# mind when you see what CPU training produces below. (Fixing this pricing
# is worksheet T3.)
#
# ## What a laptop CPU learns
#
# The instructor's 2M-step CPU run of `MicroduckWalk-v0` (≈15 min) is
# committed. The learning curve, and the behavior it bought:

# %%
df = pd.read_csv("../../assets/checkpoints/walk_progress.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(df["time/total_timesteps"], df["rollout/ep_len_mean"] / 50)
ax[0].set_xlabel("env steps"); ax[0].set_ylabel("episode length [s]")
ax[1].plot(df["time/total_timesteps"], df["rollout/ep_rew_mean"])
ax[1].set_xlabel("env steps"); ax[1].set_ylabel("episode reward")
plt.tight_layout(); plt.show()

# %%
from stable_baselines3 import PPO

model = PPO.load("../../assets/checkpoints/walk_ppo.zip", device="cpu")
obs, _ = env.reset(seed=3)
base.command[0] = 0.2  # ask for 0.2 m/s
frames, xs = [], []
for i in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, _ = env.step(action)
    xs.append(base.data.qpos[0])
    if i % 2 == 0:
        frames.append(base.render())
    if term or trunc:
        break
show_video(frames)
print(f"commanded 0.2 m/s, achieved {(xs[-1] - xs[0]) / (len(xs) / 50):.3f} m/s "
      f"over {len(xs)/50:.1f} s")

# %% [markdown]
# Look at that: **~0.015 m/s**. Two million steps, 100% survival — and the
# agent barely moves. It did not fail; it *optimized*. It found exactly the
# loophole our audit exposed: standing pays ~90% of walking's reward with
# none of the risk, so a rational agent... stands. The audit predicted the
# training outcome before we spent a single step on it — that is why you
# audit first.
#
# Three levers actually buy walking, in increasing order of cost:
#
# 1. **better shaping** (free — this lab: make motion the money-maker),
# 2. **more experience** (Colab GPU / the official pipeline — below),
# 3. **domain randomization + curriculum** (upstream's full recipe).
#
# ## Reward shaping: the agent is a lawyer
#
# Every reward is a contract, and the agent reads only the letter. Classics
# of the genre (all real, all in the literature):
#
# - reward forward velocity → dive forward (velocity spike, then floor)
# - reward not-falling only → stand perfectly still forever
# - big energy penalty → refuse to move at all ("safest step is no step")
# - unpenalized twitching → 25 Hz limb vibration that exploits the simulator
#
# The worksheet has you *cause* two of these on purpose, by editing weights.
# The fix is never "one weird trick": it's iterating measurable behavior
# metrics (tracking error, action smoothness, symmetry) against the terms.
#
# ## Domain randomization — why Lab 5 transferred
#
# Our sim ≠ mjlab ≠ reality. Upstream trains with per-episode randomization —
# battery voltage, command delay, friction, ±1° gear backlash — so the policy
# meets a *distribution* of physics and learns not to overfit one simulator.
# That's why their ONNX survived our plain MuJoCo unchanged (Lab 5), and it's
# the exact idea you'll stress-test in Lab 8 with the backlash model variants.
#
# ## The GPU track (`stage2/colab_gpu_track.ipynb`)
#
# Same PPO, same contract — but mjlab/MuJoCo Warp steps **thousands of ducks
# in lockstep on the GPU**. Switch the Colab runtime to a T4 and run Pollen's
# own pipeline; a 20-minute run is enough to *see* the difference:
#
# ```
# git clone https://github.com/pollen-robotics/microduck_rl && cd microduck_rl && uv sync
# uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 1024 --agent.max-iterations 300
# uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <entity/project/run>
# ```
#
# The export is a contract ONNX (normalizer baked in), so it drops straight into
# `PolicyRunner` and the playground. Compare three walkers on one scorecard —
# our 2M-step CPU run, a 20-minute T4 run, Pollen's shipped policy:
#
# | policy | steps/hour | survival | speed at 0.25 |
# |---|---|---|---|
# | course walk_ppo (CPU, 8 envs) | ~10 M | 1.0 | 0.015 (stands!) |
# | microduck_rl on T4 (1024 envs) | *measure it* | | |
# | shipped `walk` | (4096 envs, hours) | 1.0 | ≈0.11 |
#
# Next semester's Isaac Sim half is this idea, industrialized: parallel
# physics as the substrate of modern robot learning. You already speak it.
#
# ## Recap
#
# - audit rewards with known-good policies before training on them
# - CPU budget ⇒ locomotion without elegance; shaping is the free lever
# - the agent optimizes the letter of the reward — write it like a contract
# - domain randomization turns "one sim" into "a family of sims" ⇒ transfer
#
# **Worksheet: T1–T5.** Next week: evaluation, robustness, and shipping your
# policy to the Hub.
