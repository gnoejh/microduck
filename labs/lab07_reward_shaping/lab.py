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
# # Lab 7 — Reward Shaping & Walking · WORKSHEET
#
# The agent optimizes the letter of your reward. Today you audit, break, and
# fix that letter. T1–T5. (T5 trains ~2×4 min — start it early, analyze while
# it runs.)

# %%
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import microduck_sim.envs  # noqa: F401
from microduck_sim.envs import MicroduckWalkEnv
from microduck_sim.obs import NUM_JOINTS
from microduck_sim.policy import OnnxPolicy


def onnx_fn(policy):
    return lambda o: np.clip(policy.act(o), -0.6, 0.6).astype(np.float32)


walker, getup = OnnxPolicy("walk"), OnnxPolicy("stand")
zero_fn = lambda o: np.zeros(NUM_JOINTS, dtype=np.float32)

# %% [markdown]
# ## T1 — Audit the reward with policies you trust
#
# Write `score(env, policy_fn, episodes=4) -> mean episode reward` (seeded
# resets 0..3). Score the shipped walker, the shipped get-up (which never
# walks), and zero-action under default `MicroduckWalk-v0`. Print the
# walker/get-up ratio.

# %%
def score(env, policy_fn, episodes=4):
    # TODO
    ...


env = gym.make("MicroduckWalk-v0")
# TODO: three scores + ratio
ratio_default = ...

# %%
assert ratio_default < 1.5, "you should find the reward barely prices walking!"
print(f"T1 ok — ratio {ratio_default:.2f}: a non-walker earns ~90% of a walker's pay")

# %% [markdown]
# **Question:** which reward terms is the get-up policy collecting?
#
# *Your answer:* ...

# %% [markdown]
# ## T2 — Can weights fix it?
#
# `gym.make("MicroduckWalk-v0", reward_weights={"upright": 0.2, "track_vx": 2.0})`
# rescales terms by name. Try at least two weightings and re-compute the
# ratio. Report the best you can reach.

# %%
# TODO
ratio_weighted = ...

# %%
assert ratio_weighted < 2.0, "weights alone shouldn't get you past ~1.5 — that's the lesson"
print(f"T2 ok — best weighted ratio {ratio_weighted:.2f}")

# %% [markdown]
# **Question:** why can't multipliers fix this reward?
#
# *Your answer:* ...

# %% [markdown]
# ## T3 — Fix the structure
#
# Subclass `MicroduckWalkEnv` and change WHAT is measured:
# (a) `_sample_command`: fixed vx = 0.25 (see T4 for why 0.25),
# (b) `_reward`: add a `progress` term ≈ `2.0 · clip(vx_world/0.25, −1, 1)`,
# paid only when motion is commanded.
# Re-audit walker vs get-up. Target: ratio ≥ 2.

# %%
class MyWalkEnv(MicroduckWalkEnv):
    # TODO: override _sample_command and _reward
    ...


# TODO: audit
ratio_structural = ...

# %%
assert ratio_structural >= 2.0
print(f"T3 ok — ratio {ratio_structural:.2f}. Structure beats coefficients.")

# %% [markdown]
# ## T4 — Map the walker's envelope
#
# Sweep commanded vx ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30} through
# `evaluate_walking("walk", vx=..., episodes=3, duration=6.0)` and plot
# commanded vs achieved speed. Then evaluate vx = 0.20 once more with
# `variant="backlash"`. Two surprises — explain both.
# (Hint for the first: what command did the playground's gamepad send?
# Hint for the second: what does constant mechanical chatter do to a policy
# stuck in a stand-still attractor?)

# %%
from microduck_sim.evaluate import evaluate_walking

# TODO

# %% [markdown]
# *Finding 1 (the flat region):* ...
#
# *Finding 2 (backlash at 0.20):* ...

# %% [markdown]
# ## T5 — Challenge: train on your reward 🏆
#
# Train 500k steps (≈4 min each) on (a) the default walk reward and (b) your
# choice of weights (`--rw term=mult`). Compare achieved speed at cmd = 0.25
# over one 10 s deterministic episode each. Leaderboard: lowest |0.25 −
# achieved|. GPU ambitions → `stage2/colab_train.ipynb`.

# %%
# TODO

# %% [markdown]
# ---
# Next week: evaluation as deployment — push tests, variant sweeps, and
# publishing your best policy to the Hugging Face Hub.
