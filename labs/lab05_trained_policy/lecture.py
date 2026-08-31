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
# # Lab 5 — Running a Trained Policy 🧠
#
# **Microduck Simulation Course · session 5 of 8**
#
# You built controllers by hand (Lab 3) and estimators by hand (Lab 4).
# Pollen Robotics instead *trained* controllers — neural networks — and ships
# them with the robot. Today we run **the actual shipped networks** in our sim:
# 1. What a policy is
# 2. The observation contract (everything you built feeds it)
# 3. Walking, steering, sitting — on command
# 4. What open-loop couldn't do: standing up
# 5. Reading a network's mind (obs/action traces)

# %%
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from microduck_sim.obs import DEFAULT_POSE, JOINT_NAMES, OBS_SIZE, ObsBuilder
from microduck_sim.policy import BEHAVIORS, OnnxPolicy, PolicyRunner
from microduck_sim.viz import record_policy, show_video

# %% [markdown]
# ## What is a policy?
#
# A **policy** is a function: observation in → action out. Here it's a small
# neural network, trained with reinforcement learning (Lab 6's topic) in
# massively parallel simulation, then frozen into an ONNX file. The robot —
# and our sim — just calls it 50 times per second:
#
# ```
# obs (61 numbers) ──► neural net ──► action (14 numbers)
# ctrl = DEFAULT_POSE + action        # offsets from the STAND pose
# ```
#
# We vendored all nine shipped behaviors:

# %%
for name, file in BEHAVIORS.items():
    print(f"{name:14s} {file}")

policy = OnnxPolicy("walk")
inp, out = policy.session.get_inputs()[0], policy.session.get_outputs()[0]
print(f"\ninput  {inp.name}: {inp.shape}   output {out.name}: {out.shape}")

# %% [markdown]
# ## The observation contract
#
# All policies read the same **61-dim observation** — and you already know
# every ingredient:
#
# | slice | content | you built this in |
# |-------|---------|-------------------|
# | 0:3   | gyro (angular velocity) | Lab 4 |
# | 3:6   | projected gravity (tilt, as a vector) | Lab 4 (it's your pitch, generalized) |
# | 6:20  | joint positions − STAND pose | Lab 1–2 |
# | 20:34 | joint velocities | Lab 2 |
# | 34:48 | previous action | — (gives the net short-term memory) |
# | 48:61 | command: [vx, vy, wz, 4 head slots, zeros] | Lab 2's "commands" |
#
# **Projected gravity** replaces pitch/roll: world "down" expressed in the
# trunk frame. Upright → [0, 0, −1]; lying on its face → [−1, 0, 0]. It's
# exactly what your complementary filter estimated, minus the angles.
#
# No normalization needed — it's baked into the ONNX. Raw physics in, done.

# %%
runner = PolicyRunner("stand")
obs = runner.obs_builder.build(runner.data, np.zeros(14), np.zeros(13))
print("gyro:        ", obs[0:3].round(3))
print("proj gravity:", obs[3:6].round(3), "← upright ⇒ ≈ [0 0 −1]")
print("joint pos Δ: ", obs[6:20].round(3), "← at STAND ⇒ ≈ 0")

# %% [markdown]
# ## It walks
#
# `PolicyRunner` reproduces the training loop exactly: physics at 5 ms,
# inference every 4th step (50 Hz). Command slot 0 = forward velocity:

# %%
runner = PolicyRunner("walk")
runner.command[0] = 0.25  # m/s, the trained maximum
frames, log = record_policy(runner, duration=8.0)
show_video(frames)
print(f"displacement: {runner.data.qpos[0]:+.2f} m forward, upright: {runner.upright}")

# %% [markdown]
# Seven weeks ago that was 800 g of falling plastic; now it *walks*. And it
# steers — change the command mid-run, like a gamepad stick would:

# %%
runner = PolicyRunner("walk")
runner.command[0] = 0.25
frames1, _ = record_policy(runner, 3.0)
runner.command[2] = 1.0   # turn left, rad/s
frames2, _ = record_policy(runner, 3.0)
runner.command[2] = -1.0  # turn right
frames3, _ = record_policy(runner, 3.0)
show_video(frames1 + frames2 + frames3)

# %% [markdown]
# ## The Lab 2 grudge match: standing up
#
# Open-loop stand-up face-planted every single time. The `sitstand` policy —
# command slot 0 as a posture flag — does the transition *both ways*:

# %%
runner = PolicyRunner("sitstand")
runner.command[0] = 1.0   # sit
frames1, _ = record_policy(runner, 3.0)
runner.command[0] = 0.0   # stand back up
frames2, _ = record_policy(runner, 3.0)
show_video(frames1 + frames2)
print(f"back on its feet: z = {runner.data.qpos[2]:.3f} m, upright: {runner.upright}")

# %% [markdown]
# ## Falling is allowed now
#
# Knock the walker flat, then hand control to the get-up policy
# (`stand`, commands zeroed) — the same recovery pipeline the real robot runs:

# %%
runner = PolicyRunner("walk")
runner.command[0] = 0.25
trunk = mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
frames1, _ = record_policy(runner, 2.0)
runner.data.xfrc_applied[trunk, 0] = 8.0   # a mighty shove
for _ in range(10):
    runner.step()
runner.data.xfrc_applied[trunk, 0] = 0.0
frames2, _ = record_policy(runner, 1.0)     # down it goes

runner.policy = OnnxPolicy("stand")         # swap in the get-up network
runner.command[:] = 0.0
runner.last_action[:] = 0.0
frames3, _ = record_policy(runner, 3.0)
show_video(frames1 + frames2 + frames3)
print(f"recovered: {runner.upright}")

# %% [markdown]
# ## Reading the network's mind
#
# The policy is a black box, but its inputs and outputs aren't. During a walk,
# the actions are *rhythmic* — the network taught itself a gait:

# %%
runner = PolicyRunner("walk")
runner.command[0] = 0.25
log = runner.run(6.0)
t = log["time"] - log["time"][0]
plt.plot(t, log["action"][:, 2], label="left_hip_pitch")
plt.plot(t, log["action"][:, 3], label="left_knee")
plt.plot(t, -log["action"][:, 11], label="right_hip_pitch (sign-flipped)")
plt.xlabel("t [s]"); plt.ylabel("action (rad offset from STAND)")
plt.legend(); plt.title("the learned gait, visible in the actions"); plt.show()

# %% [markdown]
# Left and right hips oscillate in antiphase — a walking gait, discovered by
# trial and error in simulation, not programmed. (Note the sign flip on the
# right hip: Lab 2's mirrored joint conventions strike again — raw left/right
# curves *look* in-phase until you undo the mirror.) In the worksheet you'll
# measure the exact step frequency with an FFT.
#
# ## Why this worked at all (sim2sim)
#
# These networks were trained in **mjlab** (MuJoCo Warp, GPU, 4096 parallel
# ducks) — a *different simulator* than our plain CPU MuJoCo. They transfer
# because training randomized physics (Lab 7's topic), and because both sides
# honor the same contract: 61 obs, 14 actions, 50 Hz, 5 ms timestep. The same
# recipe carries the policy onto the real robot — the step we replace with
# evaluation (Lab 8), having no hardware and no regrets.
#
# ## Recap
#
# - policy = trained function obs → action; runs at 50 Hz like your controllers
# - the 61-dim obs is made of things you built by hand in Labs 1–4
# - shipped policies walk, steer, sit, stand up, and get up off the floor
# - actions show a rhythmic, *learned* gait
# - library: `ObsBuilder`, `OnnxPolicy`, `PolicyRunner`; try
#   `uv run scripts/play_policy.py` at home — arrow keys drive the duck
#
# **Worksheet: T1–T5.** Next week you *train* one of these yourself.
