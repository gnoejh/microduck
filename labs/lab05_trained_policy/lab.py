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
# # Lab 5 — Running a Trained Policy · WORKSHEET
#
# Today you command the actual neural networks that ship with the robot.
# T1–T5.

# %%
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from microduck_sim.obs import OBS_SIZE, ObsBuilder
from microduck_sim.policy import BEHAVIORS, OnnxPolicy, PolicyRunner
from microduck_sim.viz import record_policy, show_video

# %% [markdown]
# ## T1 — Inspect the artifacts
#
# Loop over `BEHAVIORS`, load each `OnnxPolicy`, and print the input/output
# names and shapes (`p.session.get_inputs()[0]`, `.get_outputs()[0]`).
# Assert every input ends in 61.

# %%
# TODO

# %% [markdown]
# ## T2 — Projected gravity is your Lab 4 pitch
#
# Implement `my_projected_gravity(quat_wxyz)`: rotate the world down-vector
# [0, 0, −1] by the **conjugated** trunk quaternion (`mujoco.mju_rotVecQuat`;
# conjugate = negate x, y, z). Run the stand policy 2 s, checking every tick:
# (a) yours matches `runner.obs_builder.projected_gravity(runner.data)`,
# (b) `asin(gx)` equals Lab 4's pitch (`quat_to_pitch_roll`) to 1e-9.
# The trunk quaternion is `runner.data.xquat[runner.obs_builder.trunk_id]`.

# %%
def my_projected_gravity(quat_wxyz):
    # TODO
    ...


# TODO: the 2 s verification loop
print("T2 ok")

# %% [markdown]
# **Question:** why do you think the training team fed the network this
# 3-vector instead of pitch/roll angles?
#
# *Your answer:* ...

# %% [markdown]
# ## T3 — Drive a square
#
# Script the walker around a square: 4 × [4 s at vx = 0.25, then ~1.6 s at
# wz = 1.0 (≈ 90°)], collecting `data.qpos[:2]` every tick. Plot the x–y
# path (`plt.axis("equal")`) and print the closure error (distance between
# start and end).
#
# **Question:** why doesn't the square close?

# %%
# TODO
closure = ...

# %%
assert closure < 1.0, "should end within 1 m of the start"
print(f"T3 ok: closure error {closure:.2f} m")

# %% [markdown]
# *Your answer:* ...

# %% [markdown]
# ## T4 — Measure the gait with an FFT
#
# Log 10 s of walking (vx = 0.25) with `runner.run(10.0)`. FFT the
# `left_hip_pitch` action (index 2; subtract the mean; `np.fft.rfft`,
# `np.fft.rfftfreq` with d = 0.02). Report the dominant frequency (skip the
# DC bin) and the *physical* correlation between left (2) and right (11) hip
# actions. Trap from Lab 2: left/right joints use mirrored sign conventions —
# negate one side, or antiphase will masquerade as in-phase.

# %%
# TODO
f_step = ...
corr = ...

# %%
assert 0.5 < f_step < 5.0 and corr < 0
print(f"T4 ok: gait at {f_step:.2f} Hz, hips antiphase (r = {corr:+.2f})")

# %% [markdown]
# ## T5 — Challenge: overdrive & the full pipeline 🏆
#
# (a) The walker was trained up to vx = 0.25 m/s. Command more (0.05 steps):
# find the highest vx surviving 10 s, and the *achieved* speed at each step
# (Δx / 10). Plot commanded vs achieved. Where does extrapolation break?
#
# (b) Build the robot's real autonomy loop in ~25 lines: walk → shove flat
# (8 N on the trunk for 0.2 s) → detect the fall (projected gravity z > −0.5)
# → swap to the `stand` get-up policy with zeroed commands & last_action →
# detect recovery (gz < −0.85) → swap back to walking. Assert the duck ends
# the 20 s run walking and upright.

# %%
# TODO (a)

# %%
# TODO (b)
final_state = ...

# %%
assert final_state == "walking"
print("T5 ok — fell, got up, walked on. That's the whole robot.")

# %% [markdown]
# ---
# Everything today used *someone else's* network. Next week: reinforcement
# learning — you train your own, and discover why reward design is an art.
