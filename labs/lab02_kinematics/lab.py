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
# # Lab 2 — Kinematics & Joint Control · WORKSHEET
#
# Tasks T1–T5. `# TODO` cells are yours; `assert` cells check you.

# %%
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from microduck_sim.model import actuator_names, load_model, reset_to_keyframe
from microduck_sim.control import (
    chain, constant, goto, keyframe_ctrl, lerp, run_control_loop, sine,
)
from microduck_sim.viz import record, render_frame, show, show_video

model = load_model("full")
data = mujoco.MjData(model)
acts = actuator_names(model)
sit = keyframe_ctrl(model, "SIT")
stand = keyframe_ctrl(model, "STAND")

# %% [markdown]
# ## T1 — Variant detective
#
# Load `"walk"` and `"full"`. For each: print how many geoms *collide*
# (count `model.geom_contype != 0`), hold the SIT pose for 2 s with
# `run_control_loop` + `constant(...)`, and plot trunk height (`qpos[:, 2]`)
# over time for both variants in one figure.
#
# **Question:** why does the seated duck sink through the floor in one variant?

# %%
# TODO (use load_model(variant), mujoco.MjData, reset_to_keyframe, run_control_loop)

# %% [markdown]
# *Your answer:* ...

# %% [markdown]
# ## T2 — Smoothness quantified
#
# Implement `goto_linear(start, target, duration)` → a ctrl function that
# interpolates **linearly** from start to target (use `lerp`; hold the target
# after `duration`). Then, seated, turn the head (`head_yaw` target 0.8 rad)
# three ways — teleport (`constant`), your `goto_linear` (0.6 s), library
# `goto` (0.6 s) — and measure each run's **peak head speed**
# (`np.abs(log["qvel"][:, 6 + head_yaw_index]).max()`). Plot the three speed
# curves and print the peaks.

# %%
def goto_linear(start, target, duration):
    # TODO
    ...


head_yaw = acts.index("head_yaw")
turned = sit.copy()
turned[head_yaw] = 0.8

peaks = {}
# TODO: fill peaks = {"teleport": ..., "linear": ..., "min-jerk": ...}

# %%
assert peaks["teleport"] > peaks["linear"] and peaks["teleport"] > peaks["min-jerk"]
print("T2 ok:", {k: round(v, 2) for k, v in peaks.items()})

# %% [markdown]
# **Question:** linear and min-jerk peaks are close — so why prefer min-jerk?
# (Look at the *slope* of the speed curves at the start and end of the move.)
#
# *Your answer:* ...

# %% [markdown]
# ## T3 — FK: the beak draws
#
# Seated, nod with `neck_pitch` (sine, amplitude 0.4 rad, 0.5 Hz) for 2 s.
# From `log["site_xpos"]`, plot the `mouth_tip` path in the x–z plane
# (side view, `plt.axis("equal")`), then compute its x-extent (`np.ptp`) and
# total path length (sum of distances between consecutive points).

# %%
site_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)]
mouth = site_names.index("mouth_tip")

# TODO: run the nod, plot the path, compute x_extent and arc_length
x_extent = ...
arc_length = ...

# %%
assert x_extent > 0.15
print(f"T3 ok: one 0.4 rad joint sweeps the beak {x_extent:.2f} m — leverage!")

# %% [markdown]
# ## T4 — Choreograph a seated dance
#
# Your turn to be creative: design a dance with **at least two different
# joints**, at least 6 s long, using `sine` / `goto` / `chain`. The duck must
# stay seated throughout (trunk z > 0.05 m at every tick). Show the video.

# %%
# TODO: build `groove`, record ~6.5 s from SIT, show_video(frames), keep `log`

# %%
assert log["t"][-1] >= 6.0 and np.all(log["qpos"][:, 2] > 0.05), "fell out of the seat!"
print("T4 ok — nice moves")

# %% [markdown]
# ## T5 — Challenge: the longest stand 🏆
#
# Write `stand_time(ctrl_fn)`: reset to STAND, run up to 10 s, return the first
# time trunk z drops below 0.10 m (10.0 if never). Holding STAND survives
# ~0.8 s. **Beat it, open loop.** Try at least three ideas and report your
# best. *Hint: where is the mass, and how low can it sit?*
#
# Class leaderboard at the end of the session — deterministic sim, so results
# are directly comparable.

# %%
def stand_time(ctrl_fn, tmax=10.0):
    # TODO
    ...


baseline = stand_time(constant(stand))
# TODO: your ideas here; keep your best time in `best`
best = ...

# %%
assert best > baseline, "keep trying — the hint is about the center of mass"
print(f"T5 ok: baseline {baseline:.2f} s → yours {best:.2f} s")

# %% [markdown]
# Nobody reaches 10 s open-loop. Next week you'll write the ~15 lines of
# feedback that change everything.
