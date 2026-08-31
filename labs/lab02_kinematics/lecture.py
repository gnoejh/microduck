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
# # Lab 2 — Kinematics & Joint Control 🦵
#
# **Microduck Simulation Course · session 2 of 8**
#
# Last week the duck mostly fell over. Today we make it *move on purpose*:
# 1. Model variants (why last week's duck sank through the floor)
# 2. Joint space: a pose is a vector
# 3. The 50 Hz control loop
# 4. Trajectories: from A to B *smoothly*
# 5. Forward kinematics: where is the beak?
# 6. Rhythm and routines — and one honest failure

# %% [markdown]
# ## Model variants
#
# Lab 1 used the `"walk"` variant, where **only the feet have collision shapes**
# (12× faster to simulate — it exists for training). Today the duck must *sit on
# the ground*, so we load `"full"`, where trunk and head collide too:

# %%
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from microduck_sim.model import actuator_names, load_model, reset_to_keyframe
from microduck_sim.control import (
    chain, constant, goto, keyframe_ctrl, run_control_loop, sine,
)
from microduck_sim.viz import record, render_frame, show, show_grid, show_video

for variant in ["walk", "full"]:
    m = load_model(variant)
    print(f"{variant:5s}: {m.ngeom} geoms, {(m.geom_contype != 0).sum()} of them collide")

model = load_model("full")
data = mujoco.MjData(model)

# %% [markdown]
# > Lesson: **a simulation model is a set of choices**, not the truth. Pollen
# > ships six variants of the same duck, each tuned to a purpose.

# %% [markdown]
# ## Joint space
#
# A full-body pose is a vector: one target angle per actuator (14). The
# keyframes from Lab 1 are exactly such vectors:

# %%
stand = keyframe_ctrl(model, "STAND")
sit = keyframe_ctrl(model, "SIT")
for name, s, q in zip(actuator_names(model), stand, sit):
    print(f"{name:16s} STAND {s:+.2f}   SIT {q:+.2f}")

# %% [markdown]
# Notice left/right mirror each other with flipped signs — remember that
# whenever you design symmetric motions.
#
# ## The 50 Hz control loop
#
# The real robot's brain picks new motor targets 50×/s while physics runs at
# 500 Hz — one control tick = 10 physics substeps. `run_control_loop` implements
# that contract: give it `ctrl_fn(t) -> ctrl vector`, it steps and logs.
#
# Simplest ctrl_fn: hold one pose. Seated, that is *actually stable*:

# %%
reset_to_keyframe(model, data, "SIT")
log = run_control_loop(model, data, constant(sit), duration=3.0)
print("logged:", {k: v.shape for k, v in log.items()})
print(f"trunk height after 3 s: {log['qpos'][-1, 2]:.3f} m (SIT pose is at 0.07)")
show(render_frame(model, data), title="sitting: a stable base to work from")

# %% [markdown]
# ## Don't teleport: trajectories
#
# Turn the head 0.8 rad while seated. Naive way: write the new target once and
# let the servo slam. Better: interpolate over 0.6 s. `goto` uses a
# **minimum-jerk** profile — slow-fast-slow, like a human reach:

# %%
ts = np.linspace(0, 1, 100)
plt.plot(ts, ts, label="linear α(t)")
plt.plot(ts, 10 * ts**3 - 15 * ts**4 + 6 * ts**5, label="min-jerk α(t)")
plt.legend(); plt.xlabel("t / duration"); plt.title("interpolation profiles"); plt.show()

# %%
head_yaw = actuator_names(model).index("head_yaw")
turned = sit.copy()
turned[head_yaw] = 0.8

for label, fn in [("teleport", constant(turned)),
                  ("min-jerk 0.6 s", goto(sit, turned, 0.6))]:
    reset_to_keyframe(model, data, "SIT")
    log = run_control_loop(model, data, fn, duration=1.0)
    head_speed = np.abs(log["qvel"][:, 6 + head_yaw])
    plt.plot(log["t"], head_speed, label=label)
    print(f"{label:15s} peak head speed {head_speed.max():.1f} rad/s")
plt.xlabel("t [s]"); plt.ylabel("|head_yaw velocity| [rad/s]"); plt.legend(); plt.show()

# %% [markdown]
# Half the peak speed for the same move. On the real robot the teleport version
# means torque spikes, battery sag, stripped gears — and it looks terrible.
#
# ## Forward kinematics: where is the beak?
#
# Joint angles are *our* coordinates; the world cares about *positions*.
# Angles → positions is **forward kinematics (FK)**, and MuJoCo computes it
# every step: `data.site_xpos` has world positions of named sites
# (`left_foot`, `mouth_tip`, `imu`, ...). The log records them each tick, so we
# can draw the path of the beak during a nod:

# %%
site_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)]
mouth = site_names.index("mouth_tip")
neck = actuator_names(model).index("neck_pitch")

amp = np.zeros(model.nu)
amp[neck] = 0.4
reset_to_keyframe(model, data, "SIT")
log = run_control_loop(model, data, sine(sit, amp, frequency=0.5), duration=2.0)

path = log["site_xpos"][:, mouth]
plt.plot(path[:, 0], path[:, 2])
plt.xlabel("x [m]"); plt.ylabel("z [m]"); plt.axis("equal")
plt.title("mouth_tip path during a nod (side view)"); plt.show()

# %% [markdown]
# One joint (neck_pitch, 0.4 rad) sweeps the beak through ~22 cm of world space
# — that's leverage, and it's why FK matters: to *place* the beak you must
# reason through the whole chain.
#
# ## Rhythm
#
# Walking is rhythmic, so we start oscillating now. `sine` turns a base pose +
# per-joint amplitudes into a ctrl function — a seated head dance:

# %%
amp = np.zeros(model.nu)
amp[actuator_names(model).index("head_yaw")] = 0.6
amp[neck] = 0.3

reset_to_keyframe(model, data, "SIT")
frames, log = record(model, data, sine(sit, amp, frequency=0.8), duration=6.0)
show_video(frames)
print(f"still seated after 6 s: trunk z = {log['qpos'][-1, 2]:.3f} m")

# %% [markdown]
# ## Routines — and an honest failure
#
# `chain` glues segments into a routine (each segment gets a local clock).
# Watch closely what happens at the end, when the duck tries to *stand up*:

# %%
routine = chain(
    (1.2, goto(stand, sit, 1.2)),      # sit down (easy — gravity helps)
    (3.0, sine(sit, amp, 0.8)),        # dance
    (1.5, goto(sit, stand, 1.5)),      # try to stand up...
    (1.5, constant(stand)),
)
reset_to_keyframe(model, data, "STAND")
frames, log = record(model, data, routine, duration=7.2)
show_video(frames)
plt.plot(log["t"], log["qpos"][:, 2])
plt.xlabel("t [s]"); plt.ylabel("trunk z [m]"); plt.title("the routine"); plt.show()

# %% [markdown]
# Sitting down works; **standing up open-loop does not** — the duck pushes,
# tips, and face-plants, no matter how smooth the trajectory. Getting up needs
# *reacting to how the body is actually moving*:
#
# - Lab 3: you build that reaction by hand (feedback control) — enough to stand.
# - Lab 5: you load `BEST_alpha_sitstand.onnx`, the neural network Pollen
#   trained for this exact move, and watch it nail what we just failed.
#
# ## Recap
#
# - model variants are engineering choices (collision geometry ⇄ speed)
# - a pose = a ctrl vector; 50 Hz control on 500 Hz physics, via `run_control_loop`
# - never teleport: `goto` (min-jerk) halves peak speeds; sequence with `chain`
# - FK = angles → world positions, logged as `site_xpos`
# - open-loop gets you poses, rhythm, choreography — but not balance
#
# **Now: `lab.ipynb`, tasks T1–T5.**
