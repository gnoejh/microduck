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
# # Lab 2 — Kinematics & Joint Control · SOLUTION

# %% [markdown]
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/microduck/blob/main/labs/lab02_kinematics/solution.ipynb)
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
from microduck_sim import colab; colab.setup("labs/lab02_kinematics")

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
# Load `"walk"` and `"full"`. For each: how many geoms *collide*
# (`model.geom_contype != 0`)? Hold SIT for 2 s in each and plot trunk height.
# Explain what you see.

# %%
for variant in ["walk", "full"]:
    m = load_model(variant)
    d = mujoco.MjData(m)
    reset_to_keyframe(m, d, "SIT")
    log = run_control_loop(m, d, constant(keyframe_ctrl(m, "SIT")), 2.0)
    plt.plot(log["t"], log["qpos"][:, 2], label=variant)
    print(f"{variant:5s}: colliding geoms = {(m.geom_contype != 0).sum()}")
plt.xlabel("t [s]"); plt.ylabel("trunk z [m]"); plt.legend(); plt.title("SIT held, per variant"); plt.show()

# In "walk" only the feet collide; a seated duck rests on its trunk, which has
# no collision shape there -> it sinks through the floor. "full" collides -> stable.

# %% [markdown]
# ## T2 — Smoothness quantified
#
# Implement `goto_linear(start, target, duration)` returning a ctrl function
# that interpolates *linearly* (use `lerp`; clip after the end).
# Seated, turn the head (`head_yaw` to 0.8 rad) three ways — teleport,
# `goto_linear(0.6 s)`, library `goto(0.6 s)` — and measure the peak head
# speed `|qvel|` for each. Rank them.

# %%
def goto_linear(start, target, duration):
    start = np.asarray(start, dtype=float).copy()
    target = np.asarray(target, dtype=float).copy()

    def fn(t):
        return lerp(start, target, t / duration)

    return fn


head_yaw = acts.index("head_yaw")
turned = sit.copy()
turned[head_yaw] = 0.8

peaks = {}
for label, fn in [
    ("teleport", constant(turned)),
    ("linear", goto_linear(sit, turned, 0.6)),
    ("min-jerk", goto(sit, turned, 0.6)),
]:
    reset_to_keyframe(model, data, "SIT")
    log = run_control_loop(model, data, fn, 1.0)
    speed = np.abs(log["qvel"][:, 6 + head_yaw])
    peaks[label] = speed.max()
    plt.plot(log["t"], speed, label=label)
plt.xlabel("t [s]"); plt.ylabel("|head speed| [rad/s]"); plt.legend(); plt.show()
print({k: round(v, 2) for k, v in peaks.items()})

assert peaks["teleport"] > peaks["linear"] and peaks["teleport"] > peaks["min-jerk"]
# Teleport is by far the worst. Linear vs min-jerk peak speeds are close
# (min-jerk trades a slightly higher mid-move speed for zero start/stop jerk —
# look at the slopes at t=0 and t=0.6).

# %% [markdown]
# ## T3 — FK: the beak draws
#
# Seated, nod with `neck_pitch` (amplitude 0.4 rad, 0.5 Hz) for 2 s. From the
# log's `site_xpos`, plot the `mouth_tip` path in the x–z plane and compute
# (a) its x-extent and (b) the total path length.

# %%
site_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)]
mouth = site_names.index("mouth_tip")

amp = np.zeros(model.nu)
amp[acts.index("neck_pitch")] = 0.4
reset_to_keyframe(model, data, "SIT")
log = run_control_loop(model, data, sine(sit, amp, 0.5), 2.0)

path = log["site_xpos"][:, mouth]
plt.plot(path[:, 0], path[:, 2]); plt.axis("equal")
plt.xlabel("x [m]"); plt.ylabel("z [m]"); plt.title("mouth_tip, side view"); plt.show()

x_extent = np.ptp(path[:, 0])
arc_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
print(f"x extent {x_extent:.3f} m, path length {arc_length:.3f} m")
assert x_extent > 0.15  # one 0.4 rad joint sweeps the beak >15 cm — leverage!

# %% [markdown]
# ## T4 — Choreograph a seated dance
#
# Design your own dance: at least two joints, at least 6 s, and the duck must
# stay seated the whole time (trunk z > 0.05 m). Render it.

# %%
# Design detail: each sine segment lasts a WHOLE number of periods, so it ends
# back at the center pose and the next segment starts without a jump.
nod = np.eye(model.nu)[acts.index("neck_pitch")]
yaw = np.eye(model.nu)[acts.index("head_yaw")]
roll = np.eye(model.nu)[acts.index("head_roll")]
groove = chain(
    (2.0, sine(sit, nod * 0.4, 1.0)),          # nod, 2 periods
    (2.0, sine(sit, yaw * 0.6, 1.0)),          # shake, 2 periods
    (2.5, sine(sit, nod * 0.3 + roll * 0.4, 0.8)),  # sway, 2 periods
)
reset_to_keyframe(model, data, "SIT")
frames, log = record(model, data, groove, duration=6.5)
show_video(frames)

assert np.all(log["qpos"][:, 2] > 0.05), "fell out of the seat!"
print("still seated — dance valid")

# %% [markdown]
# ## T5 — Challenge: the longest stand
#
# Write `stand_time(ctrl_fn)`: reset to STAND, run for up to 10 s, return the
# first time trunk z drops below 0.10 m (or 10.0 if it never does). Baseline:
# holding STAND survives ~0.8 s. Beat it, open loop — try at least three ideas
# and report your best. *Hint: where is the mass, and how far can it wander
# before the feet can't catch it?*

# %%
def stand_time(ctrl_fn, tmax=10.0):
    reset_to_keyframe(model, data, "STAND")
    log = run_control_loop(model, data, ctrl_fn, tmax)
    below = log["qpos"][:, 2] < 0.10
    return float(log["t"][np.argmax(below)]) if below.any() else tmax


baseline = stand_time(constant(stand))
print(f"baseline: {baseline:.2f} s")

# idea 1: wider stance (hip roll) — barely helps
wide = stand.copy()
wide[acts.index("left_hip_roll")] -= 0.08
wide[acts.index("right_hip_roll")] += 0.08
print(f"wide stance: {stand_time(goto(stand, wide, 0.5)):.2f} s")

# idea 2: slight crouch — lowers the center of mass, clearly better
crouch = stand.copy()
for joint, delta in [("left_hip_pitch", -0.10), ("left_knee", -0.10), ("left_ankle", 0.10),
                     ("right_hip_pitch", 0.10), ("right_knee", 0.10), ("right_ankle", -0.10)]:
    crouch[acts.index(joint)] += delta
best = stand_time(goto(stand, crouch, 0.5))
print(f"crouch: {best:.2f} s")

# idea 3: ankle lean — tips the balance point; direction matters
lean = stand.copy()
lean[acts.index("left_ankle")] += 0.03
lean[acts.index("right_ankle")] -= 0.03
print(f"ankle lean: {stand_time(goto(stand, lean, 0.5)):.2f} s")

assert best > baseline
# Typical numbers: baseline 0.82 s, crouch ~1.2 s. Nobody reaches 10 s — that
# requires feedback, which is exactly Lab 3.
