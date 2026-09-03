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
# # Lab 3 — Feedback Control ⚖️
#
# **Microduck Simulation Course · session 3 of 8**
#
# Two weeks of falling ducks. Today, ~15 lines of code fix it:
# 1. The servo you've been using *is* a controller
# 2. Why that's not enough for balance
# 3. Closing the loop: the ankle strategy
# 4. When feedback goes wrong
# 5. Getting pushed around (and surviving)

# %% [markdown]
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/microduck/blob/main/labs/lab03_feedback_control/lecture.ipynb)
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
from microduck_sim import colab; colab.setup("labs/lab03_feedback_control")

# %%
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from microduck_sim.model import actuator_names, load_model, reset_to_keyframe
from microduck_sim.control import (
    constant, keyframe_ctrl, make_balance_controller, make_push,
    quat_to_pitch_roll, run_control_loop,
)
from microduck_sim.viz import record, render_frame, show, show_video

model = load_model("full")
data = mujoco.MjData(model)
acts = actuator_names(model)
stand = keyframe_ctrl(model, "STAND")

# %% [markdown]
# ## The servo is already a controller
#
# Each joint runs a **proportional (P) controller** inside MuJoCo: torque =
# kp · (target − angle). The gain is model data — and it's small:

# %%
print("servo kp:", model.actuator_gainprm[0, 0])  # 0.55 Nm/rad, like the real XL330

# %% [markdown]
# What does kp do? Command the head to −0.3 rad (seated, so nothing falls) and
# watch the *measured* angle for different gains:

# %%
sit = keyframe_ctrl(model, "SIT")
head = acts.index("head_pitch")
jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_pitch")
qadr = model.jnt_qposadr[jid]
target = sit.copy(); target[head] = -0.3

for kp in [0.2, 0.55, 2.0, 8.0]:
    model.actuator_gainprm[head, 0] = kp
    model.actuator_biasprm[head, 1] = -kp
    reset_to_keyframe(model, data, "SIT")
    log = run_control_loop(model, data, constant(target), 3.0)
    plt.plot(log["t"], log["qpos"][:, qadr], label=f"kp={kp}")
    print(f"kp={kp:4.2f}: settles at {log['qpos'][-25:, qadr].mean():+.3f} (target −0.3)")
model.actuator_gainprm[head, 0] = 0.55  # restore the realistic value
model.actuator_biasprm[head, 1] = -0.55
plt.axhline(-0.3, ls="--", c="gray"); plt.xlabel("t [s]"); plt.ylabel("head_pitch [rad]")
plt.legend(); plt.title("P control: steady-state error under gravity"); plt.show()

# %% [markdown]
# Low kp = soft servo = the head **sags under gravity** and never reaches the
# target (steady-state error, the classic P-controller flaw). High kp tracks
# well — so why not crank it up? Stiff joints are dangerous, power-hungry, and
# on the real robot they strip gears. Microduck runs kp = 0.55. Balance must
# come from somewhere else.
#
# ## Why servo P can't balance the duck
#
# The trunk has **no motor**. Its orientation is *underactuated* — we can only
# influence it indirectly, by pushing against the ground through the legs.
# No joint-level gain can fix that, because no joint target says "trunk
# upright". We need an outer loop:
#
# > measure trunk pitch → compute a correction → shift the **ankle** targets
#
# This *cascade* (a controller commanding another controller) is how balance
# works from segways to humanoids. Step 1: measure trunk pitch. The free
# joint's quaternion is in `qpos[3:7]`:

# %%
reset_to_keyframe(model, data, "STAND")
print("quaternion:", data.qpos[3:7].round(3))
print("pitch, roll:", np.round(quat_to_pitch_roll(data.qpos[3:7]), 3))
# (today we read the simulator's ground truth — next week you earn this number
# from an IMU, like the real robot has to)

# %% [markdown]
# ## The ankle strategy
#
# When you stand in a bus that brakes, your ankles fire first. Same here —
# a PD law on trunk pitch, applied to the ankle targets (mirrored):
#
# ```python
# correction = kp * pitch + kd * pitch_rate
# ctrl[left_ankle]  += correction
# ctrl[right_ankle] -= correction
# ```
#
# Gain sweep — stand time out of 10 s:

# %%
la, ra = acts.index("left_ankle"), acts.index("right_ankle")

def stand_time(fn, tmax=10.0, on_tick=None):
    reset_to_keyframe(model, data, "STAND")
    log = run_control_loop(model, data, fn, tmax, on_tick=on_tick)
    below = log["qpos"][:, 2] < 0.10
    return float(log["t"][np.argmax(below)]) if below.any() else tmax

for kp, kd in [(0.0, 0.0), (0.5, 0.05), (1.0, 0.1), (2.0, 0.2)]:
    fn = make_balance_controller(model, data, stand, kp_pitch=kp, kd_pitch=kd)
    print(f"kp={kp:3.1f} kd={kd:4.2f}: stands {stand_time(fn):5.2f} s")

# %% [markdown]
# From 0.8 s to *indefinitely*, with two numbers. Watch it — and note the
# micro-corrections in the ankles (this is what your calves do all day):

# %%
reset_to_keyframe(model, data, "STAND")
fn = make_balance_controller(model, data, stand)
frames, log = record(model, data, fn, duration=8.0)
show_video(frames)
plt.plot(log["t"], log["ctrl"][:, la], label="left ankle target")
plt.plot(log["t"], np.degrees(np.arcsin(np.clip(2*(log["qpos"][:,3]*log["qpos"][:,5]
        - log["qpos"][:,6]*log["qpos"][:,4]), -1, 1))), label="trunk pitch [deg]")
plt.xlabel("t [s]"); plt.legend(); plt.title("balance is constant tiny work"); plt.show()

# %% [markdown]
# ## When feedback goes wrong
#
# Flip the sign of the gains — now the controller *amplifies* every lean:

# %%
fn = make_balance_controller(model, data, stand, kp_pitch=-1.0, kd_pitch=-0.1)
print(f"wrong sign: falls at {stand_time(fn):.2f} s — FASTER than no feedback (0.82 s)")

# %% [markdown]
# > Feedback is not automatically good. Wrong sign, too much gain, or too much
# > delay all turn a stabilizer into an amplifier. Every real control engineer
# > has built this at least once, usually by accident.
#
# ## Getting pushed
#
# `make_push` applies a force to the trunk mid-run (Ctrl+right-drag in the
# viewer does this interactively). How much can the controller take?

# %%
for force in [0.5, 1.0, 1.5]:
    fn = make_balance_controller(model, data, stand)
    push = make_push(model, data, force=force, axis=0)
    t = stand_time(fn, on_tick=push)
    data.xfrc_applied[:] = 0
    print(f"forward push {force} N × 0.1 s: {'survives' if t >= 10 else f'falls at {t:.2f} s'}")

# %% [markdown]
# ~1 N is the limit — for an 800 g robot that's a solid shove. Try the same
# sideways (`axis=1`) in the worksheet: the duck survives *more* laterally,
# with no roll feedback at all. Look at its feet and figure out why.
#
# ## Where this is heading
#
# Shove the robot's *shipped* balance policy with your mouse (drag the duck).
# It survives pushes ten times what your ankle strategy takes — robustness it
# was *trained* for, which is Labs 5–8:

# %%
from microduck_sim.web import show_playground

show_playground("walk", policy="stand", height=480)

# %% [markdown]
# ## Recap
#
# - servos are P controllers; P alone leaves gravity droop, stiffness has costs
# - the trunk is underactuated → cascade: trunk PD sets ankle targets
# - kp=1.0, kd=0.1 turns 0.8 s into forever; flipped signs fall *faster*
# - feedback quality has a measurable currency: max survivable push
# - library: `make_balance_controller`, `make_push`, `quat_to_pitch_roll`
#
# **Worksheet time: T1–T5.** Next week: no more ground-truth pitch — you get
# a noisy IMU, like the real robot.
