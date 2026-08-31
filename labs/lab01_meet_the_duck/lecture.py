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
# # Lab 1 — Meet the Duck 🦆
#
# **Microduck Simulation Course · session 1 of 8**
#
# Today:
# 1. What this course is (and why there is no physical robot)
# 2. The Microduck robot
# 3. How a physics simulator thinks: *model* vs *state*
# 4. Loading, posing, and rendering the duck
# 5. Your first physics surprise

# %% [markdown]
# ## The course at a glance
#
# | # | Lab | You will |
# |---|-----|----------|
# | **1** | **Meet the duck** | **load, pose, render, step the sim** |
# | 2 | Kinematics & joint control | move joints along trajectories |
# | 3 | Feedback control | write a PD controller, survive pushes |
# | 4 | Sensors & estimation | fuse IMU data into orientation |
# | 5 | Running a trained policy | drive a neural-network walker |
# | 6 | First RL training | train a policy yourself (PPO) |
# | 7 | Reward shaping & walking | make it walk better |
# | 8 | Evaluation & mini-project | stress-test and compare policies |
#
# Everything runs on **your laptop, CPU only**. The second half of the semester
# moves to NVIDIA Isaac Sim on lab machines — the concepts you learn here carry over.

# %% [markdown]
# ## Why simulation?
#
# The [Microduck](https://pollen-robotics.com/microduck/) is a real, buyable robot —
# but we work **entirely in simulation**, like most robot-learning research does:
#
# - a fall in simulation costs nothing and resets in milliseconds,
# - we can run the robot **faster than real time**, thousands of times in parallel,
# - we can measure *everything* (true velocities, contact forces, ...),
# - and every one of you gets a robot — no queue for lab hardware.
#
# Pollen Robotics trains all of Microduck's real behaviors in this exact simulator
# (MuJoCo) and then copies the learned network onto the robot. We use their actual
# robot model files and their actual trained policies.
#
# **Warm-up (2 min):** open the official browser playground and drive the duck with
# WASD/arrow keys: <https://huggingface.co/spaces/pollen-robotics/microduck-simulator>
# — everything you see there, you will build up yourself in Python over these labs.

# %% [markdown]
# ## The robot
#
# (photos: https://pollen-robotics.com/microduck/)
#
# - 25 cm tall, 800 g
# - **15 servo motors** (Dynamixel XL330): 2×5 in the legs, 4 in neck/head, 1 beak
# - two IMUs (trunk + head), camera, depth sensor
# - its "brain" (a neural network) decides new motor targets **50 times per second**
#
# Our simulation model mirrors this. The *walk* variant we use today has the 14
# leg + neck/head motors (the beak only matters for grasping labs).

# %% [markdown]
# ## MuJoCo in one slide
#
# MuJoCo separates two things you must never confuse:
#
# - **`MjModel`** — the *description* of the robot: bodies, joints, motors, masses,
#   geometry. Loaded from an XML file (called **MJCF**). Never changes while simulating.
# - **`MjData`** — the *current state*: positions `qpos`, velocities `qvel`, motor
#   commands `ctrl`, sensor readings, contacts. Changes every step.
#
# The simulation loop is just: write `ctrl` → call `mj_step` → read the new state.

# %%
import mujoco
import numpy as np

from microduck_sim.model import load_model, joint_names, actuator_names

model = load_model("walk")
print(f"bodies: {model.nbody}   joints: {model.njnt}   actuators: {model.nu}")
print(f"qpos size: {model.nq}   physics timestep: {model.opt.timestep} s")

# %% [markdown]
# 21 position coordinates for 15 joints? The first joint is special: a **free
# joint** connecting the trunk to the world — 3 numbers for position + 4 for
# orientation (a quaternion) = 7. The other 14 joints are hinges, one angle each.

# %%
for name in joint_names(model):
    print(name)

# %% [markdown]
# ## Posing the duck
#
# The model ships **keyframes** — named snapshots of `qpos` + `ctrl`:
# `INIT`, `STAND`, `SIT`, `FOLD`. Let's look at all four.

# %%
from microduck_sim.model import KEYFRAME_NAMES, reset_to_keyframe
from microduck_sim.viz import render_frame, show_grid

data = mujoco.MjData(model)

images = []
for name in KEYFRAME_NAMES:
    reset_to_keyframe(model, data, name)
    images.append(render_frame(model, data))
show_grid(images, titles=list(KEYFRAME_NAMES))

# %% [markdown]
# ## What the duck sees
#
# The model also carries the robot's sensors — including its head camera.
# Rendering from it = the robot's point of view:

# %%
from microduck_sim.viz import show

reset_to_keyframe(model, data, "STAND")
show(render_frame(model, data, camera="head_camera"), title="head camera")

# %% [markdown]
# ## Stepping physics — your first surprise
#
# `STAND` was carefully designed so the center of mass sits over the feet.
# The motors are position-controlled and we keep commanding the STAND pose.
# Surely the duck just... stands?

# %%
import matplotlib.pyplot as plt

reset_to_keyframe(model, data, "STAND")
heights, times = [], []
for _ in range(int(2.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)
    times.append(data.time)
    heights.append(data.qpos[2])  # trunk z

plt.plot(times, heights)
plt.xlabel("time [s]"); plt.ylabel("trunk height [m]"); plt.title("Open-loop STAND")
plt.axhline(0.12, ls="--", c="gray"); plt.show()

show(render_frame(model, data), title="after 2 seconds...")

# %% [markdown]
# **It falls after about a second.** Nothing is broken — this is the point:
#
# > A statically balanced pose is **not** stable. Tiny asymmetries grow, and
# > nothing *reacts*. Real standing is active: you constantly correct.
#
# In Lab 3 you will write that correction yourself (a feedback controller).
# In Labs 5–7, a neural network will learn it.
#
# (Detail: in this training-oriented model only the feet have collision shapes,
# so a fallen duck's body sinks through the floor. Harmless — and it makes
# training faster.)

# %% [markdown]
# ## The interactive viewer
#
# On your own machine (not on the projector 😉) run, from the repo root:
#
# ```
# uv run scripts/view.py
# ```
#
# - drag = orbit, scroll = zoom, **SPACE** = cycle poses, **R** = reset
# - double-click a body, then **Ctrl + right-drag** applies a force — push the duck!
#
# ## Recap
#
# - `MjModel` = description (from MJCF XML), `MjData` = state
# - `qpos` = free joint (7) + one angle per hinge; `ctrl` = motor targets
# - keyframes store named poses; `mj_step` advances physics by 2 ms
# - open-loop standing falls → we need feedback (Lab 3) or learning (Labs 5+)
#
# **Now: open `lab.ipynb` and work through tasks T1–T5.**
