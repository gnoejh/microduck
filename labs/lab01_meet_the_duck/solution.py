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
# # Lab 1 — Meet the Duck · SOLUTION
#
# Instructor reference. Same tasks as `lab.ipynb`, completed.

# %%
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from microduck_sim.model import (
    KEYFRAME_NAMES,
    actuator_names,
    joint_names,
    load_model,
    reset_to_keyframe,
)
from microduck_sim.viz import render_frame, show, show_grid

# %% [markdown]
# ## T1 — Load and inspect (warm-up)
#
# Load the `"walk"` scene and answer *in code*:
# how many bodies, joints, actuators? What is the physics timestep?
# Print every joint name next to its index.

# %%
model = load_model("walk")
data = mujoco.MjData(model)

print(f"nbody={model.nbody}  njnt={model.njnt}  nu={model.nu}  dt={model.opt.timestep}")
for i, name in enumerate(joint_names(model)):
    print(f"{i:2d}  {name}")

# CHECK: expected 16 bodies, 15 joints, 14 actuators, dt=0.002
assert (model.nbody, model.njnt, model.nu) == (16, 15, 14)

# %% [markdown]
# ## T2 — Keyframe gallery
#
# Render the duck in all four keyframes in one image grid. Then look at
# `data.qpos[2]` (trunk height) in each pose: which pose is lowest, and why?

# %%
images, heights = [], {}
for name in KEYFRAME_NAMES:
    reset_to_keyframe(model, data, name)
    images.append(render_frame(model, data))
    heights[name] = round(float(data.qpos[2]), 3)
show_grid(images, titles=[f"{n}  z={heights[n]}" for n in KEYFRAME_NAMES])
print(heights)
# SIT and FOLD are lowest (0.07 m): legs folded under the trunk.

# %% [markdown]
# ## T3 — The fall of open-loop STAND
#
# Reset to STAND and step physics for 2 simulated seconds while recording time
# and trunk height. Plot height vs time. Read off: **when** does the duck start
# to fall (height drops below 0.10 m)?

# %%
reset_to_keyframe(model, data, "STAND")
times, heights_t = [], []
for _ in range(int(2.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)
    times.append(data.time)
    heights_t.append(data.qpos[2])
times, heights_t = np.asarray(times), np.asarray(heights_t)

plt.plot(times, heights_t)
plt.axhline(0.10, ls="--", c="red")
plt.xlabel("time [s]"); plt.ylabel("trunk height [m]"); plt.show()

fall_time = times[np.argmax(heights_t < 0.10)]
print(f"falls below 0.10 m at t = {fall_time:.2f} s")
assert 0.5 < fall_time < 1.5

# %% [markdown]
# ## T4 — Your first motor command
#
# The duck should look up. While holding the STAND pose, command the
# `head_pitch` actuator to −0.5 rad, simulate 0.3 s, and render the result.
#
# *Hint:* `data.ctrl` is indexed by actuator id — find it with
# `actuator_names(model).index("head_pitch")`.

# %%
reset_to_keyframe(model, data, "STAND")
head = actuator_names(model).index("head_pitch")
data.ctrl[head] = -0.5
for _ in range(int(0.3 / model.opt.timestep)):
    mujoco.mj_step(model, data)
show(render_frame(model, data), title="looking up")

head_angle = data.qpos[6 + joint_names(model).index("head_pitch")]
print(f"head_pitch angle: {head_angle:.2f} rad (commanded -0.5)")
assert head_angle < -0.2

# %% [markdown]
# Note the angle doesn't reach −0.5 exactly: position servos are springs, not
# teleporters — gravity and inertia pull back. Lab 3 digs into exactly this.

# %% [markdown]
# ## T5 — Challenge: does a heavier head fall faster?
#
# Hypothesis: raising the head (T4) shifts mass backwards, so the open-loop
# stand should tip over *sooner*. Test it: repeat T3 but with head_pitch
# commanded to −0.5 from the start, and compare fall times.

# %%
def fall_time_with_head(head_ctrl: float) -> float:
    reset_to_keyframe(model, data, "STAND")
    data.ctrl[head] = head_ctrl
    while data.time < 3.0:
        mujoco.mj_step(model, data)
        if data.qpos[2] < 0.10:
            return data.time
    return float("inf")


for c in [0.35, 0.0, -0.5]:  # STAND default, level, up
    print(f"head_ctrl={c:+.2f}  falls at {fall_time_with_head(c):.2f} s")

# Typical result: moving the head away from the tuned STAND value changes the
# balance and the fall time measurably — mass distribution matters even for a
# 'static' pose.
