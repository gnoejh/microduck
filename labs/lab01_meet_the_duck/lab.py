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
# # Lab 1 — Meet the Duck · WORKSHEET
#
# Work through T1–T5. Cells marked `# TODO` are yours; `assert` cells check your
# work — if they pass silently, you're good. Ask when stuck, guess before asking. 🦆

# %% [markdown]
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/microduck/blob/main/labs/lab01_meet_the_duck/lab.ipynb)
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
from microduck_sim import colab; colab.setup("labs/lab01_meet_the_duck")

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
# Load the `"walk"` scene and answer *in code*: how many bodies, joints,
# actuators? What is the physics timestep? Print every joint name with its index.

# %%
model = load_model("walk")
data = mujoco.MjData(model)

# TODO: print nbody, njnt, nu and the timestep; then loop over joint_names(model)
# and print "index  name" per line.

# %%
assert (model.nbody, model.njnt, model.nu) == (16, 15, 14)
print("T1 ok")

# %% [markdown]
# ## T2 — Keyframe gallery
#
# Render the duck in all four keyframes (`KEYFRAME_NAMES`) in one image grid
# using `show_grid`. Record the trunk height `data.qpos[2]` in each pose.
#
# **Question (answer in the markdown cell below):** which poses are lowest, and why?

# %%
images, heights = [], {}
# TODO: for each keyframe name: reset_to_keyframe(...), append render_frame(...),
# store round(float(data.qpos[2]), 3) in heights[name].
# Then: show_grid(images, titles=...)

# %%
assert set(heights) == set(KEYFRAME_NAMES) and len(images) == 4
print("T2 ok:", heights)

# %% [markdown]
# *Your answer:* ...

# %% [markdown]
# ## T3 — The fall of open-loop STAND
#
# Reset to STAND and step physics for 2 simulated seconds (`mj_step` in a loop),
# recording `data.time` and trunk height each step. Plot height vs time.
# Compute `fall_time`: the first time the height drops below 0.10 m.

# %%
reset_to_keyframe(model, data, "STAND")
times, heights_t = [], []
# TODO: step for 2.0 simulated seconds, recording times and heights;
# convert to np arrays, plot, and compute fall_time.
fall_time = ...

# %%
assert 0.5 < fall_time < 1.5, f"got {fall_time}"
print(f"T3 ok: falls at {fall_time:.2f} s")

# %% [markdown]
# ## T4 — Your first motor command
#
# Make the duck look up: holding STAND, command the `head_pitch` actuator to
# −0.5 rad, simulate 0.3 s, render. Then read the *actual* joint angle from
# `data.qpos` and compare with what you commanded.
#
# *Hints:* `data.ctrl` is indexed by actuator id —
# `actuator_names(model).index("head_pitch")`. In `qpos`, hinge angles start at
# index 7 (after the free joint), in `joint_names` order (which includes the
# free joint itself — mind the offset!).

# %%
reset_to_keyframe(model, data, "STAND")
# TODO: set data.ctrl for head_pitch, step 0.3 s, show a render,
# then store the measured angle in head_angle.
head_angle = ...

# %%
assert head_angle < -0.2
print(f"T4 ok: commanded -0.5, measured {head_angle:.2f} — why the difference?")

# %% [markdown]
# *Your answer (why doesn't it reach −0.5 exactly?):* ...

# %% [markdown]
# ## T5 — Challenge: does head position change the fall time?
#
# Hypothesis: the head is heavy enough that where it points changes when the
# open-loop stand tips over. Write `fall_time_with_head(head_ctrl)` that resets
# to STAND, holds the given head_pitch command, and returns the time the trunk
# drops below 0.10 m (simulate max 3 s). Compare head_ctrl = +0.35 (STAND
# default), 0.0, and −0.5. What do you observe?

# %%
def fall_time_with_head(head_ctrl: float) -> float:
    # TODO
    ...


for c in [0.35, 0.0, -0.5]:
    print(f"head_ctrl={c:+.2f}  falls at {fall_time_with_head(c):.2f} s")

# %% [markdown]
# *Your observation:* ...
#
# ---
# **Done!** Next week: making the duck *move* — smoothly, on purpose.
