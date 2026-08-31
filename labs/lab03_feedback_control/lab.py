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
# # Lab 3 — Feedback Control · WORKSHEET
#
# Today you write the ~15 lines that keep the duck standing. T1–T5.

# %%
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from microduck_sim.model import actuator_names, load_model, reset_to_keyframe
from microduck_sim.control import (
    constant, keyframe_ctrl, make_push, quat_to_pitch_roll, run_control_loop,
)
from microduck_sim.viz import record, show_video

model = load_model("full")
data = mujoco.MjData(model)
acts = actuator_names(model)
stand = keyframe_ctrl(model, "STAND")
sit = keyframe_ctrl(model, "SIT")

# %% [markdown]
# ## T1 — Gravity droop
#
# Seated, command `head_pitch` to −0.3 rad and let it settle for 3 s. Measure
# the settled angle (mean of the last 25 log rows) for kp ∈ {0.2, 0.55, 2.0,
# 8.0}. To change a servo gain: `model.actuator_gainprm[i, 0] = kp` **and**
# `model.actuator_biasprm[i, 1] = -kp`. Restore 0.55 afterwards!
# Plot |steady-state error| vs kp (log-log).
#
# **Question:** the error shrinks with kp but Microduck ships kp = 0.55 — name
# two costs of a stiff servo.

# %%
head = acts.index("head_pitch")
qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_pitch")]

kps = [0.2, 0.55, 2.0, 8.0]
errors = []
# TODO: fill `errors` (one |settled - (-0.3)| per kp), then plot.

# %%
assert len(errors) == 4 and all(a > b for a, b in zip(errors, errors[1:]))
print("T1 ok:", {k: round(e, 3) for k, e in zip(kps, errors)})

# %% [markdown]
# *Your answer:* ...

# %% [markdown]
# ## T2 — Trunk pitch from the quaternion
#
# Implement `my_pitch(quat)` for a wxyz quaternion:
# pitch = asin(2(w·y − z·x)), with the argument clipped to [−1, 1].
# The check compares you against the library on 200 random quaternions.

# %%
def my_pitch(quat):
    # TODO
    ...


# %%
rng = np.random.default_rng(0)
for _ in range(200):
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    assert abs(my_pitch(q) - quat_to_pitch_roll(q)[0]) < 1e-9
print("T2 ok")

# %% [markdown]
# ## T3 — Build the balance controller
#
# The ankle strategy, from the lecture:
# ```
# correction = kp * pitch + kd * pitch_rate      # rate: data.qvel[4]
# ctrl[left_ankle] += correction ; ctrl[right_ankle] -= correction
# ```
# Write `balance(kp, kd)` → ctrl function, and `stand_time(fn, on_tick=None)`
# → seconds until trunk z < 0.10 m (cap 10 s; reset to STAND first).
# Sweep (kp, kd) ∈ {(0,0), (0.5,0.05), (1,0.1), (2,0.2)} and print the times.

# %%
la, ra = acts.index("left_ankle"), acts.index("right_ankle")


def balance(kp, kd):
    # TODO
    ...


def stand_time(fn, tmax=10.0, on_tick=None):
    # TODO
    ...


# TODO: sweep and print

# %%
assert stand_time(balance(1.0, 0.1)) >= 10.0
print("T3 ok — the duck stands. Watch it with record()/show_video if you like.")

# %% [markdown]
# ## T4 — Sabotage
#
# Run `balance(-1.0, -0.1)` and compare with `balance(0, 0)`.
# Explain the result in one sentence using the words *sign* and *amplify*.

# %%
# TODO: measure both times
t_wrong = ...
t_none = ...

# %%
assert t_wrong < t_none
print(f"T4 ok: wrong sign {t_wrong:.2f} s < no feedback {t_none:.2f} s")

# %% [markdown]
# *Your explanation:* ...

# %% [markdown]
# ## T5 — Push limits 🏆
#
# (a) With `balance(1.0, 0.1)`, find the **maximum forward push** (0.1 s long,
# `make_push(model, data, force=..., axis=0)` as `on_tick`; zero
# `data.xfrc_applied` after each run) the duck survives — 0.1 N resolution.
#
# (b) Repeat sideways (`axis=1`). It's much higher. Look at the duck's feet
# from the front (`render_frame`) and explain why in one sentence.
#
# (c) **Challenge (leaderboard):** modify the controller — different gains, an
# extra hip-pitch term, anything open-loop-free — and maximize the forward
# push you survive. Record your best in `best_push`.

# %%
def survives(fn, force, axis):
    # TODO (build push, run stand_time with on_tick, clean xfrc_applied)
    ...


# TODO: (a) max forward, (b) max sideways, (c) your best controller
max_forward = ...
max_sideways = ...
best_push = ...

# %%
assert max_forward >= 1.0 and max_sideways > max_forward
assert best_push >= max_forward
print(f"T5 ok: fwd {max_forward:.1f} N, side {max_sideways:.1f} N, your best {best_push:.1f} N")

# %% [markdown]
# *Why is sideways so much stronger?* ...
#
# ---
# Next week the ground-truth pitch is taken away: you get a noisy IMU and must
# *estimate* it — and your balance controller comes along for the ride.
