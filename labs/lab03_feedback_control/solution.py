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
# # Lab 3 — Feedback Control · SOLUTION

# %% [markdown]
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/microduck/blob/main/labs/lab03_feedback_control/solution.ipynb)
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
# Seated, command `head_pitch` to −0.3 rad and let it settle 3 s. Measure the
# settled angle (mean of the last 0.5 s) for kp ∈ {0.2, 0.55, 2.0, 8.0}
# (set `model.actuator_gainprm[i, 0] = kp` AND `model.actuator_biasprm[i, 1] = -kp`;
# restore 0.55 afterwards). Plot |error| vs kp.

# %%
head = acts.index("head_pitch")
qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_pitch")]
target = sit.copy(); target[head] = -0.3

kps = [0.2, 0.55, 2.0, 8.0]
errors = []
for kp in kps:
    model.actuator_gainprm[head, 0] = kp
    model.actuator_biasprm[head, 1] = -kp
    reset_to_keyframe(model, data, "SIT")
    log = run_control_loop(model, data, constant(target), 3.0)
    errors.append(abs(log["qpos"][-25:, qadr].mean() - (-0.3)))
model.actuator_gainprm[head, 0] = 0.55
model.actuator_biasprm[head, 1] = -0.55

plt.loglog(kps, errors, "o-"); plt.xlabel("kp"); plt.ylabel("|steady-state error| [rad]")
plt.title("P controller vs gravity"); plt.show()
print({k: round(e, 3) for k, e in zip(kps, errors)})
assert all(a > b for a, b in zip(errors, errors[1:])), "error must shrink with kp"

# %% [markdown]
# ## T2 — Trunk pitch from the quaternion
#
# Implement `my_pitch(quat)` (wxyz order): pitch = asin(2(w·y − z·x)), clipped.
# Check it against `quat_to_pitch_roll` on 200 random unit quaternions.

# %%
def my_pitch(quat):
    w, x, y, z = quat
    return float(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))


rng = np.random.default_rng(0)
for _ in range(200):
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    assert abs(my_pitch(q) - quat_to_pitch_roll(q)[0]) < 1e-12
print("T2 ok")

# %% [markdown]
# ## T3 — Build the balance controller
#
# Write `balance(kp, kd)` returning a ctrl function implementing the ankle
# strategy, and `stand_time(fn)` (10 s cap, trunk z < 0.10 = fallen).
# Sweep (kp, kd) ∈ {(0,0), (0.5,0.05), (1,0.1), (2,0.2)}; report stand times.

# %%
la, ra = acts.index("left_ankle"), acts.index("right_ankle")


def balance(kp, kd):
    def fn(t):
        pitch = my_pitch(data.qpos[3:7])
        rate = data.qvel[4]
        c = stand.copy()
        corr = kp * pitch + kd * rate
        c[la] += corr
        c[ra] -= corr
        return c
    return fn


def stand_time(fn, tmax=10.0, on_tick=None):
    reset_to_keyframe(model, data, "STAND")
    log = run_control_loop(model, data, fn, tmax, on_tick=on_tick)
    below = log["qpos"][:, 2] < 0.10
    return float(log["t"][np.argmax(below)]) if below.any() else tmax


times = {}
for kp, kd in [(0.0, 0.0), (0.5, 0.05), (1.0, 0.1), (2.0, 0.2)]:
    times[(kp, kd)] = stand_time(balance(kp, kd))
    print(f"kp={kp:3.1f} kd={kd:4.2f}: {times[(kp, kd)]:5.2f} s")
assert max(times.values()) >= 10.0
print("T3 ok — indefinite standing achieved")

# %% [markdown]
# ## T4 — Sabotage
#
# Run your controller with (kp, kd) = (−1.0, −0.1). Compare to the
# no-feedback baseline and explain what you see.

# %%
t_wrong = stand_time(balance(-1.0, -0.1))
t_none = stand_time(balance(0.0, 0.0))
print(f"wrong sign: {t_wrong:.2f} s   no feedback: {t_none:.2f} s")
assert t_wrong < t_none
# Explanation: with flipped signs every lean produces an ankle push in the
# SAME direction as the lean — the loop amplifies its own error. Positive
# feedback: worse than doing nothing.

# %% [markdown]
# ## T5 — Push limits 🏆
#
# (a) With the working controller (1.0, 0.1), find the maximum forward push
# (0.1 s duration, 0.1 N resolution) it survives. (b) Same for sideways
# (`axis=1`) — explain the asymmetry. (c) Challenge: change the controller
# (gains, or add a hip term) to survive a bigger forward push.

# %%
def survives(fn_factory, force, axis):
    fn = fn_factory()
    push = make_push(model, data, force=force, axis=axis, t_start=2.0)
    t = stand_time(fn, on_tick=push)
    data.xfrc_applied[:] = 0
    return t >= 10.0


def max_push(fn_factory, axis):
    f = 0.0
    while f < 5.0 and survives(fn_factory, f + 0.1, axis):
        f += 0.1
    return f


fwd = max_push(lambda: balance(1.0, 0.1), axis=0)
lat = max_push(lambda: balance(1.0, 0.1), axis=1)
print(f"max forward push: {fwd:.1f} N   max sideways push: {lat:.1f} N")
assert fwd >= 1.0
# Asymmetry: the feet are side by side — the support polygon is wide in y and
# short in x. Sideways leans are resisted by geometry alone; forward/backward
# leans rely entirely on the controller.

# %%
# (c) hip strategy: the hips join the correction, mirrored like the ankles.
# Sign matters (again!): the helpful direction is the one where hips and
# ankles pitch the body the same way. Measured: ankle-only 1.1 N → 1.4 N.
lhp, rhp = acts.index("left_hip_pitch"), acts.index("right_hip_pitch")


def balance_hip(kp, kd, hip_gain=0.4):
    def fn(t):
        pitch = my_pitch(data.qpos[3:7])
        rate = data.qvel[4]
        corr = kp * pitch + kd * rate
        c = stand.copy()
        c[la] += corr; c[ra] -= corr
        c[lhp] += hip_gain * corr; c[rhp] -= hip_gain * corr
        return c
    return fn


fwd_hip = max_push(lambda: balance_hip(1.0, 0.1), axis=0)
print(f"with hip strategy: {fwd_hip:.1f} N (ankle-only was {fwd:.1f} N)")
assert fwd_hip > fwd
