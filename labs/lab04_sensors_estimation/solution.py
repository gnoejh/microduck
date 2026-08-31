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
# # Lab 4 — Sensors & State Estimation · SOLUTION

# %%
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from microduck_sim.model import actuator_names, load_model, reset_to_keyframe
from microduck_sim.control import (
    keyframe_ctrl, make_balance_controller, make_push, quat_to_pitch_roll,
    run_control_loop, sine,
)
from microduck_sim.sensors import NoisyIMU
from microduck_sim.estimation import accel_pitch
from microduck_sim.viz import record, show_video

model = load_model("full")
data = mujoco.MjData(model)
acts = actuator_names(model)
stand = keyframe_ctrl(model, "STAND")
sit = keyframe_ctrl(model, "SIT")
DT = 0.02

# %% [markdown]
# ## T1 — Know your axes
#
# Seated, nod the head (`neck_pitch` sine, amp 0.4, 0.5 Hz, 4 s) while logging
# the (clean) gyro. Plot all three gyro axes; determine which axis carries the
# pitch rate by correlating each with the numerical derivative of true pitch.

# %%
imu_clean = NoisyIMU(model, data)
amp = np.zeros(model.nu)
amp[acts.index("neck_pitch")] = 0.4

gyro_log, pitch_log = [], []

def on_tick(m, d):
    gyro_log.append(imu_clean.read().gyro.copy())
    pitch_log.append(quat_to_pitch_roll(d.qpos[3:7])[0])

reset_to_keyframe(model, data, "SIT")
log = run_control_loop(model, data, sine(sit, amp, 0.5), 4.0, on_tick=on_tick)
G, P = np.array(gyro_log), np.array(pitch_log)
rate = np.gradient(P, DT)
for i, axis in enumerate("xyz"):
    plt.plot(log["t"], G[:, i], label=f"gyro {axis}")
    print(f"gyro {axis} vs pitch-rate correlation: {np.corrcoef(G[:, i], rate)[0, 1]:+.2f}")
plt.plot(log["t"], rate, "k--", label="d(pitch)/dt")
plt.xlabel("t [s]"); plt.legend(); plt.show()
# → the y axis is the pitch axis (correlation ≈ +1).

# %% [markdown]
# ## T2 — Accelerometer pitch: static vs dynamic
#
# Implement `my_accel_pitch(a) = atan2(−aₓ, a_z)`. Using **clean** sensors (so
# only physics can fool you), measure its mean |error| vs true pitch while the
# seated duck (a) sits quietly, (b) is shaken by a sinusoidal horizontal force
# `f·sin(2π·1.5·t)` on the trunk (`data.xfrc_applied[trunk_id, 0]`) with
# f = 4 N, for 6 s each.

# %%
def my_accel_pitch(a):
    return float(np.arctan2(-a[0], a[2]))


trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")


def accel_error(shake_force=0.0, duration=6.0):
    imu = NoisyIMU(model, data, seed=1)  # clean: isolate the dynamics effect
    errs = []

    def on_tick(m, d):
        d.xfrc_applied[trunk_id, 0] = shake_force * np.sin(2 * np.pi * 1.5 * d.time)
        s = imu.read()
        if s.valid:
            errs.append(abs(my_accel_pitch(s.accel) - quat_to_pitch_roll(d.qpos[3:7])[0]))

    from microduck_sim.control import constant
    reset_to_keyframe(model, data, "SIT")
    run_control_loop(model, data, constant(sit), duration, on_tick=on_tick)
    data.xfrc_applied[:] = 0
    return np.mean(errs)


quiet, shaken = accel_error(0.0), accel_error(4.0)
print(f"mean |error| quiet: {quiet:.3f} rad, shaken: {shaken:.3f} rad")
assert shaken > 3 * quiet
# The accelerometer measures specific force = gravity MINUS acceleration.
# While the trunk is being accelerated, the 'plumb line' points somewhere
# between 'down' and 'backwards' — the tilt it implies is fiction. Measured:
# quiet 0.005 rad vs shaken 0.086 rad (17×).

# %% [markdown]
# ## T3 — Gyro integration drifts
#
# Implement a pure gyro integrator (`pitch += gyro_y · dt`, biased gyro:
# bias 0.05, noise 0.02). Run it during 10 s of ground-truth balancing and
# plot estimate vs truth. How long until the error exceeds 0.25 rad?

# %%
imu = NoisyIMU(model, data, gyro_noise=0.02, gyro_bias=0.05, seed=2)
est = {"pitch": 0.0}
tr = {"t": [], "est": [], "true": []}

def on_tick(m, d):
    s = imu.read()
    est["pitch"] += float(s.gyro[1]) * DT
    tr["t"].append(d.time); tr["est"].append(est["pitch"])
    tr["true"].append(quat_to_pitch_roll(d.qpos[3:7])[0])

fn = make_balance_controller(model, data, stand)
reset_to_keyframe(model, data, "STAND")
run_control_loop(model, data, fn, 10.0, on_tick=on_tick)
t, e = np.array(tr["t"]), np.abs(np.array(tr["est"]) - np.array(tr["true"]))
plt.plot(t, tr["est"], label="gyro-integrated"); plt.plot(t, tr["true"], "k", label="true")
plt.xlabel("t [s]"); plt.ylabel("pitch [rad]"); plt.legend(); plt.show()
t_bad = t[np.argmax(e > 0.25)]
print(f"error exceeds 0.25 rad at t = {t_bad:.1f} s (bias 0.05 rad/s ⇒ ~5 s, as predicted)")
assert e[-1] > e[len(e) // 4]  # drift grows

# %% [markdown]
# ## T4 — Your complementary filter
#
# Implement `MyFilter` (α, dt): gyro-integrate, accel-correct, and **handle
# invalid samples** (before init: wait for a valid one; after: integrate only).
# Compare mean |error| for α ∈ {0.0, 0.9, 0.98, 1.0} over 10 s of balancing
# with full sensor nastiness (noise 0.02/0.4, bias 0.05).

# %%
class MyFilter:
    def __init__(self, alpha, dt=DT):
        self.alpha, self.dt = alpha, dt
        self.pitch = None

    def update(self, sample):
        g = float(sample.gyro[1])
        if self.pitch is None:
            self.pitch = my_accel_pitch(sample.accel) if sample.valid else 0.0
        elif sample.valid:
            self.pitch = self.alpha * (self.pitch + g * self.dt) + \
                         (1 - self.alpha) * my_accel_pitch(sample.accel)
        else:
            self.pitch += g * self.dt
        return self.pitch


def filter_error(alpha):
    imu = NoisyIMU(model, data, gyro_noise=0.02, accel_noise=0.4, gyro_bias=0.05, seed=3)
    filt = MyFilter(alpha)
    errs = []

    def on_tick(m, d):
        errs.append(abs(filt.update(imu.read()) - quat_to_pitch_roll(d.qpos[3:7])[0]))

    fn = make_balance_controller(model, data, stand)
    reset_to_keyframe(model, data, "STAND")
    run_control_loop(model, data, fn, 10.0, on_tick=on_tick)
    return np.mean(errs)


errors = {a: filter_error(a) for a in [0.0, 0.9, 0.98, 1.0]}
print({a: round(e, 3) for a, e in errors.items()})
assert min(errors, key=errors.get) in (0.9, 0.98)  # fusion beats both extremes
print("T4 ok — fusion wins")

# %% [markdown]
# ## T5 — Balance blind 🏆
#
# Wire YOUR filter into the balance controller (pitch from `MyFilter`, rate =
# raw gyro y) with full sensor nastiness, and stand 10 s. Challenge: find the
# max forward push your IMU-only duck survives, and compare with Lab 3's
# ground-truth number (~1.1 N). What does the difference cost you?

# %%
def imu_balance(alpha=0.98):
    imu = NoisyIMU(model, data, gyro_noise=0.02, accel_noise=0.4, gyro_bias=0.05, seed=4)
    filt = MyFilter(alpha)
    last = {"g": 0.0}

    def pitch_fn():
        s = imu.read()
        last["g"] = float(s.gyro[1])
        return filt.update(s)

    return make_balance_controller(model, data, stand, pitch_fn=pitch_fn,
                                   rate_fn=lambda: last["g"])


def stand_time(fn, tmax=10.0, on_tick=None):
    reset_to_keyframe(model, data, "STAND")
    log = run_control_loop(model, data, fn, tmax, on_tick=on_tick)
    below = log["qpos"][:, 2] < 0.10
    return float(log["t"][np.argmax(below)]) if below.any() else tmax


assert stand_time(imu_balance()) >= 10.0
print("stands blind ✔")

f = 0.0
while f < 3.0:
    push = make_push(model, data, force=f + 0.1, axis=0)
    t = stand_time(imu_balance(), on_tick=push)
    data.xfrc_applied[:] = 0
    if t < 10.0:
        break
    f += 0.1
print(f"max IMU-only push: {f:.1f} N (ground truth managed ~1.1 N)")
# The gap is the price of estimation error and lag — sensing quality is
# control quality. This trade returns in Lab 7 as domain randomization.
