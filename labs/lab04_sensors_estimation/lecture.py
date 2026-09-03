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
# # Lab 4 — Sensors & State Estimation 🧭
#
# **Microduck Simulation Course · session 4 of 8**
#
# Last week's balance controller read `data.qpos[3:7]` — the simulator's
# perfect ground truth. **The real robot has no such thing.** It has an IMU:
# a gyroscope and an accelerometer, noisy and biased. Today:
# 1. What an IMU actually measures (surprises included)
# 2. Modeling sensor imperfections
# 3. Three estimators: accel-only, gyro-only, and the complementary filter
# 4. Balancing on an estimate — closing the loop for real

# %% [markdown]
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/microduck/blob/main/labs/lab04_sensors_estimation/lecture.ipynb)
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
from microduck_sim import colab; colab.setup("labs/lab04_sensors_estimation")

# %%
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from microduck_sim.model import actuator_names, load_model, reset_to_keyframe
from microduck_sim.control import (
    keyframe_ctrl, make_balance_controller, make_push, quat_to_pitch_roll,
    run_control_loop, sine,
)
from microduck_sim.sensors import NoisyIMU, read_sensor
from microduck_sim.estimation import ComplementaryFilter, accel_pitch
from microduck_sim.viz import record, show_video

model = load_model("full")
data = mujoco.MjData(model)
acts = actuator_names(model)
stand = keyframe_ctrl(model, "STAND")

# %% [markdown]
# ## What the IMU measures
#
# The trunk carries an `imu` site with a **gyro** (`imu_ang_vel`, angular
# velocity, rad/s, body frame) and an **accelerometer** (`imu_accel`).
# The accelerometer does *not* measure acceleration — it measures **specific
# force**: acceleration *minus gravity*, in the sensor's own frame:
#
# - sitting still → reads **+g upward** (≈ [0, 0, 9.81] when level)
# - **free fall → reads zero**
# - tilted → gravity's direction in the sensor frame shifts → *tilt is readable*
#
# Watch both effects in the duck's first moments — it spawns a hair above the
# floor, so its first ~30 ms *are* free fall:

# %%
reset_to_keyframe(model, data, "STAND")
rows = []
for _ in range(100):
    mujoco.mj_step(model, data)
    rows.append([data.time, *read_sensor(model, data, "imu_accel")])
rows = np.array(rows)
plt.plot(rows[:, 0], np.linalg.norm(rows[:, 1:], axis=1))
plt.axhline(9.81, ls="--", c="gray", label="g")
plt.xlabel("t [s]"); plt.ylabel("|accel| [m/s²]")
plt.title("free fall (0) → touchdown spike → standing (≈g)"); plt.legend(); plt.show()

# %% [markdown]
# > Rule you'll re-learn today the hard way: **a sensor reading is only as good
# > as the assumptions it was read under.**
#
# ## Tilt from the accelerometer
#
# When (and only when) the duck is quasi-static, the accelerometer is a
# plumb line: `pitch ≈ atan2(−aₓ, a_z)`. That's `accel_pitch` in the library.
#
# ## Making the sensors honest
#
# MuJoCo's sensors are perfect; real ones aren't. `NoisyIMU` adds Gaussian
# noise and a constant **gyro bias** (cheap gyros always have one):

# %%
imu = NoisyIMU(model, data, gyro_noise=0.02, accel_noise=0.4, gyro_bias=0.05, seed=0)

reset_to_keyframe(model, data, "STAND")
balance = make_balance_controller(model, data, stand)  # ground-truth balance, for now
trace = {"true": [], "acc": [], "t": []}

def on_tick(m, d):
    s = imu.read()
    trace["t"].append(d.time)
    trace["true"].append(quat_to_pitch_roll(d.qpos[3:7])[0])
    trace["acc"].append(accel_pitch(s.accel) if s.valid else np.nan)

run_control_loop(model, data, balance, 5.0, on_tick=on_tick)
plt.plot(trace["t"], trace["acc"], alpha=0.5, label="accel-only pitch (noisy)")
plt.plot(trace["t"], trace["true"], "k", label="true pitch")
plt.xlabel("t [s]"); plt.ylabel("pitch [rad]"); plt.legend(); plt.show()

# %% [markdown]
# Right *average*, terrible per-sample. The gyro is the opposite — its rate is
# clean, so integrating it gives a smooth pitch... that **drifts**: our 0.05
# rad/s bias becomes 0.05 rad of error *per second*, forever.
#
# ## The complementary filter
#
# Combine them by frequency: trust the gyro for fast changes, let the
# accelerometer slowly pull the drift out. One line, run at 50 Hz:
#
# ```python
# pitch = α · (pitch + gyro_y · dt) + (1 − α) · accel_pitch
# ```
#
# α = 0.98 at 50 Hz ⇒ time constant ≈ 1 s: the gyro rules below a second,
# the accelerometer rules beyond it.
#
# **One trap.** At t = 0 the accel reads (0, 0, 0) — free fall + unpopulated
# sensors. `atan2(−0, 0) = π`: the filter would initialize believing the duck
# is UPSIDE DOWN, command a huge correction, and faceplant instantly (we tried;
# it falls in 0.28 s — worse than no controller). Hence the `valid` flag:
# no gravity in the reading → don't use it. Real IMU drivers do exactly this.
#
# ## The finale: balance on the estimate
#
# Same balance controller as Lab 3 — but pitch and rate now come from the
# noisy IMU through the filter. No ground truth anywhere in the loop:

# %%
def imu_balance(alpha=0.98):
    local_imu = NoisyIMU(model, data, gyro_noise=0.02, accel_noise=0.4,
                         gyro_bias=0.05, seed=0)
    filt = ComplementaryFilter(alpha=alpha)
    last = {"gyro_y": 0.0}

    def pitch_fn():
        s = local_imu.read()
        last["gyro_y"] = float(s.gyro[1])
        return filt.update(s)

    return make_balance_controller(model, data, stand, pitch_fn=pitch_fn,
                                   rate_fn=lambda: last["gyro_y"])


def stand_time(fn, tmax=10.0, on_tick=None):
    reset_to_keyframe(model, data, "STAND")
    log = run_control_loop(model, data, fn, tmax, on_tick=on_tick)
    below = log["qpos"][:, 2] < 0.10
    return float(log["t"][np.argmax(below)]) if below.any() else tmax


for alpha, label in [(0.98, "complementary α=0.98"),
                     (1.0, "gyro-only (α=1): drifts"),
                     (0.0, "accel-only (α=0)")]:
    print(f"{label:28s} stands {stand_time(imu_balance(alpha)):5.2f} s")

# %% [markdown]
# Fused wins. Accel-only *seems* to survive quiet standing — until something
# dynamic happens. Push each one (0.8 N, the fused filter shrugs it off):

# %%
for alpha, label in [(0.98, "complementary"), (0.0, "accel-only")]:
    push = make_push(model, data, force=0.8, axis=0)
    t = stand_time(imu_balance(alpha), on_tick=push)
    data.xfrc_applied[:] = 0
    print(f"{label:15s} 0.8 N push: {'survives' if t >= 10 else f'falls at {t:.2f} s'}")

# %% [markdown]
# During the push the duck accelerates → the accelerometer reads gravity
# **plus** the push → its "pitch" is a lie → the accel-only controller reacts
# to fiction and falls. The gyro doesn't care about linear acceleration, and
# the fused filter inherits that immunity where it matters.
#
# ## Recap
#
# - accelerometer = specific force: g when static, **0 in free fall**, corrupted
#   by any dynamics; gyro = clean rate but drifts through bias
# - complementary filter: one line, gyro fast / accel slow, α=0.98 @ 50 Hz
# - validate your sensors — `atan2(0,0)` once put our duck on its face
# - the full real-robot loop now runs: sense → estimate → control, at 50 Hz
# - library: `NoisyIMU`, `accel_pitch`, `ComplementaryFilter`
#
# **Worksheet: T1–T5.** Next week the neural networks arrive: we load Pollen's
# trained walking policy and feed it these very observations.
