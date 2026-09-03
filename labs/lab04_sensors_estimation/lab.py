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
# # Lab 4 — Sensors & State Estimation · WORKSHEET
#
# Ground truth is gone. Estimate or fall. T1–T5.

# %% [markdown]
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/microduck/blob/main/labs/lab04_sensors_estimation/lab.ipynb)
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
from microduck_sim.sensors import NoisyIMU
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
# Seated, nod the head (`neck_pitch` sine, amplitude 0.4, 0.5 Hz, 4 s) while
# logging the clean gyro (`NoisyIMU(model, data)` — no noise args) in an
# `on_tick`. Plot the three gyro axes together with the numerical derivative
# of true pitch (`np.gradient(pitch_array, DT)`), and print each axis's
# correlation with it. Which gyro axis is the pitch axis?

# %%
# TODO

# %% [markdown]
# *Answer:* gyro axis ... carries pitch rate.

# %% [markdown]
# ## T2 — Accelerometer pitch: static vs dynamic
#
# Implement `my_accel_pitch(a)` = atan2(−aₓ, a_z). Using **clean** sensors
# (`NoisyIMU(model, data)`, no noise — so only physics can fool you), measure
# its mean |error| vs true pitch over 6 s while the *seated* duck
# (a) sits quietly, (b) is shaken: in `on_tick`, apply
# `data.xfrc_applied[trunk_id, 0] = 4.0 * np.sin(2*np.pi*1.5*d.time)`.
# Skip invalid samples (`sample.valid`), and zero `xfrc_applied` afterwards.

# %%
def my_accel_pitch(a):
    # TODO
    ...


trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

# TODO: two runs; store mean errors
err_quiet = ...
err_shaken = ...

# %%
assert err_shaken > 3 * err_quiet
print(f"T2 ok: quiet {err_quiet:.3f} rad ≪ shaken {err_shaken:.3f} rad")

# %% [markdown]
# **Question:** the sensors are perfect in this task, yet the shaken estimate
# is ~17× worse. What is the accelerometer really measuring while the trunk
# accelerates?
#
# *Your answer:* ...

# %% [markdown]
# ## T3 — Gyro integration drifts
#
# Integrate the biased gyro (`gyro_bias=0.05`, `gyro_noise=0.02`):
# `pitch += gyro_y * DT` each tick, during 10 s of ground-truth balancing.
# Plot estimate vs truth. Predict *before running*: with a 0.05 rad/s bias,
# when should the error reach 0.25 rad? Then check.

# %%
# TODO

# %% [markdown]
# *Prediction vs measurement:* ...

# %% [markdown]
# ## T4 — Your complementary filter
#
# Implement `MyFilter(alpha, dt=DT)` with `update(sample) -> pitch`:
#
# - not initialized yet → use `my_accel_pitch` if the sample is valid, else 0.0
# - valid sample → `α·(pitch + gyro_y·dt) + (1−α)·accel_pitch`
# - invalid sample → gyro integration only
#
# Compare mean |error| for α ∈ {0.0, 0.9, 0.98, 1.0} over 10 s of balancing
# with noise 0.02/0.4 and bias 0.05. Use a fresh `NoisyIMU(..., seed=3)` and a
# fresh filter per α.

# %%
class MyFilter:
    def __init__(self, alpha, dt=DT):
        self.alpha, self.dt = alpha, dt
        self.pitch = None

    def update(self, sample):
        # TODO
        ...


# TODO: errors = {alpha: mean_abs_error}
errors = ...

# %%
assert min(errors, key=errors.get) in (0.9, 0.98), "fusion should beat both extremes"
print("T4 ok:", {a: round(e, 3) for a, e in errors.items()})

# %% [markdown]
# ## T5 — Balance blind 🏆
#
# Wire your filter into the balance controller: pitch from `MyFilter(0.98)`,
# rate from the raw (noisy) gyro y — use the `pitch_fn=` / `rate_fn=` hooks of
# `make_balance_controller`. Full sensor nastiness. The duck must stand 10 s
# with **no ground truth anywhere in the loop**.
#
# **Challenge (leaderboard):** max forward push your IMU-only duck survives.
# Lab 3's ground-truth controller managed ~1.1 N — how close can estimation get?
#
# **Bonus (hard):** replace the complementary filter with a 2-state Kalman
# filter (pitch + gyro bias). Does the estimated bias converge to 0.05?

# %%
# TODO: imu_balance(), stand_time(), the 10 s assert, then the push sweep
blind_time = ...
best_push = ...

# %%
assert blind_time >= 10.0
print(f"T5 ok: stands blind, survives {best_push:.1f} N")

# %% [markdown]
# ---
# The full real-robot loop — sense → estimate → control — now runs at 50 Hz
# with nothing borrowed from the simulator. Next week: we replace *your*
# controller with a trained neural network, and feed it these observations.
