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
# # Lab 5 — Running a Trained Policy · SOLUTION

# %%
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from microduck_sim.obs import OBS_SIZE, ObsBuilder
from microduck_sim.policy import BEHAVIORS, OnnxPolicy, PolicyRunner
from microduck_sim.viz import record_policy, show_video

# %% [markdown]
# ## T1 — Inspect the artifacts
#
# For every behavior in `BEHAVIORS`, load the ONNX and print its input/output
# names and shapes. Verify they all share the contract (61 in, ≥14 out).

# %%
for name in BEHAVIORS:
    p = OnnxPolicy(name)
    i, o = p.session.get_inputs()[0], p.session.get_outputs()[0]
    print(f"{name:14s} {i.name}{i.shape} → {o.name}{o.shape}")
    assert i.shape[-1] == OBS_SIZE

# %% [markdown]
# ## T2 — Projected gravity is your Lab 4 pitch
#
# Implement `my_projected_gravity(quat)` (rotate world [0,0,−1] by the
# *conjugated* trunk quaternion — use `mujoco.mju_rotVecQuat`). Verify against
# `ObsBuilder.projected_gravity` while the stand policy balances, and show
# that `asin(gx)` reproduces Lab 4's pitch **exactly** (for a quaternion,
# gx = 2(w·y − z·x) — the very expression inside Lab 3's pitch formula).

# %%
def my_projected_gravity(quat_wxyz):
    w, x, y, z = quat_wxyz
    conj = np.array([w, -x, -y, -z])
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, np.array([0.0, 0.0, -1.0]), conj)
    return out


from microduck_sim.control import quat_to_pitch_roll

runner = PolicyRunner("stand")
for _ in range(100):
    runner.step()
    quat = runner.data.xquat[runner.obs_builder.trunk_id]
    mine = my_projected_gravity(quat)
    ref = runner.obs_builder.projected_gravity(runner.data)
    assert np.allclose(mine, ref, atol=1e-9)
    pitch_from_g = np.arcsin(np.clip(mine[0], -1.0, 1.0))
    assert abs(pitch_from_g - quat_to_pitch_roll(quat)[0]) < 1e-9
print("T2 ok — gravity-x IS sin(pitch): same information, no trig needed")

# %% [markdown]
# ## T3 — Drive a square
#
# Script the walker around a square: 4 × (walk 4 s at vx=0.25, then turn left
# ~90° at wz=1.0 for ~1.6 s). Plot the trunk's x–y path. How far from the
# start does it end (closure error)?

# %%
runner = PolicyRunner("walk")
path = []


def leg(duration, vx, wz):
    runner.command[0], runner.command[2] = vx, wz
    for _ in range(int(duration * 50)):
        runner.step()
        path.append(runner.data.qpos[:2].copy())


for _ in range(4):
    leg(4.0, 0.25, 0.0)
    leg(1.6, 0.0, 1.0)

path = np.array(path)
plt.plot(path[:, 0], path[:, 1]); plt.axis("equal")
plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.title("commanded square"); plt.show()
closure = np.linalg.norm(path[-1] - path[0])
print(f"upright: {runner.upright}, closure error: {closure:.2f} m")
assert runner.upright
# The square is wobbly and doesn't close — velocity commands are tracked
# imperfectly and errors integrate. (That's why real navigation adds an outer
# loop: odometry/SLAM correcting the commands. Foreshadowing for Isaac-half.)

# %% [markdown]
# ## T4 — Measure the gait with an FFT
#
# Log 10 s of walking actions. Take the FFT of `left_hip_pitch`'s action
# (subtract its mean first; sample rate 50 Hz) and find the dominant
# frequency — the *step rate* the network chose. Cross-check the hips are in
# antiphase. Careful: left/right joints use MIRRORED sign conventions (Lab 2!)
# — negate one side before correlating, or "antiphase" will look in-phase.

# %%
runner = PolicyRunner("walk")
runner.command[0] = 0.25
log = runner.run(10.0)
sig = log["action"][:, 2] - log["action"][:, 2].mean()  # left_hip_pitch
freqs = np.fft.rfftfreq(len(sig), d=0.02)
spectrum = np.abs(np.fft.rfft(sig))
f_step = freqs[1:][np.argmax(spectrum[1:])]
plt.plot(freqs, spectrum); plt.axvline(f_step, ls="--", c="r")
plt.xlim(0, 10); plt.xlabel("frequency [Hz]"); plt.title(f"gait ≈ {f_step:.2f} Hz"); plt.show()

# physical antiphase check: undo the mirrored sign convention on the right
corr = np.corrcoef(log["action"][:, 2], -log["action"][:, 11])[0, 1]
print(f"step frequency {f_step:.2f} Hz, physical L/R hip correlation {corr:+.2f}")
assert 0.5 < f_step < 5.0 and corr < 0

# %% [markdown]
# ## T5 — Challenge: overdrive & the full pipeline 🏆
#
# (a) The walker was trained up to vx = 0.25 m/s. Feed it more. Find the
# highest commanded vx (0.05 steps) where it survives 10 s — and measure the
# *achieved* speed each time. Where does extrapolation break?
# (b) Build the full pipeline: walk → shove flat (8 N, 0.2 s) → detect the
# fall (projected gravity z > −0.5) → swap to the get-up policy → detect
# recovery (gz < −0.85) → swap back to walking. One unbroken run.

# %%
def survives_at(vx):
    r = PolicyRunner("walk")
    r.command[0] = vx
    x0 = r.data.qpos[0]
    r.run(10.0)
    return r.upright, (r.data.qpos[0] - x0) / 10.0


vx = 0.25
results = {}
while vx <= 0.8:
    ok, achieved = survives_at(vx)
    results[round(vx, 2)] = (ok, round(achieved, 3))
    if not ok:
        break
    vx += 0.05
print(results)
best = max(v for v, (ok, _) in results.items() if ok)
print(f"survives up to commanded vx = {best:.2f} m/s")
# Achieved speed saturates well below the command — the network was never
# trained out there, so pushing the command harder stops helping (and
# eventually breaks the gait).

# %%
# (b) the full autonomy pipeline
r = PolicyRunner("walk")
walk, getup = r.policy, OnnxPolicy("stand")
trunk = mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
r.command[0] = 0.25
state = "walking"
events = []
for tick in range(int(20 * 50)):
    if tick == 150:  # the shove
        r.data.xfrc_applied[trunk, 0] = 8.0
    if tick == 160:
        r.data.xfrc_applied[trunk, 0] = 0.0
    r.step()
    gz = r.obs_builder.projected_gravity(r.data)[2]
    if state == "walking" and gz > -0.5:
        state = "recovering"
        r.policy, r.command[:], r.last_action[:] = getup, 0.0, 0.0
        events.append((tick / 50, "fell → get-up policy"))
    elif state == "recovering" and gz < -0.85:
        state = "walking"
        r.policy, r.last_action[:] = walk, 0.0
        r.command[0] = 0.25
        events.append((tick / 50, "recovered → walking"))
for t, e in events:
    print(f"t={t:5.1f} s  {e}")
assert state == "walking" and r.upright
print("pipeline ok — fell, got up, walked on")
