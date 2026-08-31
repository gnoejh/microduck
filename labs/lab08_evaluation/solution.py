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
# # Lab 8 — Evaluation, Robustness & Shipping · SOLUTION

# %%
import json
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from microduck_sim.evaluate import evaluate_walking, max_push, variant_sweep
from microduck_sim.policy import OnnxPolicy, PolicyRunner

# %% [markdown]
# ## T1 — Scorecards
#
# Run `evaluate_walking` (3 episodes, 6 s, vx=0.25) on (a) the shipped
# walker and (b) the committed CPU-trained `walk_ppo.onnx`. Put both in one
# table and interpret the gap.

# %%
rows = {
    "shipped walker": evaluate_walking("walk", episodes=3, duration=6.0),
    "our walk_ppo": evaluate_walking("../../assets/checkpoints/walk_ppo.onnx",
                                     episodes=3, duration=6.0),
}
for name, r in rows.items():
    print(f"{name:15s} survival {r['survival_rate']:.0%}  "
          f"speed {r['mean_speed']:+.3f}  tracking err {r['tracking_error']:.3f}")
# Expect: comparable survival, but the shipped walker moves faster/cleaner.
# Survival is the easy metric; tracking is where training scale shows.

# %% [markdown]
# ## T2 — The push tournament
#
# Measure `max_push` (resolution 0.2 N) for the committed `stand_ppo.onnx`
# and the shipped `stand` policy (cap it at 3 N to save time). Add Lab 3's
# hand-PD result (1.1 N, from that lab) and rank the three. Explain the
# ranking in two sentences.

# %%
ours = max_push("../../assets/checkpoints/stand_ppo.onnx", resolution=0.2, limit=3.0)
shipped = max_push("stand", resolution=0.2, limit=3.0)
print(f"our stand_ppo: {ours:.1f} N   hand PD (Lab 3): 1.1 N   shipped stand: ≥{shipped:.1f} N")
assert ours < 1.1 < shipped
# Our policy trained in a world without pushes — robustness you don't train
# for doesn't exist. The shipped policy saw randomized physics AND can stand
# back up, so a push is barely an event for it.

# %% [markdown]
# ## T3 — Variant sweep (sim2sim)
#
# Sweep the shipped walker and our walk_ppo across variants
# ("full", "walk", "backlash") at vx = 0.25 and compare degradation.

# %%
for name, pol in [("shipped", "walk"),
                  ("ours", "../../assets/checkpoints/walk_ppo.onnx")]:
    sweep = variant_sweep(pol, vx=0.25, episodes=2)
    line = "  ".join(f"{v}: surv {r['survival_rate']:.0%}/spd {r['mean_speed']:+.2f}"
                     for v, r in sweep.items())
    print(f"{name:8s} {line}")
# The interesting cell is OUR policy on 'backlash': it trained on the clean
# 'walk' model only, so gear chatter is out-of-distribution physics for it.

# %% [markdown]
# ## T4 — Reproduce a community eval
#
# `RemiFabre/microduck-flamingo-cycle` publishes policy.onnx + manifest.json.
# Download both, print the manifest's command spec, then reproduce TWO
# documented eval facts in our sim:
# (a) commanded [1, +1] → lifted-foot height ≈ 0.09 m;
# (b) "falls on backward pushes ≥ 0.18 m/s" → verify it survives a gentle
# backward shove but falls to a strong one.

# %%
base = "https://huggingface.co/RemiFabre/microduck-flamingo-cycle/resolve/main"
dl = Path("downloads"); dl.mkdir(exist_ok=True)
for f in ["policy.onnx", "manifest.json"]:
    if not (dl / f).exists():
        urllib.request.urlretrieve(f"{base}/{f}", dl / f)

manifest = json.loads((dl / "manifest.json").read_text(encoding="utf-8"))
print("command spec:", json.dumps(manifest["command"]["twist"], indent=1))
print("documented lifted foot height:", manifest["eval"]["lifted_foot_height_m"], "m")

# (a) reproduce the one-foot stand
runner = PolicyRunner(OnnxPolicy(dl / "policy.onnx"))
sites = [mujoco.mj_id2name(runner.model, mujoco.mjtObj.mjOBJ_SITE, i)
         for i in range(runner.model.nsite)]
lf = sites.index("left_foot")
runner.command[0], runner.command[1] = 1.0, 1.0  # one foot, right side down
runner.run(3.0)
height = float(runner.data.site_xpos[lf, 2])
print(f"measured lifted foot height: {height:.3f} m")
assert abs(height - manifest["eval"]["lifted_foot_height_m"]) < 0.03
# Different simulator than the author used — same behavior, to the cm.

# %%
# (b) the documented backward-push limit
def backward_push_survives(force):
    r = PolicyRunner(OnnxPolicy(dl / "policy.onnx"))
    r.command[0], r.command[1] = 1.0, 1.0
    trunk = mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    for tick in range(400):
        if tick == 150:
            r.data.xfrc_applied[trunk, 0] = -force
        if tick == 155:
            r.data.xfrc_applied[trunk, 0] = 0.0
        r.step()
        if not r.upright:
            return False
    return True


for f in [0.3, 2.0]:
    print(f"backward push {f} N: {'survives' if backward_push_survives(f) else 'falls'}")
# Gentle push held, strong push fell — consistent with the manifest's
# "known_limits". Documented limits you can CHECK are what make a stranger's
# policy usable.

# %% [markdown]
# ## T5 — Ship your best policy 🏆
#
# Assemble a release folder for your best policy (here: the course's
# walk_ppo as a stand-in): `policy.onnx`, an honest `manifest.json` (contract,
# command semantics, the eval numbers you MEASURED above, known limits), and
# a README. Then (live, requires `huggingface-cli login`) push it to the Hub.

# %%
release = Path("my_release"); release.mkdir(exist_ok=True)
import shutil
shutil.copy("../../assets/checkpoints/walk_ppo.onnx", release / "policy.onnx")

my_eval = rows["our walk_ppo"]
manifest = {
    "schema_version": 2,
    "model_api": 1,
    "name": "course-walk-cpu",
    "kind": "perpetual",
    "obs_len": 61, "action_len": 14, "action_scale": 1.0,
    "entry_pose": "standing",
    "command": {"twist": ["vx m/s in [0, 0.25]", "unused (0)", "unused (0)"],
                "head": "unused (zeros)", "idle": [0, 0, 0]},
    "robot": {"model": "microduck", "servos": "xl330", "control_hz": 50},
    "training": {"task_id": "MicroduckWalk-v0 (course SB3 track)",
                 "repo": "course repo", "steps": 2_000_000, "hardware": "laptop CPU"},
    "eval": {
        "sim_proxy": "CPU MuJoCo, walk model",
        "survival_rate": my_eval["survival_rate"],
        "mean_speed_at_cmd_0.25": round(my_eval["mean_speed"], 3),
        "tracking_error": round(my_eval["tracking_error"], 3),
        "max_push_N": ours,
        "known_limits": "trained without pushes (max_push above); "
                        "clean-model physics only — degrades on backlash variant (T3)",
    },
}
# encoding="utf-8" matters: on Korean/Japanese Windows, write_text defaults
# to the locale codepage (cp949/cp932) and crashes on characters like "—".
(release / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
(release / "README.md").write_text(
    "# course-walk-cpu\n\nMicroduck walking policy trained on a laptop CPU "
    "(course SB3 track, 2M steps). Honest manifest inside — check "
    "`eval.known_limits` before use.\n",
    encoding="utf-8",
)
print((release / "manifest.json").read_text(encoding="utf-8")[:400], "...")

# %%
# The actual upload (uncomment after `huggingface-cli login`):
# from huggingface_hub import HfApi
# api = HfApi()
# repo = api.create_repo("YOUR_NAME/microduck-course-walk", repo_type="model", exist_ok=True)
# api.upload_folder(folder_path="my_release", repo_id=repo.repo_id)
# print("published:", repo.repo_id)

# %% [markdown]
# 🎓 **Course complete.** You can pose it, control it, estimate its state,
# run its brain, train a new one, prove how good it is, and ship it — all
# without ever touching the robot. See you in Isaac Sim.
