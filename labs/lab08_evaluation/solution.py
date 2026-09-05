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

# %% [markdown]
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/microduck/blob/main/labs/lab08_evaluation/solution.ipynb)
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
from microduck_sim import colab; colab.setup("labs/lab08_evaluation")

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
# ## T4 — Does a stranger's policy survive OUR simulator?
#
# Seven Hub checkpoints, all Apache-2.0, all declaring the same 61/14/50 Hz
# contract. Four are vendored in `assets/community/`; three we download.
# (a) reproduce one documented number, (b) run all seven, (c) find the manifest
# field that predicts which ones transfer.

# %%
COMMUNITY = Path("../../assets/community")
VENDORED = ["remifabre-rough-walk-e", "remifabre-rough-walk-g",
            "remifabre-flamingo-cycle", "joanfox-happy-hop"]
HUB = {
    "running": "HannesVonEssen/microduck-running",
    "collision-flamingo-ii": "Teethyfish/microduck-collision-flamingo-ii",
    "step-up": "Nupr-Haokun/microduck-step-up",
}
dl = Path("downloads"); dl.mkdir(exist_ok=True)
for slug, repo in HUB.items():
    (dl / slug).mkdir(exist_ok=True)
    for f in ["policy.onnx", "manifest.json"]:
        if not (dl / slug / f).exists():
            urllib.request.urlretrieve(
                f"https://huggingface.co/{repo}/resolve/main/{f}", dl / slug / f)

flamingo = COMMUNITY / "remifabre-flamingo-cycle"
manifest = json.loads((flamingo / "manifest.json").read_text(encoding="utf-8"))
print("command spec:", json.dumps(manifest["command"]["twist"], indent=1))
print("documented lifted foot height:", manifest["eval"]["lifted_foot_height_m"], "m")

# %%
# (a) reproduce the one-foot stand. Note the commands mean something else here:
# same 13 slots, same sliders, completely different semantics — which is exactly
# why the manifest has to spell them out.
runner = PolicyRunner(OnnxPolicy("community/remifabre-flamingo-cycle"))
sites = [mujoco.mj_id2name(runner.model, mujoco.mjtObj.mjOBJ_SITE, i)
         for i in range(runner.model.nsite)]
lf = sites.index("left_foot")
runner.command[0], runner.command[1] = 1.0, 1.0  # one foot, right side down
runner.run(3.0)
measured_height = float(runner.data.site_xpos[lf, 2])
print(f"measured lifted foot height: {measured_height:.3f} m")
assert abs(measured_height - manifest["eval"]["lifted_foot_height_m"]) < 0.03
# Different simulator than the author used — same behavior, to the centimetre.

# %%
# (a, optional) the documented backward-push limit
def backward_push_survives(force):
    r = PolicyRunner(OnnxPolicy("community/remifabre-flamingo-cycle"))
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
# Gentle push held, strong push fell — consistent with the manifest.

# %%
# (b) run all seven for 5 s at zero command, (c) next to their sim_proxy.
# Survival = never fell, checked every tick. `upright` alone is instantaneous:
# a duck that face-plants and tumbles can land upright-ish again by t = 5 s and
# score a false pass. Measurement design is part of the protocol.
def survives(onnx_path, duration=5.0):
    r = PolicyRunner(OnnxPolicy(onnx_path))
    for _ in range(int(duration * 50)):
        r.step()
        if not r.upright:
            return False
    return True


folders = [COMMUNITY / v for v in VENDORED] + [dl / s for s in HUB]
audit = []
for folder in folders:
    man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    proxy = (man.get("eval") or {}).get("sim_proxy") or "(not declared)"
    audit.append((man.get("name", folder.name), survives(folder / "policy.onnx"), proxy))

print(f"{'policy':<22} {'5 s':<9} evaluated by its author in")
for name, up, proxy in audit:
    print(f"{name:<22} {'survived' if up else 'FELL':<9} {proxy[:58]}")

survived = {name for name, up, _ in audit if up}

# %%
assert survived == {"rough-walk-e", "rough-walk-g", "flamingo-cycle", "happy-hop"}
print(f"{len(survived)} of {len(audit)} transferred")

# %% [markdown]
# **The answer: `eval.sim_proxy`.** Every policy whose author evaluated it in
# *CPU MuJoCo with the BAM XL330 servo model* kept its behaviour here (4/4).
# Every policy evaluated only in *mjlab / MuJoCo Warp*, or with no `sim_proxy`
# declared at all, fell within 0.5–3 s (0/3) — at every command we tried, not
# just at zero.
#
# All seven speak an identical contract: 61 observations in, 14 actions out,
# 50 Hz, same joint order, same `DEFAULT_POSE` offsets. **The contract is a
# wire format, not a promise about physics.** A policy also depends on the
# actuator model it was trained against — Pollen's BAM model gives XL330 servos
# backlash, back-EMF, current saturation and delay; our position actuators are a
# coarser approximation. Cross that gap and a fast running gait, which leans on
# the actuator's precise force response, simply falls over.
#
# `joanfox-happy-hop` shows the middle case: upright, recognisably the same
# behaviour, but the trunk rises 7 mm where the author's sim shows a real hop.
# Partial transfer, not binary.
#
# This is why T5 asks you to write down what you evaluated on. A stranger
# reading your manifest cannot re-run your training, but they *can* check
# whether your simulator resembles theirs — and that one line tells them more
# about whether your duck will work for them than any score in the file.

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
