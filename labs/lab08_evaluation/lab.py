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
# # Lab 8 — Evaluation, Robustness & Shipping · WORKSHEET
#
# Final lab: prove how good a policy is, then ship it. T1–T5.

# %% [markdown]
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/microduck/blob/main/labs/lab08_evaluation/lab.ipynb)
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
# `evaluate_walking` (3 episodes, 6 s, vx=0.25) on (a) the shipped walker
# (`"walk"`), (b) `../../assets/checkpoints/walk_ppo.onnx` (the CPU-trained
# course policy — or your own Lab 7 export if you prefer). One table; two
# sentences of interpretation: which metric separates them and why?

# %%
# TODO

# %% [markdown]
# *Your interpretation:* ...

# %% [markdown]
# ## T2 — The push tournament
#
# `max_push` (resolution 0.2, limit 3.0) for `stand_ppo.onnx` (course-trained)
# and the shipped `"stand"`. Add Lab 3's hand-PD number (1.1 N). Rank all
# three and explain the ranking — especially why a 15-line PD controller
# beats a 1M-step RL policy.

# %%
# TODO
ours = ...
shipped = ...

# %%
assert ours < 1.1 < shipped
print("T2 ok — now write the explanation, it's the course's biggest lesson")

# %% [markdown]
# *Your explanation:* ...

# %% [markdown]
# ## T3 — Variant sweep (sim2sim)
#
# `variant_sweep` at vx=0.25 (2 episodes) for the shipped walker and the
# course policy. Which cell of the table is the interesting one, and what
# does it say about training distributions?

# %%
# TODO

# %% [markdown]
# *Your answer:* ...

# %% [markdown]
# ## T4 — Reproduce a community eval
#
# Download `policy.onnx` + `manifest.json` from
# `https://huggingface.co/RemiFabre/microduck-flamingo-cycle/resolve/main/`.
# Read the manifest: command semantics and `eval` section. Reproduce TWO
# documented facts in our sim:
# (a) command [flag=1, side=+1] for 3 s → measure the lifted foot height
# via `data.site_xpos` (manifest says 0.09 m; ±0.03 tolerance),
# (b) its `known_limits` say it falls to strong backward pushes — verify a
# gentle backward shove (0.3 N × 0.1 s) is survived and a strong one (2 N)
# is not.

# %%
# TODO
measured_height = ...

# %%
assert abs(measured_height - 0.09) < 0.03
print(f"T4 ok — reproduced a stranger's eval to the cm: {measured_height:.3f} m")

# %% [markdown]
# ## T5 — Ship it 🏆
#
# Assemble `my_release/` for YOUR best policy from Labs 6–7:
# - `policy.onnx` (your export)
# - `manifest.json` — follow the flamingo schema: contract numbers, honest
#   command semantics, the eval numbers you *measured today* (T1–T3), and
#   real `known_limits`
# - `README.md` (3 sentences minimum)
#
# Use `make_manifest(...)` for the schema and run `validate_submission("my_release",
# quick=False)` — that is exactly what CI runs on your pull request.
#
# Then **deploy it**: open a pull request that adds your folder as
# `assets/community/<your-github-handle>/`. When it merges, the playground rebuilds
# and your duck is in everyone's browser (`show_playground("walk", policy=...)`).
# Publishing to the Hugging Face Hub as well is optional.
#
# **Manifest honesty is the grade**: we will run your eval battery.

# %%
# TODO

# %% [markdown]
# ---
# 🎓 That's the course. Pose → control → estimate → run → train → prove →
# ship, hardware-free. The Isaac Sim half starts from exactly this loop.
