# GPU track — instructor notes (Colab T4 · RTX 4060 · paid tiers)

The official training pipeline is [`microduck_rl`](https://github.com/pollen-robotics/microduck_rl):
mjlab (MuJoCo Warp) + PPO, **CUDA required**. This is the pipeline that produced the
9 shipped ONNX policies our labs replay. Students meet it in Lab 7's GPU track
(`stage2/colab_gpu_track.ipynb`, Colab T4); it is also the conceptual bridge to the
Isaac Sim half-semester (same massively-parallel model). Everything below that is not
marked *measured* is from upstream's docs, not yet reproduced by us.

## Setup (any CUDA machine, Colab included)

```bash
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl
uv sync            # pulls a CUDA torch + warp; ~5 min on Colab
```

## Training

Upstream default is 4096 parallel envs, sized for bigger cards. Start at 1024 and raise
until VRAM is tight (`nvidia-smi -l 2`):

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 1024 --agent.max-iterations 300
```

- Upstream quotes 1–2 h to a usable gait at 4096 envs; expect 2–4× that wall-clock at
  1024–2048 envs. In class we only run ~20 min to *see* the learning curve rise; the
  usable checkpoint (`assets/checkpoints/mjlab_walk.onnx`, once trained) is the fallback.
- `uv run play` visualizes a checkpoint; resume with `--agent.load-checkpoint`.
- Export with `scripts/export.py` → ONNX with the same 61-obs contract (normalizer baked
  in); verify in OUR sim with `evaluate_walking(path)` / `PolicyRunner`, and in the
  browser by adding it to `playground/build.py`.

## Measurements (fill in as they land)

| machine | num-envs | steps/s | VRAM | minutes to upright | minutes to walking |
|---|---|---|---|---|---|
| Colab free T4 (16 GB) | 1024 | *measure* | *measure* | *measure* | *measure* |
| RTX 4060 (8 GB) | 1024 | *measure* | *measure* | *measure* | *measure* |
| Colab paid (L4 / A100) | 4096 | *measure* | *measure* | *measure* | *measure* |

## No local GPU?

1. **Google Colab** (free T4): `stage2/colab_gpu_track.ipynb`. Free GPUs are not
   guaranteed at peak hours — plan the class demo on the instructor's paid tier and
   let students run it at home.
2. **Hugging Face Jobs**: add `--hf-jobs` to any microduck_rl train command (paid,
   per-minute billing) — see their README.

## Status

- [ ] TODO(instructor): run the T4 recipe once end-to-end (spike S5 in CLAUDE.md), fill
  the table, commit `mjlab_walk.onnx` to `assets/checkpoints/` and it appears in the
  playground automatically (`playground/build.py` picks it up).
- [ ] TODO(instructor): same on the RTX 4060 and one paid-tier run.
