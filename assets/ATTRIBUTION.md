# Asset Attribution

These assets are vendored from Pollen Robotics' open-source Microduck project and are
used here for **non-commercial teaching purposes**.

## `mjcf/` — robot model (MJCF XML + STL meshes)

- Source: https://github.com/pollen-robotics/microduck_rl
  (`src/mjlab_microduck/robot/microduck/`, vendored 2026-08-31)
- License: code Apache-2.0; **3D models CC BY-SA-NC** (Creative Commons
  Attribution-ShareAlike-NonCommercial) — do not reuse in commercial material.
- Copyright: Pollen Robotics. Models exported from Onshape via onshape-to-robot.

## `policies/` — pretrained ONNX policies (9)

- Source: https://huggingface.co/spaces/pollen-robotics/microduck-simulator
  (`app/public/policies/`, vendored 2026-08-31)
- Trained by Pollen Robotics with `microduck_rl` (mjlab/MuJoCo Warp + PPO); observation
  normalization is baked into each ONNX. All share the 61-dim observation contract.
- License: Apache-2.0 (code/weights published under the project's Apache-2.0 licensing).

| File | Behavior |
|------|----------|
| BEST_alpha_walking.onnx | velocity-tracking walk (legged) |
| BEST_alpha_stand.onnx | stand / balance |
| BEST_alpha_sitstand.onnx | sit ↔ stand transitions |
| alpha_ground_pick.onnx | ground pick (beak grasp) |
| ball_kick_left.onnx / ball_kick_right.onnx | ball kicks |
| roulade.onnx | rolling recovery |
| BEST_roller.onnx | roller-skating locomotion |
| BEST_roller_crouch.onnx | crouch-glide skating |

## `space/` — browser-simulator reference files

- Source: same HF Space (`app/public/robot/mjlab/`): the MJCF variants and
  `kinematics*.json` the browser playground uses to replay these policies in plain
  MuJoCo (WASM). Kept as the reference for our Python obs/action wiring (Lab 5).

## `community/` — third-party policies vendored from the Hugging Face Hub

Community Microduck checkpoints, added 2026-09-05 so Lab 8 has real strangers'
policies to compare against and the playground has more than one walker. Each
folder keeps the author's `policy.onnx` and `manifest.json` **unmodified**; the
`README.md` in each folder is ours and records what we measured in our own sim.

| Folder | Author | Source (Apache-2.0) |
|--------|--------|---------------------|
| `remifabre-rough-walk-e` | RemiFabre | https://huggingface.co/RemiFabre/microduck-rough-walk-e |
| `remifabre-rough-walk-g` | RemiFabre | https://huggingface.co/RemiFabre/microduck-rough-walk-g |
| `remifabre-flamingo-cycle` | RemiFabre | https://huggingface.co/RemiFabre/microduck-flamingo-cycle |
| `joanfox-happy-hop` | joanfox | https://huggingface.co/joanfox/microduck-happy-hop |

`course-walk-cpu` is ours (the Lab 7 run kept as the example submission).
