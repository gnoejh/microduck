# Community policies (Lab 8 submissions)

One folder per submission:

```
assets/community/<your-handle>/
├── policy.onnx      obs[1,61] -> actions[1,14], < 5 MB (export.py output)
├── manifest.json    microduck_sim.evaluate.make_manifest(...) - MEASURED numbers + known_limits
└── README.md        what it is, how it was trained, what it can't do
```

Open a pull request that adds your folder. CI runs `validate_submission` on it;
after the merge the playground rebuilds and your duck appears in the browser:
`show_playground("walk", policy="<your-handle>")`.

Only walking-layout policies (twist + head command slots, the `walk` scene) are
accepted here - that is the contract every course policy speaks.

## Vendored gallery (not student submissions)

Four third-party policies from the Hugging Face Hub live here too, so Lab 8 has
strangers' work to measure and the playground's walk scene has more than one
walker. Sources and licences: `assets/ATTRIBUTION.md`.

| Folder | What it does | In our sim |
|--------|--------------|------------|
| `remifabre-rough-walk-e` | velocity walk, rough-terrain tuned | walks, no deadband at vx = 0.2 |
| `remifabre-rough-walk-g` | sibling gait, stairs + slopes | walks, no deadband at vx = 0.2 |
| `remifabre-flamingo-cycle` | stand on one foot (twist = flag/side) | lifted foot 0.093 m vs manifest 0.09 |
| `joanfox-happy-hop` | episodic two-foot hop | upright, but only a 7 mm hop |

### The manifest field that predicted transfer (measured 2026-09-05)

Seven Hub checkpoints were downloaded and run in our sim. Four kept their
behaviour; three fell within 0.5-3 s at every command tried
(`HannesVonEssen/microduck-running`, `Teethyfish/microduck-collision-flamingo-ii`,
`Nupr-Haokun/microduck-step-up`). The split lines up exactly with one line of
their manifests:

| `eval.sim_proxy` says | transferred |
|---|---|
| "CPU MuJoCo with the BAM XL330 servo model" (4 policies) | 4 / 4 |
| "mjlab / MuJoCo Warp", or no `sim_proxy` at all (3 policies) | 0 / 3 |

Same 61-dim contract, same 14 actions, same 50 Hz - and still they fall. What a
policy was *evaluated on* predicts whether it survives a change of simulator
better than anything else in the file. That is the sim2sim gap Lab 8 is about,
and the argument for filling in `eval` honestly in your own manifest.
