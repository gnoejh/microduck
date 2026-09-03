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
