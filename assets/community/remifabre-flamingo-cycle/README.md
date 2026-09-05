# flamingo-cycle (vendored from the Hugging Face Hub)

Stand on one foot on command, by **RemiFabre** - the checkpoint Lab 8 T4 uses to
reproduce a stranger's published eval.

- Source: <https://huggingface.co/RemiFabre/microduck-flamingo-cycle> (Apache-2.0)
- `policy.onnx` and `manifest.json` are the author's files, unmodified.

**Command semantics differ from the walking layout** - same 61-dim contract, same
sliders, different meaning (this is the point of the manifest):
`twist[0]` = 0 two feet / 1 one foot, `twist[1]` = +1 right foot down, -1 left foot down.

**Verified in our CPU sim**: commanded `[1, +1]`, the left foot lifts to
**0.093 m** against the manifest's documented 0.09 m - a stranger's number
reproduced to the centimetre in a different simulator.

Vendored so Lab 8 works without a network round-trip; the notebook still shows
the download-from-the-Hub path.
