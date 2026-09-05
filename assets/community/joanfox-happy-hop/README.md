# happy-hop (vendored from the Hugging Face Hub)

An episodic two-foot hop, by **joanfox**. Commands are unused (zeros) - it runs
its motion from a stable standing entry.

- Source: <https://huggingface.co/joanfox/microduck-happy-hop> (Apache-2.0)
- `policy.onnx` and `manifest.json` are the author's files, unmodified.

**Verified in our CPU sim, with a caveat worth teaching**: the duck stays upright
for the full rollout, but the trunk rises only from 0.117 m to **0.124 m** - a 7 mm
hop. The author trained and evaluated against the BAM XL330 actuator model
(backlash, current saturation); our position actuators are a coarser servo model,
and the explosive part of the motion does not survive the crossing.

So this one is the **partial-transfer** example: contract-compatible, upright,
recognisably the same behaviour, quantitatively weaker. Compare with
`rough-walk-e` (transfers cleanly) and see this folder's parent README for the
three policies that do not transfer at all.
