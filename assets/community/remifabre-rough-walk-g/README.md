# rough-walk-g (vendored from the Hugging Face Hub)

Sibling of `rough-walk-e` with a different gait, by **RemiFabre**. The author
reports it as the only policy of the family that survived both 2 cm stairs and
a 9 deg slope.

- Source: <https://huggingface.co/RemiFabre/microduck-rough-walk-g> (Apache-2.0)
- `policy.onnx` and `manifest.json` are the author's files, unmodified.

**Verified in our CPU sim** (10 s rollouts): 0.073 / 0.096 / 0.111 / 0.182 m/s at
commanded vx = 0.2 / 0.25 / 0.3 / 0.4, upright throughout. Like `rough-walk-e`, it
has no deadband at vx = 0.2, where the shipped walker stands still.

Its manifest is honest about the cost: "+17% motor power vs alpha_walking",
"weak turn response". Two sibling policies with the same contract and different
trade-offs make a clean Lab 8 comparison.
