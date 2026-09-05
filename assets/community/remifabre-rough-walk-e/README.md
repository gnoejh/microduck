# rough-walk-e (vendored from the Hugging Face Hub)

Velocity-commanded walking tuned for rough ground, by **RemiFabre**.

- Source: <https://huggingface.co/RemiFabre/microduck-rough-walk-e> (Apache-2.0)
- `policy.onnx` and `manifest.json` are the author's files, unmodified.

**Verified in our CPU sim** (`PolicyRunner`, walk model, 10 s rollouts):

| commanded vx | 0.20 | 0.25 | 0.30 | 0.40 |
|---|---|---|---|---|
| rough-walk-e | 0.071 | 0.100 | 0.152 | 0.200 m/s |
| shipped `alpha_walking` | **0.001** | 0.107 | 0.127 | 0.171 m/s |

It walks where the shipped policy has its command deadband (Lab 7's finding):
at vx = 0.2 the shipped walker stands still and this one moves. Upright for the
full 10 s at every command tested.

The author's own numbers (rough terrain, 14/110 falls vs alpha_walking's 32/110)
were measured on their setup, not reproduced here - `eval.known_limits` in the
manifest is theirs. Reproducing a stranger's claim is Lab 8 T4.
