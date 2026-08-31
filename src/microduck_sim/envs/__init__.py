"""Gymnasium environments for training Microduck policies (Labs 6–7).

Both tasks speak the SAME contract as Pollen's shipped policies (61-dim obs,
14 actions, 50 Hz, ctrl = DEFAULT_POSE + action): anything you train here can
be exported to ONNX and run by `PolicyRunner` — or published to the Hub.
"""

from gymnasium.envs.registration import register

from microduck_sim.envs.microduck_env import MicroduckStandEnv, MicroduckWalkEnv

register(id="MicroduckStand-v0", entry_point="microduck_sim.envs:MicroduckStandEnv")
register(id="MicroduckWalk-v0", entry_point="microduck_sim.envs:MicroduckWalkEnv")

__all__ = ["MicroduckStandEnv", "MicroduckWalkEnv"]
