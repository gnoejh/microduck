"""M4 tests: Gymnasium envs, contract compatibility, SB3 ONNX export."""

import gymnasium as gym
import numpy as np
import pytest

import microduck_sim.envs  # noqa: F401
from microduck_sim.obs import NUM_JOINTS, OBS_SIZE


@pytest.mark.parametrize("env_id", ["MicroduckStand-v0", "MicroduckWalk-v0"])
def test_env_api(env_id):
    from gymnasium.utils.env_checker import check_env

    env = gym.make(env_id)
    check_env(env.unwrapped, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_SIZE,) and obs.dtype == np.float32
    obs, reward, term, trunc, info = env.step(np.zeros(NUM_JOINTS, dtype=np.float32))
    assert "reward_terms" in info and np.isfinite(reward)
    env.close()


def test_reset_is_seeded_deterministic():
    env = gym.make("MicroduckStand-v0")
    a, _ = env.reset(seed=42)
    b, _ = env.reset(seed=42)
    assert np.allclose(a, b)
    env.close()


def test_zero_action_stands_a_while():
    """action = 0 means 'hold STAND': survives ≥ 0.6 s even with reset noise."""
    env = gym.make("MicroduckStand-v0")
    env.reset(seed=1)
    steps = 0
    for _ in range(500):
        _, _, term, trunc, _ = env.step(np.zeros(NUM_JOINTS, dtype=np.float32))
        steps += 1
        if term or trunc:
            break
    assert steps >= 30  # ≥ 0.6 s at 50 Hz
    env.close()


def test_walk_env_commands_vary():
    env = gym.make("MicroduckWalk-v0")
    cmds = set()
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        cmds.add(round(float(obs[48]), 3))
    assert len(cmds) > 5  # per-episode command sampling
    assert all(0.0 <= c <= 0.25 for c in cmds)
    env.close()


def test_sb3_export_matches_predict(tmp_path):
    """Untrained PPO → ONNX: outputs must match SB3's deterministic predict."""
    from stable_baselines3 import PPO

    from microduck_sim.export import export_sb3_to_onnx
    from microduck_sim.policy import OnnxPolicy

    env = gym.make("MicroduckStand-v0")
    model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[32, 32]),
                device="cpu", seed=0)
    ckpt = tmp_path / "model.zip"
    model.save(ckpt)
    onnx_path = export_sb3_to_onnx(ckpt, tmp_path / "model.onnx")

    policy = OnnxPolicy(onnx_path)
    rng = np.random.default_rng(0)
    for _ in range(5):
        obs = rng.normal(size=OBS_SIZE).astype(np.float32)
        onnx_action = policy.act(obs)
        sb3_action, _ = model.predict(obs, deterministic=True)
        assert np.allclose(onnx_action, np.clip(sb3_action, -0.6, 0.6), atol=1e-5)
    env.close()


def test_reward_weights_scale_terms():
    e1 = gym.make("MicroduckStand-v0").unwrapped
    e2 = gym.make("MicroduckStand-v0", reward_weights={"upright": 2.0}).unwrapped
    a = np.zeros(NUM_JOINTS, dtype=np.float32)
    e1.reset(seed=3); e2.reset(seed=3)
    _, _, _, _, i1 = e1.step(a)
    _, _, _, _, i2 = e2.step(a)
    assert i2["reward_terms"]["upright"] == pytest.approx(
        2.0 * i1["reward_terms"]["upright"]
    )


def test_unknown_reward_weight_raises():
    e = gym.make("MicroduckStand-v0", reward_weights={"tpyo": 1.0}).unwrapped
    e.reset(seed=0)
    with pytest.raises(KeyError):
        e.step(np.zeros(NUM_JOINTS, dtype=np.float32))


def test_evaluate_walking_smoke():
    from microduck_sim.evaluate import evaluate_walking

    result = evaluate_walking("walk", episodes=1, duration=2.0)
    assert set(result) == {"survival_rate", "mean_speed", "tracking_error"}
    assert result["survival_rate"] in (0.0, 1.0)
