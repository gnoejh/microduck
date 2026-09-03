"""Train a Microduck policy with Stable-Baselines3 PPO on a laptop CPU.

Usage:
    uv run scripts/train_sb3.py [--env MicroduckStand-v0] [--steps 300000]
                                [--out runs/stand] [--n-envs 8] [--vec subproc|dummy]

Produces:
    <out>/model.zip        SB3 checkpoint (resume / evaluate)
    <out>/progress.csv     per-rollout logs (plot learning curves)

Defaults are tuned so MicroduckStand-v0 reaches full-episode standing in a
few minutes of laptop-CPU training (Lab 6's budget).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="MicroduckStand-v0")
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--out", default="runs/stand")
    parser.add_argument(
        "--n-envs", type=int, default=None,
        help="parallel envs (default: min(8, CPU count)); each is one process with --vec subproc",
    )
    parser.add_argument(
        "--vec", choices=["subproc", "dummy"], default="subproc",
        help="subproc = one process per env (real parallelism); dummy = serial in one process",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-std-init", type=float, default=-1.0,
        help="initial exploration noise (log std); -1.0 ⇒ σ≈0.37 rad. "
        "SB3's default 0.0 (σ=1.0) saturates the ±0.6 rad actions.",
    )
    parser.add_argument(
        "--rw", action="append", default=[], metavar="TERM=MULT",
        help="reward-term multiplier, repeatable: --rw upright=0.2 --rw track_vx=2",
    )
    args = parser.parse_args()
    reward_weights = {}
    for spec in args.rw:
        term, _, mult = spec.partition("=")
        reward_weights[term] = float(mult)

    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.logger import configure

    import microduck_sim.envs  # noqa: F401  (registers the envs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os

    n_envs = args.n_envs or max(1, min(8, os.cpu_count() or 1))
    vec_env = make_vec_env(
        args.env, n_envs=n_envs, seed=args.seed,
        env_kwargs={"reward_weights": reward_weights} if reward_weights else None,
        vec_env_cls=SubprocVecEnv if args.vec == "subproc" else DummyVecEnv,
    )
    print(f"{args.env}: {n_envs} envs ({args.vec}), {args.steps} steps")
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=512,
        batch_size=1024,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.001,
        policy_kwargs=dict(net_arch=[128, 128], log_std_init=args.log_std_init),
        seed=args.seed,
        verbose=1,
        device="cpu",  # MLP + CPU beats GPU transfer overhead at this size
    )
    model.set_logger(configure(str(out), ["stdout", "csv"]))
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(out / "model")
    print(f"saved {out / 'model.zip'}")


if __name__ == "__main__":
    main()
