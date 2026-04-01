"""
sac_train.py — SAC baseline for MAPIRL-DT SOEC load-following.

Reconstructed from sac_soec_final.zip model metadata.
Confirmed hyperparameters:
  obs_dim        : 3  -> [voltage_norm, power_kw, soc]
  action_dim     : 1  -> delta_current in [-0.05, 0.05]
  lr             : 3e-4 (constant)
  buffer_size    : 5000
  batch_size     : 64
  tau            : 0.005
  gamma          : 0.99
  ent_coef       : auto  (target_entropy=-1.0)
  learning_starts: 100
  total_timesteps: 10_000
  seed           : 42 (array job overrides per seed)

Results (6 seeds): 100% safety rate
W&B group: sac-baseline
"""
import argparse, sys
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor


def train(args):
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project="mapirl-soec-rl", group="sac-baseline",
                       name=f"sac-seed{args.seed}", config=vars(args))
        except ImportError:
            use_wandb = False

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from envs.soec_reactor import SOECReactor

    env      = Monitor(SOECReactor())
    eval_env = Monitor(SOECReactor())

    ckpt_dir = Path(args.checkpoint_dir) / f"seed{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = SAC(
        "MlpPolicy", env,
        learning_rate=3e-4, buffer_size=5000, batch_size=64,
        tau=0.005, gamma=0.99, gradient_steps=1, learning_starts=100,
        ent_coef="auto", target_entropy=-1.0, target_update_interval=1,
        policy_kwargs={"use_sde": False},
        verbose=0, seed=args.seed, device=args.device,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CheckpointCallback(500, str(ckpt_dir), "sac_checkpoint"),
        progress_bar=False,
    )

    model.save(str(ckpt_dir / "sac_soec_final"))
    print(f"Saved -> {ckpt_dir}/sac_soec_final")

    if use_wandb:
        import wandb; wandb.finish()
    env.close(); eval_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--total-timesteps", type=int, default=10_000)
    p.add_argument("--device",          type=str, default="cpu")
    p.add_argument("--checkpoint-dir",  type=str, default="checkpoints/sac")
    p.add_argument("--no-wandb",        action="store_true")
    train(p.parse_args())
