"""
train_re_sac_v2.py — RE-SAC v2 with LSTM wind forecast state augmentation.

What's new vs v1:
  - SOECEnvV2 state: [v_cell, dT, util, RE_now, RE_predicted]
  - RE_predicted comes from the trained LSTM forecaster (lstm_forecaster.pt)
  - Composite reward: load_matching + safety + stability
  - 100K timesteps  (10× the v1 baseline)
  - Richer W&B logging: per-component rewards, load_error, safety rate

Usage:
  python train_re_sac_v2.py --seed 1                 # single seed
  python train_re_sac_v2.py --seed $PBS_ARRAYID      # PBS array job

W&B project : mapirl-soec-rl
W&B group   : re-sac-v2-100k
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# ---------------------------------------------------------------------------
# RE forecast injector
# ---------------------------------------------------------------------------

def build_re_forecast(
    lstm_path: str,
    episode_len: int = 200,
    seq_len: int = 48,
    device: str = "cpu",
) -> np.ndarray:
    """
    Run the trained LSTM forecaster to produce a normalised RE forecast
    array of shape (episode_len,) representing t+1 predictions.

    Uses a warm-up context window of zeros then rolls through the episode.
    This is pre-computed once and passed to SOECEnvV2 at episode reset.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "forecaster"))
    from lstm_forecaster import LSTMForecaster
    from wind_data import N_FEATURES

    model = LSTMForecaster.load(lstm_path, device=device)
    model.eval()

    # Build a synthetic RE profile to generate forecasts from
    t      = np.linspace(0, 4 * np.pi, episode_len + seq_len)
    noise  = np.random.default_rng(0).normal(0, 0.05, episode_len + seq_len)
    re_all = np.clip(0.5 + 0.35 * np.sin(t) + noise, 0.0, 1.0).astype(np.float32)

    # Build feature matrix: [wind_power_norm, wind_speed_norm, hour_sin, hour_cos, dow_sin, dow_cos]
    import math
    feats = np.zeros((episode_len + seq_len, N_FEATURES), dtype=np.float32)
    feats[:, 0] = re_all
    for i in range(episode_len + seq_len):
        h = 2 * math.pi * (i % 48) / 48.0
        d = 2 * math.pi * (i % 336) / 336.0
        feats[i, 2] = math.sin(h)
        feats[i, 3] = math.cos(h)
        feats[i, 4] = math.sin(d)
        feats[i, 5] = math.cos(d)

    # Roll through to get t+1 prediction at each step
    forecast = np.zeros(episode_len, dtype=np.float32)
    for i in range(episode_len):
        window = feats[i : i + seq_len]   # (seq_len, N_FEATURES)
        mean, _ = model.predict_np(window, device=device)
        forecast[i] = float(mean[0])      # take t+1 prediction only

    return np.clip(forecast, 0.0, 1.0)


# ---------------------------------------------------------------------------
# W&B callback — logs all reward components and safety metrics
# ---------------------------------------------------------------------------
class RichWandbCallback(BaseCallback):
    def __init__(self, log_freq: int = 500):
        super().__init__()
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq != 0:
            return True
        try:
            import wandb
            log = {"timestep": self.num_timesteps}

            # Episode stats from monitor buffer
            if len(self.model.ep_info_buffer) > 0:
                ep  = self.model.ep_info_buffer[-1]
                log["train/ep_reward"] = ep["r"]
                log["train/ep_length"] = ep["l"]

            # Info from last step (injected by Monitor wrapper via locals)
            infos = self.locals.get("infos", [{}])
            if infos:
                info = infos[0]
                for key in ["load_error", "v_cell", "dT", "util",
                            "r_load", "r_safe", "r_stab"]:
                    if key in info:
                        log[f"train/{key}"] = info[key]
                if "safety_violation" in info:
                    log["train/safety_violation"] = float(info["safety_violation"])

            wandb.log(log, step=self.num_timesteps)
        except Exception:
            pass
        return True


# ---------------------------------------------------------------------------
# Safety + load-error eval callback
# ---------------------------------------------------------------------------
class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq: int = 5000, n_episodes: int = 5):
        super().__init__()
        self.eval_env   = eval_env
        self.eval_freq  = eval_freq
        self.n_episodes = n_episodes
        self.best_load_error = float("inf")

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        safe_steps = total_steps = 0
        load_errors = []

        for _ in range(self.n_episodes):
            obs, _ = self.eval_env.reset()
            done   = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                total_steps += 1
                if not info.get("safety_violation", False):
                    safe_steps += 1
                if "load_error" in info:
                    load_errors.append(info["load_error"])

        safety_rate = safe_steps / max(total_steps, 1)
        mean_load_error = float(np.mean(load_errors)) if load_errors else float("nan")

        tag = " ★" if mean_load_error < self.best_load_error else ""
        if mean_load_error < self.best_load_error:
            self.best_load_error = mean_load_error

        print(
            f"  [Eval @ {self.num_timesteps:6d}]  "
            f"safety={safety_rate:.1%}  "
            f"load_error={mean_load_error:.4f}{tag}"
        )

        try:
            import wandb
            wandb.log({
                "eval/safety_rate":      safety_rate,
                "eval/load_error":       mean_load_error,
                "eval/best_load_error":  self.best_load_error,
            }, step=self.num_timesteps)
        except Exception:
            pass

        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    t_start = time.time()

    # ---- W&B ----
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="mapirl-soec-rl",
                group="re-sac-v2-100k",
                name=f"re-sac-v2-seed{args.seed}",
                config=vars(args),
                tags=["v2", "lstm-forecast", "100k"],
            )
        except ImportError:
            print("wandb not installed — disabling")
            use_wandb = False

    # ---- LSTM forecast ----
    print(f"Loading LSTM forecaster from {args.lstm_path} …")
    re_forecast = build_re_forecast(
        lstm_path=args.lstm_path,
        episode_len=200,
        seq_len=48,
        device="cpu",   # forecaster always on CPU (lightweight)
    )
    print(f"  RE forecast ready  mean={re_forecast.mean():.3f}  std={re_forecast.std():.3f}")

    # ---- Environments ----
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))
    from envs.soec_env_v2 import SOECEnvV2

    # Pass forecast into env so RE_predicted is populated at every step
    env      = Monitor(SOECEnvV2(re_forecast=re_forecast, seed=args.seed))
    eval_env = Monitor(SOECEnvV2(re_forecast=re_forecast, seed=args.seed + 100))

    # ---- Checkpoint dir ----
    ckpt_dir = Path(args.checkpoint_dir) / f"seed{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- SAC model (RE-SAC = SAC on RE-augmented state) ----
    # Hyperparameters: inherit proven v1 config, scale buffer for 100K
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=50_000,       # 10× v1 (100K run needs larger replay)
        batch_size=256,           # larger batch for stability
        tau=0.005,
        gamma=0.99,
        gradient_steps=1,
        learning_starts=1000,     # proportionally scaled
        ent_coef="auto",
        target_entropy=-1.0,
        target_update_interval=1,
        policy_kwargs={
            "use_sde": False,
            "net_arch": [256, 256],   # deeper network for richer state
        },
        verbose=0,
        seed=args.seed,
        device=args.device,
    )

    print(
        f"\nRE-SAC v2  |  seed={args.seed}  device={args.device}\n"
        f"  obs_dim : {env.observation_space.shape[0]}  "
        f"(v_cell, dT, util, RE_now, RE_predicted)\n"
        f"  steps   : {args.total_timesteps:,}\n"
    )

    # ---- Callbacks ----
    callbacks = [
        CheckpointCallback(
            save_freq=10_000,
            save_path=str(ckpt_dir),
            name_prefix="re_sac_v2_checkpoint",
        ),
        EvalCallback(eval_env, eval_freq=5_000, n_episodes=5),
    ]
    if use_wandb:
        callbacks.append(RichWandbCallback(log_freq=500))

    # ---- Train ----
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=False,
        reset_num_timesteps=True,
    )

    # ---- Save final ----
    final_path = ckpt_dir / "re_sac_v2_final"
    model.save(str(final_path))

    elapsed = time.time() - t_start
    print(f"\nDone  seed={args.seed}  elapsed={elapsed/60:.1f} min")
    print(f"Saved → {final_path}")

    if use_wandb:
        import wandb
        wandb.log({"train/total_time_min": elapsed / 60})
        wandb.finish()

    env.close()
    eval_env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RE-SAC v2 — LSTM-augmented, 100K steps")
    p.add_argument("--seed",            type=int,   default=1)
    p.add_argument("--total-timesteps", type=int,   default=100_000)
    p.add_argument("--device",          type=str,   default="cpu")
    p.add_argument("--checkpoint-dir",  type=str,   default="checkpoints/re_sac_v2")
    p.add_argument("--lstm-path",       type=str,   default="lstm_forecaster.pt",
                   help="Path to trained LSTM .pt file")
    p.add_argument("--no-wandb",        action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(get_args())
