import argparse, sys, time
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

def build_re_forecast(lstm_path, episode_len=200, seq_len=48):
    import math
    import torch
    class _LSTM(torch.nn.Module):
        def __init__(self,n_f=8,hid=256,nl=2,hor=6):
            super().__init__()
            self.horizon=hor
            self.lstm=torch.nn.LSTM(n_f,hid,nl,batch_first=True,dropout=0.2)
            self.fc=torch.nn.Sequential(torch.nn.Linear(hid,hid//2),torch.nn.GELU(),torch.nn.Linear(hid//2,hor))
        def forward(self,x,h=None):
            o,_=self.lstm(x,h); c=o[:,-1]
            return torch.sigmoid(self.fc(c)), None
    ckpt=torch.load(lstm_path,map_location="cpu")
    cfg=ckpt["config"]
    model=_LSTM(cfg["n_features"],cfg["hidden"],cfg["layers"],cfg["horizon"])
    model.load_state_dict(ckpt["model_state"]); model.eval()
    model.eval()
    n = episode_len + seq_len
    t = np.linspace(0, 4*np.pi, n)
    noise = np.random.default_rng(0).normal(0, 0.05, n)
    re_all = np.clip(0.5 + 0.35*np.sin(t) + noise, 0., 1.).astype(np.float32)
    feats = np.zeros((n, cfg["n_features"]), dtype=np.float32)
    feats[:, 0] = re_all
    for i in range(n):
        h = 2*math.pi*(i%48)/48.; d = 2*math.pi*(i%336)/336.
        feats[i,2]=math.sin(h); feats[i,3]=math.cos(h)
        feats[i,4]=math.sin(d); feats[i,5]=math.cos(d)
    forecast = np.zeros(episode_len, dtype=np.float32)
    seq_len=cfg["seq_len"]
    for i in range(episode_len):
        w = feats[i:i+seq_len]
        import torch
        t2 = torch.from_numpy(w[np.newaxis])
        with torch.no_grad():
            mean, _ = model(t2)
        forecast[i] = float(mean[0, 0].cpu())
    return np.clip(forecast, 0., 1.)

class WandbCallback(BaseCallback):
    def __init__(self, log_freq=500):
        super().__init__(); self.log_freq=log_freq
    def _on_step(self):
        if self.n_calls % self.log_freq != 0: return True
        try:
            import wandb
            log = {"timestep": self.num_timesteps}
            if len(self.model.ep_info_buffer) > 0:
                ep = self.model.ep_info_buffer[-1]
                log["train/ep_reward"] = ep["r"]; log["train/ep_length"] = ep["l"]
            infos = self.locals.get("infos", [{}])
            if infos:
                info = infos[0]
                for k in ["load_error","v_cell","dT","util","r_load","r_safe","r_stab"]:
                    if k in info: log[f"train/{k}"] = info[k]
                if "safety_violation" in info:
                    log["train/safety_violation"] = float(info["safety_violation"])
            wandb.log(log, step=self.num_timesteps)
        except Exception: pass
        return True

class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=5000, n_episodes=5):
        super().__init__(); self.eval_env=eval_env
        self.eval_freq=eval_freq; self.n_episodes=n_episodes
        self.best_load_error=float("inf")
    def _on_step(self):
        if self.n_calls % self.eval_freq != 0: return True
        safe_steps=total_steps=0; load_errors=[]
        for _ in range(self.n_episodes):
            obs,_ = self.eval_env.reset(); done=False
            while not done:
                action,_ = self.model.predict(obs, deterministic=True)
                obs,_,term,trunc,info = self.eval_env.step(action)
                done=term or trunc; total_steps+=1
                if not info.get("safety_violation",False): safe_steps+=1
                if "load_error" in info: load_errors.append(info["load_error"])
        safety_rate=safe_steps/max(total_steps,1)
        mle=float(np.mean(load_errors)) if load_errors else float("nan")
        tag=" BEST" if mle < self.best_load_error else ""
        if mle < self.best_load_error: self.best_load_error=mle
        print(f"  [Eval @ {self.num_timesteps:6d}]  safety={safety_rate:.1%}  load_error={mle:.4f}{tag}")
        try:
            import wandb
            wandb.log({"eval/safety_rate":safety_rate,"eval/load_error":mle,"eval/best_load_error":self.best_load_error},step=self.num_timesteps)
        except Exception: pass
        return True

def train(args):
    t0=time.time()
    use_wandb=not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project="mapirl-soec-rl",group="re-sac-v2-100k",
                name=f"re-sac-v2-seed{args.seed}",config=vars(args),
                tags=["v2","lstm-forecast","100k"])
        except ImportError: use_wandb=False
    print(f"Loading LSTM from {args.lstm_path} ...")
    re_forecast = build_re_forecast(args.lstm_path)
    print(f"  forecast ready  mean={re_forecast.mean():.3f}  std={re_forecast.std():.3f}")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from envs.soec_env_v2 import SOECEnvV2
    env      = Monitor(SOECEnvV2(re_forecast=re_forecast, seed=args.seed))
    eval_env = Monitor(SOECEnvV2(re_forecast=re_forecast, seed=args.seed+100))
    ckpt_dir = Path(args.checkpoint_dir)/f"seed{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = SAC("MlpPolicy", env,
        learning_rate=3e-4, buffer_size=50000, batch_size=256,
        tau=0.005, gamma=0.99, gradient_steps=1, learning_starts=1000,
        ent_coef="auto", target_entropy=-1.0, target_update_interval=1,
        policy_kwargs={"use_sde":False,"net_arch":[256,256]},
        verbose=0, seed=args.seed, device=args.device)
    print(f"RE-SAC v2 | seed={args.seed} | obs_dim={env.observation_space.shape[0]} | steps={args.total_timesteps:,}")
    callbacks=[
        CheckpointCallback(10000, str(ckpt_dir), "re_sac_v2_ckpt"),
        EvalCallback(eval_env, eval_freq=5000, n_episodes=5),
    ]
    if use_wandb: callbacks.append(WandbCallback(500))
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=False)
    model.save(str(ckpt_dir/"re_sac_v2_final"))
    elapsed=(time.time()-t0)/60
    print(f"Done  seed={args.seed}  {elapsed:.1f} min  ->  {ckpt_dir}/re_sac_v2_final")
    if use_wandb:
        import wandb; wandb.log({"train/total_time_min":elapsed}); wandb.finish()
    env.close(); eval_env.close()

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",            type=int, default=1)
    p.add_argument("--total-timesteps", type=int, default=100000)
    p.add_argument("--device",          type=str, default="cpu")
    p.add_argument("--checkpoint-dir",  type=str, default="checkpoints/re_sac_v2")
    p.add_argument("--lstm-path",       type=str, default="lstm_forecaster.pt")
    p.add_argument("--no-wandb",        action="store_true")
    train(p.parse_args())
