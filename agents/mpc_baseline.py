import argparse, sys, time, math
from pathlib import Path
from typing import List
import numpy as np
from scipy.optimize import minimize
from stable_baselines3.common.monitor import Monitor

TAU_V=0.85; TAU_T=0.95; V_NOM=1.20; V_LOW=0.90; V_HIGH=1.40
DT_MAX=50.0; ALPHA=0.30; BETA=2.0; I_MAX=1.0

def predict_plant(v0, dT0, i0, actions):
    v,dT,i = v0,dT0,i0
    vs,dTs,utils = [],[],[]
    for a in actions:
        i = np.clip(i+a, 0., I_MAX)
        v = TAU_V*v + (1-TAU_V)*(V_NOM + ALPHA*(i-0.5))
        dT = TAU_T*dT + (1-TAU_T)*BETA*(i-0.5)**2*DT_MAX
        vs.append(v); dTs.append(dT); utils.append(i/I_MAX)
    return np.array(vs), np.array(dTs), np.array(utils)

class MPC:
    def __init__(self, horizon=10, w_load=1.0, w_safe=5.0, w_stab=0.5, du_limit=0.05):
        self.N=horizon; self.w_load=w_load; self.w_safe=w_safe
        self.w_stab=w_stab; self.du_lim=du_limit
    def cost(self, u_seq, v0, dT0, i0, re_seq, prev_u):
        vs,dTs,utils = predict_plant(v0, dT0, i0, u_seq)
        c_load = self.w_load * np.sum((utils-re_seq)**2)
        c_safe = self.w_safe * (np.sum(np.maximum(0,vs-V_HIGH)**2) + np.sum(np.maximum(0,V_LOW-vs)**2))
        u_full = np.concatenate([[prev_u], u_seq])
        c_stab = self.w_stab * np.sum(np.diff(u_full)**2)
        return c_load + c_safe + c_stab
    def solve(self, obs, re_seq, prev_u, i0):
        v0,dT0 = obs[0],obs[1]
        result = minimize(self.cost, np.zeros(self.N),
            args=(v0,dT0,i0,re_seq,prev_u), method="SLSQP",
            bounds=[(-self.du_lim, self.du_lim)]*self.N,
            options={"maxiter":100,"ftol":1e-6})
        return float(np.clip(result.x[0], -self.du_lim, self.du_lim))

def get_lstm_forecast(lstm_path, episode_len=200):
    import torch
    ckpt = torch.load(lstm_path, map_location="cpu")
    cfg  = ckpt["config"]
    class _LSTM(torch.nn.Module):
        def __init__(self,n_f,hid,nl,hor):
            super().__init__()
            self.lstm=torch.nn.LSTM(n_f,hid,nl,batch_first=True,dropout=0.2)
            self.fc=torch.nn.Sequential(torch.nn.Linear(hid,hid//2),torch.nn.GELU(),torch.nn.Linear(hid//2,hor))
        def forward(self,x,h=None):
            o,_=self.lstm(x,h); return torch.sigmoid(self.fc(o[:,-1])),None
    m=_LSTM(cfg["n_features"],cfg["hidden"],cfg["layers"],cfg["horizon"])
    m.load_state_dict(ckpt["model_state"]); m.eval()
    n=episode_len+48
    t=np.linspace(0,4*np.pi,n)
    noise=np.random.default_rng(0).normal(0,0.05,n)
    re_all=np.clip(0.5+0.35*np.sin(t)+noise,0.,1.).astype(np.float32)
    feats=np.zeros((n,cfg["n_features"]),dtype=np.float32)
    feats[:,0]=re_all
    for i in range(n):
        h=2*math.pi*(i%48)/48.; d=2*math.pi*(i%336)/336.
        feats[i,2]=math.sin(h); feats[i,3]=math.cos(h)
        feats[i,4]=math.sin(d); feats[i,5]=math.cos(d)
    fc=np.zeros(episode_len,dtype=np.float32)
    for i in range(episode_len):
        w=torch.from_numpy(feats[i:i+48][np.newaxis])
        with torch.no_grad(): mean,_=m(w)
        fc[i]=float(mean[0,0].cpu())
    return np.clip(fc,0.,1.)

def evaluate_mpc(variant, n_episodes=20, horizon=10, seed=42, lstm_path="lstm_forecaster.pt"):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from envs.soec_env_v2 import SOECEnvV2
    lstm_forecast = get_lstm_forecast(lstm_path) if variant=="lstm" else None
    env = Monitor(SOECEnvV2(re_forecast=lstm_forecast, seed=seed))
    mpc = MPC(horizon=horizon)
    rng = np.random.default_rng(seed)
    all_le, all_sr, all_rw = [], [], []
    for ep in range(n_episodes):
        obs,_ = env.reset(seed=int(rng.integers(0,10000)))
        done=False; prev_u=0.; i_curr=0.5; step=0
        ep_le=[]; ep_safe=0; ep_tot=0; ep_rw=0.
        while not done:
            if variant=="perfect":
                t_now=min(step, len(env.unwrapped._re_profile)-1)
                re_seq=np.array([env.unwrapped._re_profile[min(t_now+k,len(env.unwrapped._re_profile)-1)] for k in range(1,mpc.N+1)],dtype=np.float32)
            elif variant=="lstm" and lstm_forecast is not None:
                t_now=min(step,len(lstm_forecast)-1)
                re_seq=np.array([lstm_forecast[min(t_now+k,len(lstm_forecast)-1)] for k in range(1,mpc.N+1)],dtype=np.float32)
            else:
                re_seq=np.full(mpc.N, obs[3], dtype=np.float32)
            action=mpc.solve(obs, re_seq, prev_u, i_curr)
            i_curr=np.clip(i_curr+action, 0., I_MAX); prev_u=action
            obs,rew,term,trunc,info=env.step(np.array([action],dtype=np.float32))
            done=term or trunc; step+=1
            ep_le.append(info["load_error"]); ep_rw+=rew; ep_tot+=1
            if not info["safety_violation"]: ep_safe+=1
        all_le.append(float(np.mean(ep_le)))
        all_sr.append(ep_safe/max(ep_tot,1))
        all_rw.append(ep_rw)
    env.close()
    return {"variant":variant,"mean_load_error":float(np.mean(all_le)),
            "std_load_error":float(np.std(all_le)),
            "mean_safety_rate":float(np.mean(all_sr)),"mean_reward":float(np.mean(all_rw))}

def main(args):
    use_wandb=not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project="mapirl-soec-rl",group="mpc-baseline",
                name=f"mpc-{args.variant}",config=vars(args),tags=["mpc","baseline",args.variant])
        except ImportError: use_wandb=False
    print(f"MPC baseline  variant={args.variant}  horizon={args.horizon}  episodes={args.n_episodes}")
    t0=time.time()
    r=evaluate_mpc(args.variant,args.n_episodes,args.horizon,args.seed,args.lstm_path)
    elapsed=time.time()-t0
    print(f"{'='*50}")
    print(f"  MPC-{args.variant}")
    print(f"  load_error  : {r['mean_load_error']:.4f} +/- {r['std_load_error']:.4f}")
    print(f"  safety_rate : {r['mean_safety_rate']:.1%}")
    print(f"  mean_reward : {r['mean_reward']:.2f}")
    print(f"  time        : {elapsed:.1f}s")
    print(f"{'='*50}")
    print(f"--- Comparison ---")
    print(f"  SAC/TD3 baseline : load_error=0.0550  safety=100%")
    print(f"  RE-SAC v2 (100K) : load_error=0.0210  safety=100%")
    print(f"  MPC-{args.variant:<12}: load_error={r['mean_load_error']:.4f}  safety={r['mean_safety_rate']:.0%}")
    if use_wandb:
        import wandb
        wandb.log({f"mpc/{k}":v for k,v in r.items() if isinstance(v,(int,float))})
        wandb.summary["comparison/sac_load_error"]=0.055
        wandb.summary["comparison/re_sac_v2_load_error"]=0.021
        wandb.summary["comparison/mpc_load_error"]=r["mean_load_error"]
        wandb.finish()

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--variant",    type=str, default="lstm", choices=["perfect","lstm","naive"])
    p.add_argument("--horizon",    type=int, default=10)
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--lstm-path",  type=str, default="lstm_forecaster.pt")
    p.add_argument("--no-wandb",   action="store_true")
    main(p.parse_args())
