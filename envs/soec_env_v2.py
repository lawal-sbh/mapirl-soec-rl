from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

V_NOM=1.20; V_LOW=0.90; V_HIGH=1.40; DT_MAX=50.0
UTIL_NOM=0.70; I_MAX=1.0; TAU_V=0.85; TAU_T=0.95; EPISODE_LEN=200
DEFAULT_W_LOAD=1.0; DEFAULT_W_SAFE=5.0; DEFAULT_W_STAB=0.5

class SOECEnvV2(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self,re_profile=None,re_forecast=None,w_load=DEFAULT_W_LOAD,w_safe=DEFAULT_W_SAFE,w_stab=DEFAULT_W_STAB,seed=42):
        super().__init__()
        self.w_load=w_load; self.w_safe=w_safe; self.w_stab=w_stab
        self._rng=np.random.default_rng(seed)
        self._re_profile_fixed=re_profile; self._re_forecast_fixed=re_forecast
        self.observation_space=spaces.Box(low=np.array([V_LOW,-DT_MAX,0.,0.,0.],dtype=np.float32),high=np.array([V_HIGH,DT_MAX,1.,1.,1.],dtype=np.float32))
        self.action_space=spaces.Box(low=np.array([-0.05],dtype=np.float32),high=np.array([0.05],dtype=np.float32))
        self._v_cell=V_NOM; self._dT=0.; self._util=UTIL_NOM
        self._current=0.5; self._prev_action=0.; self._step_count=0
        self._re_profile=np.array([]); self._re_forecast=np.array([])
    def _make_re_profile(self):
        t=np.linspace(0,4*np.pi,EPISODE_LEN)
        noise=self._rng.normal(0,0.05,EPISODE_LEN)
        re=np.clip(0.5+0.35*np.sin(t)+noise,0.,1.).astype(np.float32)
        forecast=np.roll(re,-1); forecast[-1]=forecast[-2]
        return re,forecast
    def _get_obs(self):
        t=min(self._step_count,len(self._re_profile)-1)
        return np.array([self._v_cell,self._dT,self._util,float(self._re_profile[t]),float(self._re_forecast[t])],dtype=np.float32)
    def _plant_dynamics(self,action):
        alpha=0.30; beta=2.0
        self._current=np.clip(self._current+action,0.,I_MAX)
        v_ss=V_NOM+alpha*(self._current-0.5)
        self._v_cell=float(np.clip(TAU_V*self._v_cell+(1-TAU_V)*v_ss,V_LOW-0.05,V_HIGH+0.05))
        dT_drive=beta*(self._current-0.5)**2
        self._dT=float(np.clip(TAU_T*self._dT+(1-TAU_T)*dT_drive*DT_MAX,-DT_MAX,DT_MAX))
        self._util=float(np.clip(self._current/I_MAX,0.,1.))
    def reset(self,*,seed=None,options=None):
        if seed is not None: self._rng=np.random.default_rng(seed)
        if self._re_profile_fixed is not None:
            self._re_profile=self._re_profile_fixed
            self._re_forecast=self._re_forecast_fixed if self._re_forecast_fixed is not None else np.roll(self._re_profile,-1)
        else:
            self._re_profile,self._re_forecast=self._make_re_profile()
        self._v_cell=float(self._rng.uniform(1.10,1.30)); self._dT=float(self._rng.uniform(-5.,5.))
        self._util=UTIL_NOM; self._current=0.5; self._prev_action=0.; self._step_count=0
        return self._get_obs(),{}
    def step(self,action):
        a=float(np.clip(action[0],-0.05,0.05))
        self._plant_dynamics(a)
        t=min(self._step_count,len(self._re_profile)-1)
        re_now=float(self._re_profile[t])
        r_load=-abs(re_now-self._util)
        safety_violation=not(V_LOW<=self._v_cell<=V_HIGH)
        r_safe=-1. if safety_violation else 0.
        r_stab=-abs(a-self._prev_action)
        reward=self.w_load*r_load+self.w_safe*r_safe+self.w_stab*r_stab
        self._prev_action=a; self._step_count+=1
        info={"load_error":abs(re_now-self._util),"safety_violation":safety_violation,"v_cell":self._v_cell,"dT":self._dT,"util":self._util,"RE_now":re_now,"r_load":r_load,"r_safe":r_safe,"r_stab":r_stab,"wind_power_norm":re_now}
        return self._get_obs(),float(reward),False,self._step_count>=EPISODE_LEN,info
    def render(self):
        print(f"step={self._step_count:4d}  v={self._v_cell:.3f}V  dT={self._dT:+.1f}C  util={self._util:.2f}")
