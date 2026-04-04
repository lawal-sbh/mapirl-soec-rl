import logging, warnings
from typing import Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("idaes").setLevel(logging.ERROR)
logging.getLogger("pyomo").setLevel(logging.ERROR)

V_LOW=1.00; V_HIGH=1.50; V_NOM=1.287
T_NOM=1023.0; T_DEV_MAX=50.0
UTIL_NOM=0.70; UTIL_MIN=0.10; UTIL_MAX=0.90
EPISODE_LEN=200
DEFAULT_W_LOAD=1.0; DEFAULT_W_SAFE=5.0; DEFAULT_W_STAB=0.5

FEED_COMP  = {"H2": 0.01, "H2O": 0.99}
SWEEP_COMP = {"O2": 0.2074, "H2O": 0.0099, "CO2": 0.0003, "N2": 0.7732, "Ar": 0.0092}

def build_soec_flowsheet():
    import pyomo.environ as pyo
    from idaes.core import FlowsheetBlock
    from idaes.models_extra.power_generation.unit_models.soec_design import SoecDesign, EosType
    from idaes.models_extra.power_generation.properties.natural_gas_PR import get_prop
    from idaes.models.properties.modular_properties.base.generic_property import GenericParameterBlock
    from idaes.core.solvers import get_solver
    import idaes.core.util.scaling as iscale

    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.h2_prop_params = GenericParameterBlock(**get_prop(FEED_COMP,  {"Vap"}, eos=EosType.IDEAL))
    m.fs.o2_prop_params = GenericParameterBlock(**get_prop(SWEEP_COMP, {"Vap"}, eos=EosType.IDEAL))
    m.fs.soec = SoecDesign(
        hydrogen_side_property_package=m.fs.h2_prop_params,
        oxygen_side_property_package=m.fs.o2_prop_params,
        reaction_eos=EosType.IDEAL,
        has_heat_transfer=False,
    )
    soec = m.fs.soec
    soec.hydrogen_side_inlet.flow_mol[0].fix(2.0)
    soec.hydrogen_side_inlet.temperature[0].fix(T_NOM)
    soec.hydrogen_side_inlet.pressure[0].fix(20e5)
    for c, v in FEED_COMP.items():
        soec.hydrogen_side_inlet.mole_frac_comp[:, c].fix(v)
    soec.oxygen_side_inlet.flow_mol[0].fix(2.0)
    soec.oxygen_side_inlet.temperature[0].fix(T_NOM)
    soec.oxygen_side_inlet.pressure[0].fix(20e5)
    for c, v in SWEEP_COMP.items():
        soec.oxygen_side_inlet.mole_frac_comp[:, c].fix(v)
    # outlet temperatures computed by IDAES (not fixed)
    soec.water_utilization[0].fix(UTIL_NOM)
    # Fix DOF=2: adiabatic heat=0, current determined by flowsheet
    # Fix current to nominal operating point (Faraday: I = F * n_H2 * util)
    soec.current[0].fix(96485 * 2.0 * 0.99 * 0.70 * 2)  # ~267kA nominal
    soec.sweep_heater.control_volume.heat[0].fix(0.0)
    iscale.calculate_scaling_factors(m)
    logger.info("Initialising IDAES SOEC flowsheet ...")
    from idaes.core.util.initialization import propagate_state
    soec.h2_inlet_translator.initialize()
    propagate_state(soec.strm1)
    soec.electrolysis_reactor.control_volume.properties_in[0].mole_frac_comp["O2"].set_value(1e-4)
    soec.electrolysis_reactor.control_volume.properties_in[0].mole_frac_comp["H2O"].set_value(0.9989)
    soec.electrolysis_reactor.control_volume.properties_in[0].mole_frac_comp["H2"].set_value(0.0001)
    soec.electrolysis_reactor.initialize()
    propagate_state(soec.strm2)
    soec.o2_separator.initialize()
    propagate_state(soec.strm3)
    soec.h2_outlet_translator.initialize()
    propagate_state(soec.strm4)
    soec.o2_translator.initialize()
    propagate_state(soec.strm5)
    soec.o2_mixer.initialize()
    propagate_state(soec.strm6)
    soec.sweep_heater.initialize()
    solver = get_solver("ipopt", solver_options={"max_iter": 3000, "tol": 1e-4, "print_level": 0})
    solver.solve(m, tee=True)
    logger.info("IDAES SOEC initialised OK")
    solver = get_solver("ipopt", solver_options={"max_iter": 3000, "tol": 1e-4, "print_level": 0})
    return m, solver

class SOECEnvIDEAS(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, re_profile=None, re_forecast=None,
                 w_load=DEFAULT_W_LOAD, w_safe=DEFAULT_W_SAFE, w_stab=DEFAULT_W_STAB, seed=42):
        super().__init__()
        self.w_load=w_load; self.w_safe=w_safe; self.w_stab=w_stab
        self._rng=np.random.default_rng(seed)
        self._re_profile_fixed=re_profile; self._re_forecast_fixed=re_forecast
        logger.info("Building IDAES SOEC flowsheet ...")
        self._m, self._solver = build_soec_flowsheet()
        self._soec = self._m.fs.soec
        self.observation_space=spaces.Box(
            low=np.array([V_LOW,-T_DEV_MAX,UTIL_MIN,0.,0.],dtype=np.float32),
            high=np.array([V_HIGH,T_DEV_MAX,UTIL_MAX,1.,1.],dtype=np.float32))
        self.action_space=spaces.Box(
            low=np.array([-0.05],dtype=np.float32),
            high=np.array([0.05],dtype=np.float32))
        self._util=UTIL_NOM; self._v_cell=V_NOM; self._dT=0.
        self._prev_action=0.; self._step_count=0
        self._re_profile=np.array([]); self._re_forecast=np.array([])

    def _make_re_profile(self):
        t=np.linspace(0,4*np.pi,EPISODE_LEN)
        noise=self._rng.normal(0,0.05,EPISODE_LEN)
        re=np.clip(0.5+0.35*np.sin(t)+noise,0.,1.).astype(np.float32)
        fc=np.roll(re,-1); fc[-1]=fc[-2]
        return re,fc

    def _solve_soec(self, util):
        import pyomo.environ as pyo
        util_c=float(np.clip(util,UTIL_MIN,UTIL_MAX))
        self._soec.water_utilization[0].fix(util_c)
        try:
            result=self._solver.solve(self._m, tee=False)
            if pyo.check_optimal_termination(result):
                v=float(pyo.value(self._soec.cell_potential[0]))
                T=float(pyo.value(self._soec.hydrogen_side_outlet_temperature[0]))
                dT=float(np.clip(T-T_NOM,-T_DEV_MAX,T_DEV_MAX))
                return v,dT,util_c
            return self._v_cell,self._dT,self._util
        except:
            return self._v_cell,self._dT,self._util

    def _get_obs(self):
        t=min(self._step_count,len(self._re_profile)-1)
        return np.array([self._v_cell,self._dT,self._util,
                         float(self._re_profile[t]),float(self._re_forecast[t])],dtype=np.float32)

    def reset(self,*,seed=None,options=None):
        if seed is not None: self._rng=np.random.default_rng(seed)
        if self._re_profile_fixed is not None:
            self._re_profile=self._re_profile_fixed
            self._re_forecast=self._re_forecast_fixed if self._re_forecast_fixed is not None else np.roll(self._re_profile,-1)
        else:
            self._re_profile,self._re_forecast=self._make_re_profile()
        init_util=float(self._rng.uniform(0.60,0.80))
        self._v_cell,self._dT,self._util=self._solve_soec(init_util)
        self._prev_action=0.; self._step_count=0
        return self._get_obs(),{}

    def step(self,action):
        a=float(np.clip(action[0],-0.05,0.05))
        self._v_cell,self._dT,self._util=self._solve_soec(self._util+a)
        t=min(self._step_count,len(self._re_profile)-1)
        re_now=float(self._re_profile[t])
        r_load=-abs(re_now-self._util)
        sv=not(V_LOW<=self._v_cell<=V_HIGH)
        r_safe=-1. if sv else 0.
        r_stab=-abs(a-self._prev_action)
        reward=self.w_load*r_load+self.w_safe*r_safe+self.w_stab*r_stab
        self._prev_action=a; self._step_count+=1
        info={"load_error":abs(re_now-self._util),"safety_violation":sv,
              "v_cell":self._v_cell,"dT":self._dT,"util":self._util,
              "RE_now":re_now,"r_load":r_load,"r_safe":r_safe,"r_stab":r_stab,
              "wind_power_norm":re_now}
        return self._get_obs(),float(reward),False,self._step_count>=EPISODE_LEN,info

    def render(self):
        print(f"step={self._step_count:4d}  v={self._v_cell:.3f}V  dT={self._dT:+.1f}K  util={self._util:.2f}")

    def close(self): pass
