import numpy as np, pandas as pd, pickle, gymnasium as gym, os, time, wandb, argparse
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from pyomo.environ import ConcreteModel, value, TransformationFactory
from idaes.core import FlowsheetBlock
from idaes.models_extra.power_generation.unit_models.soec_design import SoecDesign
from idaes.models_extra.power_generation.properties.natural_gas_PR import get_prop, EosType
from idaes.models.properties.modular_properties.base.generic_property import GenericParameterBlock
from idaes.core.solvers import get_solver

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
SEED = args.seed

# ── W&B ──────────────────────────────────────────────────────────────
wandb.init(
    project="mapirl-soec-rl",
    name=f"re-sac-seed{SEED}-{time.strftime('%Y%m%d-%H%M%S')}",
    config={
        "algorithm": "SAC-RE",
        "total_timesteps": 10000,
        "learning_rate": 3e-4,
        "seed": SEED,
        "re_input": True,
        "load_following": True,
    }
)

print("=== MAPIRL-DT SAC with RE Load Following ===")
print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# ── Load UK RE data ───────────────────────────────────────────────────
re_data = pd.read_csv("uk_battery_dispatch_complete_data.csv")
re_data['datetime'] = pd.to_datetime(re_data['datetime'])
wind = re_data['EMBEDDED_WIND_GENERATION'].values.astype(np.float32)
solar = re_data['EMBEDDED_SOLAR_GENERATION'].values.astype(np.float32)
re_power = (wind + solar)
re_min, re_max = re_power.min(), re_power.max()
re_normalised = (re_power - re_min) / (re_max - re_min)
print(f"RE data loaded: {len(re_normalised)} timesteps ✅")

# ── Build Digital Twin ────────────────────────────────────────────────
m = ConcreteModel()
m.fs = FlowsheetBlock(dynamic=False)
m.fs.prop_h2 = GenericParameterBlock(**get_prop({"H2O","H2"}, phases={"Vap"}, eos=EosType.IDEAL))
m.fs.prop_o2 = GenericParameterBlock(**get_prop({"O2","H2O"}, phases={"Vap"}, eos=EosType.IDEAL))
m.fs.soec = SoecDesign(oxygen_side_property_package=m.fs.prop_o2, hydrogen_side_property_package=m.fs.prop_h2, has_heat_transfer=True)
TransformationFactory("network.expand_arcs").apply_to(m)
m.fs.soec.hydrogen_side_inlet.flow_mol[0].fix(0.10)
m.fs.soec.hydrogen_side_inlet.mole_frac_comp[0,"H2O"].fix(0.90)
m.fs.soec.hydrogen_side_inlet.mole_frac_comp[0,"H2"].fix(0.10)
m.fs.soec.hydrogen_side_inlet.temperature[0].fix(1073.15)
m.fs.soec.hydrogen_side_inlet.pressure[0].fix(1.5e5)
m.fs.soec.oxygen_side_inlet.flow_mol[0].fix(0.10)
m.fs.soec.oxygen_side_inlet.mole_frac_comp[0,"O2"].fix(0.21)
m.fs.soec.oxygen_side_inlet.mole_frac_comp[0,"H2O"].fix(0.79)
m.fs.soec.oxygen_side_inlet.temperature[0].fix(1073.15)
m.fs.soec.oxygen_side_inlet.pressure[0].fix(1.5e5)
m.fs.soec.water_utilization[0].fix(0.80)
m.fs.soec.heat[0].fix(0.0)
m.fs.soec.cell_potential[0].fix(1.28)
m.fs.soec.sweep_heater.outlet.temperature[0].unfix()
m.fs.soec.sweep_heater.control_volume.heat[0].fix(0.0)
solver = get_solver()
m.fs.soec.initialize(outlvl=0)
result = solver.solve(m, tee=False)
print(f"DT: {result.solver.termination_condition}")

# ── Load PINN ─────────────────────────────────────────────────────────
with open("pinn_boundary_classifier.pkl","rb") as f:
    d = pickle.load(f)
pinn, pinn_scaler = d["model"], d["scaler"]
print("PINN loaded ✅")

# ── RE-coupled SOEC Environment ───────────────────────────────────────
class RESOECEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.flow_min, self.flow_max = 0.05, 0.50
        self.current_flow = 0.10
        self.step_count = 0
        self.max_steps = 100
        self.re_idx = 0

        # Extended state: [V_cell, dT, flow_norm, RE_now, RE_next]
        self.action_space = gym.spaces.Box(
            low=np.array([-0.05]), high=np.array([0.05]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.array([0.9, -700.0, 0.0, 0.0, 0.0]),
            high=np.array([1.4,  100.0, 1.0, 1.0, 1.0]),
            dtype=np.float32)

    def _get_re(self):
        re_now  = re_normalised[self.re_idx % len(re_normalised)]
        re_next = re_normalised[(self.re_idx + 1) % len(re_normalised)]
        return float(re_now), float(re_next)

    def _obs(self):
        v    = value(m.fs.soec.cell_potential[0])
        dT   = value(m.fs.soec.hydrogen_side_outlet_temperature[0]) - \
               value(m.fs.soec.oxygen_side_outlet_temperature[0])
        fn   = (self.current_flow - self.flow_min) / (self.flow_max - self.flow_min)
        re_now, re_next = self._get_re()
        return np.array([v, dT, fn, re_now, re_next], dtype=np.float32)

    def step(self, action):
        prev_flow = self.current_flow
        re_now, re_next = self._get_re()

        # RE target: scale flow to match RE power
        re_target_flow = self.flow_min + re_now * (self.flow_max - self.flow_min)

        # Apply agent action
        self.current_flow = float(np.clip(
            self.current_flow + float(np.clip(action[0], -0.05, 0.05)),
            self.flow_min, self.flow_max))

        m.fs.soec.hydrogen_side_inlet.flow_mol[0].fix(self.current_flow)
        m.fs.soec.oxygen_side_inlet.flow_mol[0].fix(self.current_flow)
        try: m.fs.soec.initialize(outlvl=0)
        except: pass
        solver.solve(m, tee=False)

        obs = self._obs()
        v, dT = obs[0], obs[1]

        # ── Multi-objective reward ─────────────────────────────────
        # 1. Load following — primary objective
        load_error = abs(self.current_flow - re_target_flow)
        r_load = 1.0 - (load_error / (self.flow_max - self.flow_min))

        # 2. Safety — hard constraints
        r_safety = 0.0
        if not (0.9 <= v <= 1.3):
            r_safety -= 10.0 * abs(v - 1.1) / 0.2
        if abs(dT) > 20:
            r_safety -= 50.0 * (abs(dT) - 20.0) / 20.0

        # 3. Stability — penalise rapid changes
        r_stable = -2.0 * abs(self.current_flow - prev_flow) / 0.05

        reward = r_load + r_safety + r_stable

        self.step_count += 1
        self.re_idx += 1

        X = pinn_scaler.transform([[v, dT]])
        safe = bool(pinn.predict(X)[0])

        return obs, reward, False, self.step_count >= self.max_steps, {
            "pinn_safe": safe, "v_cell": v, "dT": dT,
            "flow": self.current_flow, "re_target": re_target_flow,
            "load_error": load_error, "r_load": r_load
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_flow = 0.10
        self.step_count = 0
        self.re_idx = np.random.randint(0, len(re_normalised))
        m.fs.soec.hydrogen_side_inlet.flow_mol[0].fix(0.10)
        m.fs.soec.oxygen_side_inlet.flow_mol[0].fix(0.10)
        try: m.fs.soec.initialize(outlvl=0)
        except: pass
        solver.solve(m, tee=False)
        return self._obs(), {}

env = RESOECEnv()
print("RE Environment ready ✅")

# ── Logger ────────────────────────────────────────────────────────────
class Logger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.safe = 0
        self.total = 0

    def _on_step(self):
        info = self.locals["infos"][0]
        self.total += 1
        if info.get("pinn_safe"): self.safe += 1
        wandb.log({
            "step": self.total,
            "safety_rate": self.safe / self.total * 100,
            "v_cell": info["v_cell"],
            "dT": info["dT"],
            "flow": info["flow"],
            "re_target": info["re_target"],
            "load_error": info["load_error"],
            "r_load": info["r_load"],
            "reward": self.locals["rewards"][0],
        })
        if self.total % 500 == 0:
            print(f"  Step {self.total}: safety={self.safe/self.total*100:.1f}% "
                  f"load_error={info['load_error']:.3f} V={info['v_cell']:.4f}", flush=True)
            self.model.save(f"re_sac_checkpoint_{self.total}")
        return True

# ── Train ─────────────────────────────────────────────────────────────
model = SAC("MlpPolicy", env, learning_rate=3e-4, batch_size=64,
            buffer_size=5000, learning_starts=100, verbose=0, seed=SEED)
cb = Logger()
print("Training RE-SAC for 10000 steps...")
model.learn(total_timesteps=10000, callback=cb)
model.save("re_sac_soec_final")
print(f"Done! Safety rate: {cb.safe/cb.total*100:.1f}%")
print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
wandb.finish()
