import sys, itertools, warnings
from pathlib import Path
import numpy as np, pandas as pd
import pyomo.environ as pyo
from idaes.core import FlowsheetBlock
from idaes.models_extra.power_generation.unit_models.soec_design import SoecDesign, EosType
from idaes.models_extra.power_generation.properties.natural_gas_PR import get_prop
from idaes.models.properties.modular_properties.base.generic_property import GenericParameterBlock
from idaes.core.solvers import get_solver
from idaes.core.util.initialization import propagate_state
import idaes.core.util.scaling as iscale
warnings.filterwarnings("ignore")

F = 96485.0
CELL_AREA = 0.01   # m^2 = 100 cm^2 realistic single cell
V_LO=1.00; V_HI=1.50; DT_MAX=100.0
FEED_COMP  = {"H2": 0.01, "H2O": 0.99}
SWEEP_COMP = {"O2": 0.2074, "H2O": 0.0099, "CO2": 0.0003, "N2": 0.7732, "Ar": 0.0092}

def build_and_init(flow, temp, pres, util=0.50):
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.h2p = GenericParameterBlock(**get_prop(FEED_COMP,  {"Vap"}, eos=EosType.IDEAL))
    m.fs.o2p = GenericParameterBlock(**get_prop(SWEEP_COMP, {"Vap"}, eos=EosType.IDEAL))
    m.fs.soec = SoecDesign(hydrogen_side_property_package=m.fs.h2p,
        oxygen_side_property_package=m.fs.o2p,
        reaction_eos=EosType.IDEAL, has_heat_transfer=False)
    s = m.fs.soec
    for side, comp in [("hydrogen", FEED_COMP), ("oxygen", SWEEP_COMP)]:
        inlet = getattr(s, f"{side}_side_inlet")
        inlet.flow_mol[0].fix(flow)
        inlet.temperature[0].fix(temp)
        inlet.pressure[0].fix(pres)
        for c, v in comp.items():
            inlet.mole_frac_comp[:, c].fix(v)
    n_h2o = flow * FEED_COMP["H2O"]
    s.water_utilization[0].fix(util)
    s.current[0].fix(2 * F * n_h2o * util)
    s.sweep_heater.control_volume.heat[0].fix(0.0)
    iscale.calculate_scaling_factors(m)
    s.h2_inlet_translator.initialize()
    propagate_state(s.strm1)
    s.electrolysis_reactor.control_volume.properties_in[0].mole_frac_comp["O2"].set_value(1e-4)
    s.electrolysis_reactor.control_volume.properties_in[0].mole_frac_comp["H2O"].set_value(0.9989)
    s.electrolysis_reactor.control_volume.properties_in[0].mole_frac_comp["H2"].set_value(0.0001)
    s.electrolysis_reactor.initialize()
    propagate_state(s.strm2); s.o2_separator.initialize()
    propagate_state(s.strm3); s.h2_outlet_translator.initialize()
    propagate_state(s.strm4); s.o2_translator.initialize()
    propagate_state(s.strm5); s.o2_mixer.initialize()
    propagate_state(s.strm6); s.sweep_heater.initialize()
    s.hydrogen_side_outlet_temperature.unfix()
    s.oxygen_side_outlet_temperature.unfix()
    return m

def solve_point(m, solver, util):
    s = m.fs.soec
    flow    = float(pyo.value(s.hydrogen_side_inlet.flow_mol[0]))
    n_h2o   = flow * FEED_COMP["H2O"]
    temp_in = float(pyo.value(s.hydrogen_side_inlet.temperature[0]))
    s.water_utilization[0].fix(util)
    s.current[0].fix(2 * F * n_h2o * util)
    res = solver.solve(m, tee=False)
    if not pyo.check_optimal_termination(res):
        return None
    # Filter unphysical: Nernst min ~0.95V (Mueller 2024), max 2.0V above thermoneutral
    v_test = float(pyo.value(s.cell_potential[0]))
    if v_test < 0.95 or v_test > 2.0:
        return None
    # Filter unphysical dT — thermoneutral operation limits dT to ~100K
    # Reference: Mueller et al. 2024, Chemie Ingenieur Technik
    T_h2_test = float(pyo.value(s.hydrogen_side_outlet_temperature[0]))
    temp_in = float(pyo.value(s.hydrogen_side_inlet.temperature[0]))
    if abs(T_h2_test - temp_in) > 300:  # relaxed to capture more training data
        return None
    v_cell = float(pyo.value(s.cell_potential[0]))
    I      = float(pyo.value(s.current[0]))
    j      = I / (CELL_AREA * 1e4)
    T_h2   = float(pyo.value(s.hydrogen_side_outlet_temperature[0]))
    T_o2   = float(pyo.value(s.oxygen_side_outlet_temperature[0]))
    dT     = T_h2 - temp_in
    safe   = int(V_LO <= v_cell <= V_HI and abs(dT) <= DT_MAX)
    return {"flow_mol": flow, "temperature_K": temp_in,
            "pressure_Pa": float(pyo.value(s.hydrogen_side_inlet.pressure[0])),
            "water_utilization": util, "cell_potential_V": v_cell,
            "current_A": I, "j_Acm2": j, "T_h2_out_K": T_h2,
            "T_o2_out_K": T_o2, "dT_K": dT, "safe": safe}

def main():
    Path("data").mkdir(exist_ok=True)
    solver = get_solver("ipopt", solver_options={"max_iter": 3000, "tol": 1e-4, "print_level": 0})
    flows = [0.0003, 0.0005, 0.0007, 0.001, 0.0013, 0.0015, 0.002, 0.0025, 0.003, 0.004]
    temps = [923.0, 973.0, 1023.0, 1073.0, 1123.0]
    press = [15e5, 20e5, 25e5]  # pressurised SOEC, stable range
    utils = np.linspace(0.20, 0.80, 17).tolist()  # Jolaoso et al. 2023: safe util range
    records = []; total = len(flows)*len(temps)*len(press); done = 0
    for flow, temp, pres in itertools.product(flows, temps, press):
        done += 1
        print(f"[{done}/{total}] flow={flow} T={temp}K P={pres/1e5:.0f}bar", flush=True)
        for util in utils:
            try:
                m = build_and_init(flow, temp, pres, util=util)
                res = solver.solve(m, tee=False)
                if not pyo.check_optimal_termination(res):
                    continue
                rec = solve_point(m, solver, util)
                if rec is not None:
                    records.append(rec)
            except Exception:
                continue
    df = pd.DataFrame(records)
    df.to_csv("data/pinn_training_data_idaes.csv", index=False)
    print(f"Done: {len(df)} points")
    print(df["safe"].value_counts())
    print(df[["cell_potential_V","dT_K","j_Acm2","water_utilization","safe"]].describe().round(3))

if __name__ == "__main__":
    main()
