"""
generate_synthetic_runs.py (variable selectivity version)
"""
import itertools
import pandas as pd
from tqdm import tqdm
from plant_calibrated import CO2_to_SAF_Plant
from config import PROC_DIR

CALIBRATED_CO2_FEED = 0.128585  # <-- copy the exact value from plant_calibrated.py output

facets = ["110", "100", "111"]
promoters = ["none", "K", "Co", "Pt", "Pd", "Ru", "Rh", "Re"]
T_grid = range(473, 678, 5)
P_grid = range(1, 41, 2)

PROMOTER_MULTIPLIER = {
    "none": 1.00, "K": 1.15, "Co": 1.08, "Pt": 1.10,
    "Pd": 1.12, "Ru": 1.18, "Rh": 1.20, "Re": 1.25,
}

rows = []
total = len(facets) * len(promoters) * len(T_grid) * len(P_grid)
pbar = tqdm(total=total, desc="Simulating")

for f, p, T, P in itertools.product(facets, promoters, T_grid, P_grid):
    mult = PROMOTER_MULTIPLIER[p]
    plant = CO2_to_SAF_Plant(
        T_rwgs=673, P_bar=P, T_ft=T, catalyst_mass_g=1.0,
        ft_co_conversion=0.51, promoter_multiplier=mult,
    )
    fresh = {"CO2": CALIBRATED_CO2_FEED, "H2": 3 * CALIBRATED_CO2_FEED}
    res = plant.run_simulation(fresh, simulation_hours=1.0)

    rows.append({
        "facet": f,
        "promoter": p,
        "temperature_K": T,
        "pressure_bar": P,
        "STY_mg_gcat_h": res["STY_mg_gcat_h"],
        "SAF_selectivity": res["SAF_selectivity"],
    })
    pbar.update(1)

pbar.close()
df = pd.DataFrame(rows)
out_path = PROC_DIR / "synthetic_plant_runs.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} synthetic runs to {out_path}")