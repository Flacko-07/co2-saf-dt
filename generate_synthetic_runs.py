"""
generate_synthetic_runs.py (descriptor‑driven, relative volcano)
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from plant_calibrated import CO2_to_SAF_Plant
from config import PROC_DIR, DESCRIPTORS_CSV

desc = pd.read_csv(DESCRIPTORS_CSV)
desc["facet"] = desc["facet"].astype(str)

# Load reference catalyst (same as used in calibration)
ref_row = desc[(desc["facet"] == "110") & (desc["promoter"] == "K")].iloc[0]
REF_E_CO = ref_row["E_CO"]
REF_E_H  = ref_row["E_H"]

# Use the calibrated value from plant_calibrated.py
REF_CO_CONSUMPTION = 0.04897  # <-- replace with the printed value

T_grid = np.arange(473, 678, 5)
P_grid = np.arange(1, 41, 2)

rows = []
for _, cat in tqdm(desc.iterrows(), total=len(desc), desc="Catalysts"):
    for T in T_grid:
        for P in P_grid:
            plant = CO2_to_SAF_Plant(
                T_rwgs=673, P_bar=P, T_ft=T, catalyst_mass_g=1.0,
                ref_E_CO=REF_E_CO, ref_E_H=REF_E_H,
                E_CO=cat["E_CO"], E_H=cat["E_H"], E_O=cat["E_O"], E_OH=cat["E_OH"],
                ref_co_consumption=REF_CO_CONSUMPTION,
            )
            fresh = {"CO2": 0.1191, "H2": 3 * 0.1191}   # same feed rate as during calibration
            res = plant.run_simulation(fresh)
            rows.append({
                "facet": cat["facet"],
                "promoter": cat["promoter"],
                "temperature_K": T,
                "pressure_bar": P,
                "STY_mg_gcat_h": res["STY_mg_gcat_h"],
                "SAF_selectivity": res["SAF_selectivity"],
            })

df = pd.DataFrame(rows)
out_path = PROC_DIR / "synthetic_plant_runs.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} synthetic runs to {out_path}")