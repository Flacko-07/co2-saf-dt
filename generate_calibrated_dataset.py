"""
generate_calibrated_dataset.py
Generate a CSV of plant runs over the catalyst and operating space.
"""
import itertools
import pandas as pd
from tqdm import tqdm
from config import PROC_DIR
from plant_calibrated import CO2_to_SAF_Plant

# –––––– Design space ––––––
FE_FACETS = ["110", "100", "111"]  # Miller indices
PROMOTERS = ["none", "K", "Co", "Pt", "Pd", "Ru", "Rh", "Re"]
TEMPERATURES = list(range(473, 678, 10))  # K
PRESSURES = list(range(1, 41, 2))         # bar

# Conversion multipliers (same as in config)
PROMOTER_MULTIPLIER = {
    "none": 1.00, "K": 1.15, "Co": 1.08, "Pt": 1.10,
    "Pd": 1.12, "Ru": 1.18, "Rh": 1.20, "Re": 1.25,
}

def main():
    rows = []
    total = len(FE_FACETS) * len(PROMOTERS) * len(TEMPERATURES) * len(PRESSURES)
    pbar = tqdm(total=total, desc="Simulating")

    for facet, promoter, T, P in itertools.product(FE_FACETS, PROMOTERS, TEMPERATURES, PRESSURES):
        mult = PROMOTER_MULTIPLIER[promoter]
        plant = CO2_to_SAF_Plant(
            T_rwgs=673,
            P_bar=P,
            T_ft=T,
            catalyst_mass_g=1.0,
            ft_co_conversion=0.51,
            promoter_multiplier=mult,
        )
        # Feed composition – representative of a syngas‑like mixture
        fresh = {"CO2": 1.0, "H2": 3.0}   # molar flow ratios
        res = plant.run_simulation(fresh, simulation_hours=1.0)

        rows.append({
            "facet": facet,
            "promoter": promoter,
            "temperature_K": T,
            "pressure_bar": P,
            "SAF_kg_h_per_g": res["STY_mg_gcat_h"] / 1e6,  # convert to kg/h/g
            "SAF_selectivity": res["per_step_metrics"][-1]["selectivity"],
            "CO_conversion": res["CO_conversion_overall"],
        })
        pbar.update(1)

    df = pd.DataFrame(rows)
    out_path = PROC_DIR / "plant_calibrated_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    # Quick sanity check
    ref = df[(df["temperature_K"] == 603) & (df["pressure_bar"] == 20) & (df["promoter"] == "K")]
    if not ref.empty:
        sty_mg = ref.iloc[0]["SAF_kg_h_per_g"] * 1e6
        print(f"Reference point STY: {sty_mg:.2f} mg/g/h (expected 252.7)")

if __name__ == "__main__":
    main()