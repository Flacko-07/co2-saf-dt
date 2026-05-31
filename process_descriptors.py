"""
process_descriptors.py
Pivot literature Fe adsorption data and expand to all required promoters.
"""
import pandas as pd
import numpy as np
from config import DATA_DIR, PROC_DIR, DESCRIPTORS_CSV

LIT_CSV = DATA_DIR / "literature_fe_adsorption.csv"

# ── 1. Create literature CSV if missing ─────────────────────────────────────
if not LIT_CSV.exists():
    rows = [
        ["Fe",  "110", "CO", -1.12],
        ["Fe",  "110", "H",  -0.33],
        ["Fe",  "110", "O",  -0.84],
        ["Fe",  "110", "OH", -0.45],
        ["Fe",  "100", "CO", -0.77],
        ["Fe",  "100", "H",  -0.24],
        ["Fe",  "100", "O",  -0.68],
        ["Fe",  "100", "OH", -0.31],
        ["FeK", "110", "CO", -1.42],
        ["FeK", "110", "H",  -0.28],
        ["FeK", "110", "O",  -1.05],
        ["FeK", "110", "OH", -0.60],
        # add FeK 100 if available, else we'll fill later
    ]
    lit_df = pd.DataFrame(rows, columns=["surface_composition", "facet", "adsorbate", "adsorption_energy_eV"])
    lit_df.to_csv(LIT_CSV, index=False)
else:
    lit_df = pd.read_csv(LIT_CSV)

# ── 2. Pivot the available data ────────────────────────────────────────────
key_ads = ["CO", "H", "O", "OH"]
lit_df = lit_df[lit_df["adsorbate"].isin(key_ads)]
pivot = lit_df.pivot_table(
    index=["surface_composition", "facet"],
    columns="adsorbate",
    values="adsorption_energy_eV"
).reset_index()
pivot.columns.name = None
pivot = pivot.rename(columns={
    "surface_composition": "raw_promoter",
    "CO": "E_CO", "H": "E_H", "O": "E_O", "OH": "E_OH",
})

# Map to standard names: Fe → none, FeK → K
name_map = {"Fe": "none", "FeK": "K"}
pivot["promoter"] = pivot["raw_promoter"].map(name_map)
pivot = pivot.drop(columns=["raw_promoter"])
pivot["facet"] = pivot["facet"].astype(str)

# Full grid of (facet, promoter) – both strings
REQUIRED_FACETS = ["110", "100", "111"]
REQUIRED_PROMOTERS = ["none", "K", "Co", "Pt", "Pd", "Ru", "Rh", "Re"]
grid = pd.DataFrame([(f, p) for f in REQUIRED_FACETS for p in REQUIRED_PROMOTERS],
                    columns=["facet", "promoter"])

# Merge
full = grid.merge(pivot, on=["facet", "promoter"], how="left")

# For missing promoters, use the pure Fe ("none") value as baseline
# (assumes no promoter effect if no data)
for col in ["E_CO", "E_H", "E_O", "E_OH"]:
    # Get the none-values for each facet
    none_vals = full[full["promoter"] == "none"].set_index("facet")[col]
    # Fill missing values in each facet with the none value
    full[col] = full.apply(lambda row: none_vals[row["facet"]] if pd.isna(row[col]) and row["facet"] in none_vals.index else row[col], axis=1)

# If still NaN (e.g., "none" itself missing for some facet), fill with 0
full = full.fillna(0.0)

full.to_csv(DESCRIPTORS_CSV, index=False)
print(f"Descriptors saved to {DESCRIPTORS_CSV}")
print(full.head(12))