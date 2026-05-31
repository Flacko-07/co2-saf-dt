"""
process_chgnet_descriptors.py
Pivot CHGNet raw energies into per‑catalyst descriptor table (E_CO, E_H, …).
"""
import pandas as pd
from config import PROC_DIR, DESCRIPTORS_CSV

raw = pd.read_csv(PROC_DIR / "chgnet_descriptors_raw.csv")
pivot = raw.pivot_table(index=["facet", "promoter"],
                        columns="adsorbate", values="E_total_eV").reset_index()
pivot.columns.name = None

# Compute adsorption energies relative to clean slab
for ads in ["CO", "H", "O", "OH"]:
    if ads in pivot.columns and "none" in pivot.columns:
        pivot[f"E_{ads}"] = pivot[ads] - pivot["none"]

# Keep only required columns
cols = ["facet", "promoter"] + [f"E_{ads}" for ads in ["CO","H","O","OH"] if f"E_{ads}" in pivot.columns]
pivot = pivot[cols]
pivot["facet"] = pivot["facet"].astype(str)

pivot.to_csv(DESCRIPTORS_CSV, index=False)
print(f"Catalyst descriptor table saved to {DESCRIPTORS_CSV}")
