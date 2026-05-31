"""
extract_oc20_descriptors.py (fairchem edition)
Compute adsorption energies using fairchem's GemNet-OC predictor.
"""
import torch
import pandas as pd
from ase.io import read
from ase import Atoms
from fairchem.core.predictors import IS2REPredictor
from config import DATA_DIR, PROC_DIR

SLAB_DIR = DATA_DIR / "mp_structures"
CKPT_PATH = "checkpoints/gemnet-oc_is2re_all.pt"   # downloaded by download_fairchem_model.py

FACETS = ["110", "100", "111"]
PROMOTERS = ["none", "K", "Co", "Pt", "Pd", "Ru", "Rh", "Re"]
ADSORBATES = ["none", "CO", "H", "O", "OH"]

def load_slab(facet):
    return read(str(SLAB_DIR / f"Fe_{facet}.traj"))

def add_promoter(slab, promoter):
    if promoter == "none":
        return slab.copy()
    slab = slab.copy()
    z_max = slab.positions[:, 2].max()
    surf_idx = [i for i, z in enumerate(slab.positions[:, 2]) if abs(z - z_max) < 0.5]
    slab.symbols[surf_idx[0]] = promoter
    return slab

def add_adsorbate(slab, ads, height=2.0):
    if ads == "none":
        return slab.copy()
    slab = slab.copy()
    top_idx = slab.positions[:, 2].argmax()
    pos = slab.positions[top_idx] + [0, 0, height]
    if ads == "CO":
        slab += Atoms("CO", positions=[pos, pos + [0, 0, 1.15]])
    elif ads == "H":
        slab += Atoms("H", positions=[pos])
    elif ads == "O":
        slab += Atoms("O", positions=[pos])
    elif ads == "OH":
        slab += Atoms("OH", positions=[pos, pos + [0, 0, 0.98]])
    else:
        raise ValueError(ads)
    return slab

# ---- Predictor (uses GPU if available) ----
device = "cuda" if torch.cuda.is_available() else "cpu"
predictor = IS2REPredictor(checkpoint_path=CKPT_PATH, device=device)

def get_energy(atoms):
    """Return predicted relaxed total energy (eV)."""
    return predictor.predict(atoms)

if __name__ == "__main__":
    rows = []
    for facet in FACETS:
        clean = load_slab(facet)
        for promoter in PROMOTERS:
            promoted = add_promoter(clean, promoter)
            e_slab = get_energy(promoted)

            for ads in ADSORBATES:
                if ads == "none":
                    e_ads = e_slab
                else:
                    ads_slab = add_adsorbate(promoted, ads)
                    e_ads = get_energy(ads_slab)
                rows.append({
                    "facet": facet,
                    "promoter": promoter,
                    "adsorbate": ads,
                    "E_total_eV": e_ads,
                    "E_slab_eV": e_slab,
                })
            print(f"Done {facet} / {promoter}")

    df = pd.DataFrame(rows)
    df.to_csv(PROC_DIR / "oc20_descriptors.csv", index=False)
    print("Raw descriptors saved.")