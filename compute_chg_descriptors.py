"""
compute_chgnet_descriptors.py
Use CHGNet to compute adsorption energies for all Fe catalyst variants.
"""
import torch
import pandas as pd
from ase.io import read
from ase import Atoms
from ase.optimize import BFGS
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator

from config import DATA_DIR, PROC_DIR

SLAB_DIR = DATA_DIR / "mp_structures"
FACETS = ["110", "100", "111"]
PROMOTERS = ["none", "K", "Co", "Pt", "Pd", "Ru", "Rh", "Re"]
ADSORBATES = ["none", "CO", "H", "O", "OH"]

# Load CHGNet (pretrained)
model = CHGNet.load()
calc = CHGNetCalculator(model, device="cuda" if torch.cuda.is_available() else "cpu")

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

def get_energy(atoms, relax=True):
    atoms = atoms.copy()
    atoms.calc = calc
    if relax:
        opt = BFGS(atoms)
        opt.run(fmax=0.02, steps=50)
    return atoms.get_potential_energy()

if __name__ == "__main__":
    rows = []
    for facet in FACETS:
        clean = load_slab(facet)
        for promoter in PROMOTERS:
            promoted = add_promoter(clean, promoter)
            e_slab = get_energy(promoted, relax=True)

            for ads in ADSORBATES:
                if ads == "none":
                    e_ads_slab = e_slab
                else:
                    ads_slab = add_adsorbate(promoted, ads)
                    e_ads_slab = get_energy(ads_slab, relax=True)
                rows.append({
                    "facet": facet,
                    "promoter": promoter,
                    "adsorbate": ads,
                    "E_total_eV": e_ads_slab,
                    "E_slab_eV": e_slab,
                })
            print(f"Done {facet} / {promoter}")

    df = pd.DataFrame(rows)
    df.to_csv(PROC_DIR / "mace_descriptors_raw.csv", index=False)  # same filename for compatibility
    print("Descriptors saved to data/processed/mace_descriptors_raw.csv")