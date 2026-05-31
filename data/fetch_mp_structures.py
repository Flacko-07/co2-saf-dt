"""
data/fetch_mp_structures.py
Download bcc Fe bulk and surface slabs from Materials Project.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pymatgen.ext.matproj import MPRester
from pymatgen.core.surface import SlabGenerator
from ase.io import write
from config import MP_API_KEY, DATA_DIR

MP_DIR = DATA_DIR / "mp_structures"
MP_DIR.mkdir(parents=True, exist_ok=True)

FACETS = {"110": (1,1,0), "100": (1,0,0), "111": (1,1,1)}

def fetch_fe_bulk(api_key):
    with MPRester(api_key) as mpr:
        structure = mpr.get_structure_by_material_id("mp-13")
    return structure

def generate_slab(bulk, miller, layers=4, vacuum=15.0, supercell=(2,2)):
    slabs = SlabGenerator(bulk, miller, min_slab_size=layers,
                          min_vacuum_size=vacuum, center_slab=True,
                          in_unit_planes=True).get_slabs()
    slab = slabs[0]
    slab.make_supercell([[supercell[0],0,0],[0,supercell[1],0],[0,0,1]])
    return slab

if __name__ == "__main__":
    if not MP_API_KEY or MP_API_KEY == "YOUR_MP_API_KEY_HERE":
        raise RuntimeError("Set MP_API_KEY in config.py first!")
    bulk = fetch_fe_bulk(MP_API_KEY)
    bulk.to(filename=str(MP_DIR / "Fe_bulk.cif"))
    print("Fetched Fe bulk (mp-13)")

    for name, miller in FACETS.items():
        slab = generate_slab(bulk, miller)
        write(str(MP_DIR / f"Fe_{name}.traj"), slab.to_ase_atoms())
        print(f"Saved Fe({name}) slab")
