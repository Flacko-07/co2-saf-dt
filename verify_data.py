"""
verify_data.py
══════════════
Run this FIRST to confirm everything on disk is wired up correctly.
It checks OC20 LMDBs, the checkpoint, and the adsorption CSVs,
then prints a clean status table so you know exactly what to fix.

Usage:
    python verify_data.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import importlib, subprocess

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_DIR, CKPT_DIR, OC20_LMDB_PATHS,
    OCP_PRETRAINED_URL, CATHUB_GRAPHQL
)

OK  = "✓"
ERR = "✗"
WARN = "⚠"


def _check(label: str, path: Path, min_bytes: int = 1024) -> bool:
    exists = path.exists()
    size   = path.stat().st_size if exists else 0
    ok     = exists and size >= min_bytes
    flag   = OK if ok else ERR
    mb     = size / 1e6
    print(f"  {flag}  {label:<40s}  {mb:>8.1f} MB  {path}")
    return ok


def check_oc20() -> int:
    print("\n── OC20 IS2RE LMDBs ──────────────────────────────────────────────")
    ok_count = 0
    for split, folder in OC20_LMDB_PATHS.items():
        lmdb = folder / "data.lmdb"
        if _check(split, lmdb, min_bytes=1024):
            ok_count += 1
    print(f"  {ok_count}/{len(OC20_LMDB_PATHS)} splits present.")
    return ok_count


def check_checkpoint() -> bool:
    print("\n── Pretrained Checkpoint ─────────────────────────────────────────")
    ckpt = CKPT_DIR / "gemnet_oc_is2re_all.pt"
    ok   = _check("GemNet-OC checkpoint", ckpt, min_bytes=1_000_000)
    if not ok:
        print(f"  → Download:  python download_oc20.py --pretrained")
    return ok


def check_adsorption_csvs() -> bool:
    print("\n── Adsorption Energy CSVs ────────────────────────────────────────")
    cathub_csv = DATA_DIR / "cathub_fe_adsorption.csv"
    lit_csv    = DATA_DIR / "literature_fe_adsorption.csv"
    has_cathub = _check("CatHub CSV",      cathub_csv, 100)
    has_lit    = _check("Literature CSV",  lit_csv,    100)
    if not has_cathub and not has_lit:
        print(f"  → Run:  python fetch_catalysis_hub.py --literature-only")
    return has_cathub or has_lit


def check_mp_structures() -> bool:
    print("\n── Materials Project Structures ──────────────────────────────────")
    mp_dir = DATA_DIR / "mp_structures"
    trajs  = list(mp_dir.glob("*.traj")) if mp_dir.exists() else []
    cifs   = list(mp_dir.glob("*.cif"))  if mp_dir.exists() else []
    print(f"  {'✓' if trajs else '✗'}  {len(trajs)} .traj files in {mp_dir}")
    print(f"  {'✓' if cifs  else '✗'}  {len(cifs)}  .cif  files in {mp_dir}")
    return bool(trajs or cifs)


def check_cathub_api() -> None:
    print("\n── Catalysis Hub API (live probe) ────────────────────────────────")
    try:
        import requests
        q = '{ reactions(first: 1, surfaceComposition: "Fe") { totalCount } }'
        r = requests.post(CATHUB_GRAPHQL, data={"query": q}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            tc   = data.get("data", {}).get("reactions", {}).get("totalCount", "?")
            print(f"  {OK}  CatHub API reachable  (totalCount for Fe = {tc})")
        else:
            print(f"  {ERR}  CatHub API returned HTTP {r.status_code}")
            print(f"  → Use --literature-only flag as fallback.")
    except Exception as exc:
        print(f"  {WARN}  CatHub API unreachable: {exc}")
        print(f"  → Use:  python fetch_catalysis_hub.py --literature-only")


def check_python_deps() -> None:
    print("\n── Python dependencies ───────────────────────────────────────────")
    required = {
        "torch":        "torch",
        "torch_geometric": "torch_geometric",
        "ase":          "ase",
        "lmdb":         "lmdb",
        "sklearn":      "scikit-learn",
        "pandas":       "pandas",
        "numpy":        "numpy",
        "scipy":        "scipy",
        "joblib":       "joblib",
        "tqdm":         "tqdm",
        "requests":     "requests",
    }
    for mod, pkg in required.items():
        try:
            importlib.import_module(mod)
            print(f"  {OK}  {pkg}")
        except ImportError:
            print(f"  {ERR}  {pkg}   →  pip install {pkg}")


def main() -> None:
    print("=" * 70)
    print("  CO2-to-SAF GNN pipeline — data verification")
    print("=" * 70)

    check_python_deps()
    oc20_ok  = check_oc20()
    ckpt_ok  = check_checkpoint()
    csv_ok   = check_adsorption_csvs()
    mp_ok    = check_mp_structures()
    check_cathub_api()

    print("\n── Summary ───────────────────────────────────────────────────────")
    print(f"  OC20 splits : {oc20_ok}/{len(OC20_LMDB_PATHS)}")
    print(f"  Checkpoint  : {'OK' if ckpt_ok else 'MISSING'}")
    print(f"  Adsorption  : {'OK' if csv_ok  else 'MISSING – run fetch_catalysis_hub.py'}")
    print(f"  MP structs  : {'OK' if mp_ok   else 'none found'}")

    if oc20_ok == 0:
        print("\n  ACTION: python download_oc20.py --download   (downloads ~8 GB bundle)")
    if not ckpt_ok:
        print("  ACTION: python download_oc20.py --pretrained")
    if not csv_ok:
        print("  ACTION: python fetch_catalysis_hub.py --literature-only")
    print()


if __name__ == "__main__":
    main()
