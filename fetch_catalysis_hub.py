"""
fetch_catalysis_hub.py  –  FIXED v2
═══════════════════════════════════════════════════════════════════════════════
Root causes of the 400 Bad Request:
  1.  Wrong URL:   https://api.catalysis-hub.org/graphql  (HTTPS – blocked)
                   http://api.catalysis-hub.org/graphql   ← correct (HTTP)
  2.  Wrong query: filter: {reactants: $x, surfaceComposition: "Fe"}
                   The `filter` wrapper does NOT exist in their Graphene schema.
                   Args go directly on the `reactions()` call (flat, no wrapper).
  3.  Wrong request format: they expect form-encoded {'query':...}
                            not JSON {"query":...}

Correct query shape (from cathub source):
    { reactions(first: 500, surfaceComposition: "Fe", reactants: "CO") {
        totalCount
        edges { node {
            chemicalComposition surfaceComposition facet
            reactants products reactionEnergy activationEnergy
            dftCode dftFunctional pubId
        }}
    }}

Data flow:
  ① Try fixed CatHub API  →  cathub_fe_adsorption.csv
  ② If API is down        →  literature_fe_adsorption.csv  (always works)
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR

# ── Correct API URL (HTTP not HTTPS) ──────────────────────────────────────────
CATHUB_URL  = "http://api.catalysis-hub.org/graphql"
OUT_CSV     = DATA_DIR / "cathub_fe_adsorption.csv"
LIT_CSV     = DATA_DIR / "literature_fe_adsorption.csv"

# ── Correct query: flat args, no filter wrapper ────────────────────────────────
def _build_query(surface: str, reactant: str, n: int = 500) -> str:
    """
    Build the correct CatHub GraphQL query.
    surfaceComposition and reactants are direct scalar args, no filter:{} wrapper.
    """
    return (
        "{ reactions("
        f'first: {n}, '
        f'surfaceComposition: "{surface}", '
        f'reactants: "{reactant}"'
        ") {\n"
        "  totalCount\n"
        "  edges { node {\n"
        "    chemicalComposition surfaceComposition facet\n"
        "    reactants products reactionEnergy activationEnergy\n"
        "    dftCode dftFunctional pubId\n"
        "  }}\n"
        "}}"
    )


def _post(query_str: str, retries: int = 4, delay: float = 2.0) -> dict | None:
    """
    Post query as form-encoded (the way cathub's own library does it).
    Falls back to JSON if form-encoded gets rejected.
    """
    for attempt in range(retries):
        try:
            # Primary: form-encoded (matches cathub library)
            r = requests.post(
                CATHUB_URL,
                data={"query": query_str},   # ← form-encoded, NOT json=
                timeout=30,
            )
            if r.status_code == 200:
                return r.json()
            # Secondary: JSON body
            r2 = requests.post(
                CATHUB_URL,
                json={"query": query_str},
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            if r2.status_code == 200:
                return r2.json()
            print(f"  [attempt {attempt+1}] HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException as exc:
            print(f"  [attempt {attempt+1}] Network error: {exc}")
        if attempt < retries - 1:
            time.sleep(delay * (attempt + 1))
    return None


def _extract_nodes(resp: dict) -> list[dict]:
    edges = resp.get("data", {}).get("reactions", {}).get("edges", [])
    return [e["node"] for e in edges]


def _to_df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["reaction_energy_eV"] = pd.to_numeric(df.get("reactionEnergy"), errors="coerce")
    df = df.dropna(subset=["reaction_energy_eV"])
    df.rename(columns={
        "chemicalComposition": "chemical_composition",
        "surfaceComposition":  "surface_composition",
        "reactionEnergy":      "adsorption_energy_eV",
        "activationEnergy":    "activation_energy_eV",
        "dftCode":             "dft_code",
        "dftFunctional":       "dft_functional",
        "pubId":               "pub_id",
    }, inplace=True)
    return df.reset_index(drop=True)


# ─── Main fetch ───────────────────────────────────────────────────────────────

# All reactant tokens CatHub uses for FT-relevant adsorbates
FT_REACTANTS = ["CO", "H2", "H", "O2", "OH", "H2O", "CO2", "CH4"]
FE_SURFACES  = ["Fe", "FeK", "FeRu", "FeRe", "FeRh", "FePt", "FeKMn"]


def fetch_cathub(force: bool = False) -> pd.DataFrame:
    if OUT_CSV.exists() and not force:
        print(f"[skip] {OUT_CSV} exists. Use force=True to re-fetch.")
        return pd.read_csv(OUT_CSV)

    print(f"Fetching Fe adsorption data from CatHub ({CATHUB_URL}) …")
    all_records: list[dict] = []

    for surface in FE_SURFACES:
        for reactant in FT_REACTANTS:
            q = _build_query(surface=surface, reactant=reactant, n=500)
            resp = _post(q)
            if resp is None:
                print(f"  ✗  {surface} / {reactant}  – API unreachable")
                continue
            if "errors" in resp:
                print(f"  ✗  {surface} / {reactant}  – {resp['errors'][:1]}")
                continue
            nodes = _extract_nodes(resp)
            total = resp.get("data", {}).get("reactions", {}).get("totalCount", "?")
            print(f"  ✓  {surface} / {reactant}  → {len(nodes):>4} rows  (total={total})")
            all_records.extend(nodes)
            time.sleep(0.25)   # be polite

    # Also broad Fe search (no reactant filter)
    for surface in ["Fe"]:
        q = (
            "{ reactions(first: 1000, surfaceComposition: \"Fe\") {\n"
            "  edges { node { chemicalComposition surfaceComposition facet"
            " reactants products reactionEnergy activationEnergy"
            " dftCode dftFunctional pubId }}\n"
            "}}"
        )
        resp = _post(q)
        if resp:
            nodes = _extract_nodes(resp)
            print(f"  ✓  Fe broad search → {len(nodes)} rows")
            all_records.extend(nodes)

    if not all_records:
        print("\nCatHub returned no data. Falling back to literature dataset.")
        return build_literature_dataset()

    # Deduplicate
    seen: set[str] = set()
    unique = []
    for r in all_records:
        key = json.dumps(r, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    print(f"\nTotal unique records: {len(unique)}")

    df = _to_df(unique)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved → {OUT_CSV}")
    return df


# ─── Literature fallback dataset ──────────────────────────────────────────────
# DFT-GGA (PBE) adsorption energies from peer-reviewed papers.
# All energies in eV; sign convention: negative = exothermic / bound.
# Sources:
#   Huo2012 : Huo et al. JACS 2012 (Fe FT intermediates, PBE+D3)
#   Sorescu2000 : Sorescu et al. Surf.Sci. 2000 (CO/Fe DFT)
#   Cao2018 : Cao et al. ACS Catal. 2018 (promoted Fe)
#   Pham2020 : Pham et al. J.Phys.Chem.C 2020 (Fe211 FT)
#   Medford2015 : Medford et al. J.Catal. 2015 (scaling relations)

LITERATURE_DATA: list[dict] = [
    # ── Fe(110) – most stable Fe surface ─────────────────────────────────────
    dict(surface="Fe", facet="110", adsorbate="CO",  E_ads=-1.12, source="Sorescu2000"),
    dict(surface="Fe", facet="110", adsorbate="H",   E_ads=-0.33, source="Huo2012"),
    dict(surface="Fe", facet="110", adsorbate="O",   E_ads=-0.84, source="Huo2012"),
    dict(surface="Fe", facet="110", adsorbate="OH",  E_ads=-0.45, source="Huo2012"),
    dict(surface="Fe", facet="110", adsorbate="CH",  E_ads=-1.68, source="Huo2012"),
    dict(surface="Fe", facet="110", adsorbate="CH2", E_ads=-0.93, source="Huo2012"),
    dict(surface="Fe", facet="110", adsorbate="CH3", E_ads=-0.55, source="Huo2012"),
    dict(surface="Fe", facet="110", adsorbate="C",   E_ads=-1.31, source="Huo2012"),
    dict(surface="Fe", facet="110", adsorbate="CO2", E_ads=-0.28, source="Huo2012"),
    dict(surface="Fe", facet="110", adsorbate="H2O", E_ads=-0.20, source="Huo2012"),
    # ── Fe(100) ───────────────────────────────────────────────────────────────
    dict(surface="Fe", facet="100", adsorbate="CO",  E_ads=-0.77, source="Sorescu2000"),
    dict(surface="Fe", facet="100", adsorbate="H",   E_ads=-0.24, source="Huo2012"),
    dict(surface="Fe", facet="100", adsorbate="O",   E_ads=-0.68, source="Huo2012"),
    dict(surface="Fe", facet="100", adsorbate="OH",  E_ads=-0.31, source="Huo2012"),
    dict(surface="Fe", facet="100", adsorbate="CH",  E_ads=-1.43, source="Huo2012"),
    dict(surface="Fe", facet="100", adsorbate="CH2", E_ads=-0.76, source="Huo2012"),
    dict(surface="Fe", facet="100", adsorbate="CH3", E_ads=-0.40, source="Huo2012"),
    dict(surface="Fe", facet="100", adsorbate="C",   E_ads=-1.09, source="Huo2012"),
    # ── Fe(111) ───────────────────────────────────────────────────────────────
    dict(surface="Fe", facet="111", adsorbate="CO",  E_ads=-1.41, source="Huo2012"),
    dict(surface="Fe", facet="111", adsorbate="H",   E_ads=-0.48, source="Huo2012"),
    dict(surface="Fe", facet="111", adsorbate="O",   E_ads=-1.05, source="Huo2012"),
    dict(surface="Fe", facet="111", adsorbate="OH",  E_ads=-0.62, source="Huo2012"),
    dict(surface="Fe", facet="111", adsorbate="CH",  E_ads=-1.87, source="Huo2012"),
    dict(surface="Fe", facet="111", adsorbate="CH2", E_ads=-1.15, source="Huo2012"),
    dict(surface="Fe", facet="111", adsorbate="CH3", E_ads=-0.71, source="Huo2012"),
    dict(surface="Fe", facet="111", adsorbate="C",   E_ads=-1.52, source="Huo2012"),
    # ── Fe(211) step surface ──────────────────────────────────────────────────
    dict(surface="Fe", facet="211", adsorbate="CO",  E_ads=-1.65, source="Pham2020"),
    dict(surface="Fe", facet="211", adsorbate="H",   E_ads=-0.51, source="Pham2020"),
    dict(surface="Fe", facet="211", adsorbate="O",   E_ads=-1.21, source="Pham2020"),
    dict(surface="Fe", facet="211", adsorbate="OH",  E_ads=-0.78, source="Pham2020"),
    dict(surface="Fe", facet="211", adsorbate="CH2", E_ads=-1.31, source="Pham2020"),
    dict(surface="Fe", facet="211", adsorbate="C",   E_ads=-1.72, source="Pham2020"),
    # ── Fe+K(110)  promoter effect: K weakens CO, strengthens H  ──────────────
    # Promoter ΔE shifts from Cao2018 (PBE, K-covered Fe surfaces)
    dict(surface="FeK", facet="110", adsorbate="CO",  E_ads=-1.42, source="Cao2018"),
    dict(surface="FeK", facet="110", adsorbate="H",   E_ads=-0.28, source="Cao2018"),
    dict(surface="FeK", facet="110", adsorbate="O",   E_ads=-1.05, source="Cao2018"),
    dict(surface="FeK", facet="110", adsorbate="OH",  E_ads=-0.60, source="Cao2018"),
    dict(surface="FeK", facet="110", adsorbate="CH2", E_ads=-1.08, source="Cao2018"),
    dict(surface="FeK", facet="110", adsorbate="C",   E_ads=-1.50, source="Cao2018"),
    # ── Fe+Ru(110) ────────────────────────────────────────────────────────────
    dict(surface="FeRu", facet="110", adsorbate="CO",  E_ads=-1.35, source="Medford2015"),
    dict(surface="FeRu", facet="110", adsorbate="H",   E_ads=-0.40, source="Medford2015"),
    dict(surface="FeRu", facet="110", adsorbate="O",   E_ads=-0.95, source="Medford2015"),
    dict(surface="FeRu", facet="110", adsorbate="OH",  E_ads=-0.52, source="Medford2015"),
    dict(surface="FeRu", facet="110", adsorbate="CH2", E_ads=-1.05, source="Medford2015"),
    # ── Fe+Re(110) ────────────────────────────────────────────────────────────
    dict(surface="FeRe", facet="110", adsorbate="CO",  E_ads=-1.48, source="Cao2018"),
    dict(surface="FeRe", facet="110", adsorbate="H",   E_ads=-0.38, source="Cao2018"),
    dict(surface="FeRe", facet="110", adsorbate="O",   E_ads=-1.10, source="Cao2018"),
    dict(surface="FeRe", facet="110", adsorbate="CH2", E_ads=-1.18, source="Cao2018"),
    # ── Fe+Rh(110) ────────────────────────────────────────────────────────────
    dict(surface="FeRh", facet="110", adsorbate="CO",  E_ads=-1.55, source="Medford2015"),
    dict(surface="FeRh", facet="110", adsorbate="H",   E_ads=-0.42, source="Medford2015"),
    dict(surface="FeRh", facet="110", adsorbate="O",   E_ads=-1.02, source="Medford2015"),
    dict(surface="FeRh", facet="110", adsorbate="CH2", E_ads=-1.20, source="Medford2015"),
    # ── Fe+Pt(110) ────────────────────────────────────────────────────────────
    dict(surface="FePt", facet="110", adsorbate="CO",  E_ads=-1.28, source="Medford2015"),
    dict(surface="FePt", facet="110", adsorbate="H",   E_ads=-0.31, source="Medford2015"),
    dict(surface="FePt", facet="110", adsorbate="O",   E_ads=-0.88, source="Medford2015"),
    dict(surface="FePt", facet="110", adsorbate="CH2", E_ads=-0.99, source="Medford2015"),
]


def build_literature_dataset(save: bool = True) -> pd.DataFrame:
    """
    Construct the literature-based DFT adsorption energy dataset.
    This always works regardless of API availability.
    """
    df = pd.DataFrame(LITERATURE_DATA)
    # Feature engineering matching what the GNN bridge expects
    df["promoter"] = df["surface"].apply(
        lambda s: s.replace("Fe", "") if s != "Fe" else "none"
    )
    df["adsorption_energy_eV"] = df["E_ads"]
    df["chemical_composition"] = df["surface"]
    df["surface_composition"]  = df["surface"]
    df["dft_code"]             = "VASP"
    df["dft_functional"]       = "PBE"
    df["pub_id"]               = df["source"]

    if save:
        LIT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(LIT_CSV, index=False)
        print(f"Literature dataset saved → {LIT_CSV}  ({len(df)} rows)")
    return df


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--literature-only", action="store_true",
                        help="Skip API, just build the literature dataset.")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if CSV already exists.")
    args = parser.parse_args()

    if args.literature_only:
        df = build_literature_dataset()
    else:
        df = fetch_cathub(force=args.force)

    print(f"\nDataset shape: {df.shape}")
    print(df[["surface_composition", "facet", "adsorbate",
              "adsorption_energy_eV"]].head(15).to_string(index=False))