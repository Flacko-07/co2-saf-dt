# CO₂-to-SAF Digital Twin

> A closed-loop AI system for catalyst discovery and process optimisation targeting Sustainable Aviation Fuel (SAF) synthesis from CO₂ via Fischer–Tropsch chemistry.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

---

## Overview

This project builds a **digital twin** of a CO₂-to-SAF Fischer–Tropsch reactor. It couples a physics-informed plant model with a machine-learning oracle, a GAN-based catalyst generator, and an active-learning loop to iteratively discover catalysts that maximise the product score `STY × SAF selectivity`.

**Key results:**
- Best product score improved from **94.8 → 144.2** (52% gain) after active learning.
- Oracle achieves near-perfect parity with the plant model (R² = 1.000, MAE ≈ 0.0).
- GAN-generated candidates with K, Rh, Ru, Pd, Re, and Pt promoters consistently outperform the Shanghai Pilot baseline in both STY and SAF selectivity.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Digital Twin Loop                    │
│                                                         │
│  Plant Model  ──►  Oracle (RF)  ──►  Active Learning    │
│       ▲                                    │            │
│       │           GAN Generator  ◄─────────┘            │
│       │                 │                               │
│       └──── CHGNet Descriptors ◄── Catalyst Pool        │
└─────────────────────────────────────────────────────────┘
```

| Module | File(s) | Description |
|---|---|---|
| Plant Model | `plant_calibrated.py` | Physics-informed FT reactor model calibrated to pilot data |
| Oracle | `train_oracle.py`, `active_learning_oracle.py` | Random Forest trained on blended synthetic + real data |
| Active Learning | `active_learning.py` | Expected Improvement acquisition on `STY × selectivity` |
| GAN Generator | `catalyst_gan.py`, `process_gan.py` | Property-guided WGAN-GP conditioned on top-20% class |
| Descriptor Engine | `compute_chg_descriptors.py` | CHGNet universal ML potential for adsorption energies |
| Data Pipeline | `blend_datasets.py`, `build_pool.py`, `generate_*.py` | Synthetic + real data blending and candidate pool building |
| Scoring | `score_candidates.py` | Evaluates GAN candidates via oracle |
| Visualisation | `visualize.py` | All result plots (parity, Pareto front, active learning) |

---

## Model Assumptions

1. **Plant model** uses constant single-pass CO conversion (0.51) with a promoter-dependent multiplier.
2. **Chain growth probability (α)** varies linearly with temperature (base α = 0.78, −0.0005 K⁻¹) and promoter (+0.01 per multiplier unit).
3. **Selectivity** is mass selectivity of C8–C16 paraffins (SAF range).
4. **Adsorption energies** from CHGNet (universal ML potential) used as catalyst descriptors.
5. **Oracle** is a Random Forest trained on blended synthetic + real plant data.
6. **Active learning** uses Expected Improvement on product score (`STY × selectivity`).
7. **GANs** are property-guided, conditioned on top-20% class, with gradient penalty (WGAN-GP).

---

## Results

| Metric | Value |
|---|---|
| Initial best product score | 94.8 |
| Post-active-learning best score | **144.2** |
| Oracle STY R² | 1.000 |
| Oracle selectivity R² | 1.000 |
| Best promoters (Pareto front) | Rh, Ru, Pd |
| Best STY observed | ~345 mg/g/h |
| Best SAF selectivity observed | ~0.419 |

---

## Installation

### Option A — Conda (recommended)

```bash
git clone https://github.com/Flacko-07/co2-saf-dt.git
cd co2-saf-dt
conda env create -f environment.yml
conda activate co2-saf-dt
```

### Option B — venv

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # if present, else install manually
```

---

## Quick Start

```bash
# 1. Generate synthetic plant runs
python generate_synthetic_runs.py

# 2. Blend with real calibration data
python blend_datasets.py

# 3. Compute CHGNet descriptors
python compute_chg_descriptors.py
python process_chg_descriptors.py

# 4. Build candidate pool
python build_pool.py

# 5. Train oracle
python train_oracle.py

# 6. Train GANs
python catalyst_gan.py
python process_gan.py

# 7. Sample GAN candidates
python sample_catalyst_gan.py
python sample_process_gan.py

# 8. Run active learning loop
python active_learning.py

# 9. Score and rank candidates
python score_candidates.py

# 10. Visualise results
python visualize.py
```

---

## Project Structure

```
co2-saf-dt/
├── data/                          # Raw and processed datasets
├── active_learning.py             # Active learning loop (EI acquisition)
├── active_learning_oracle.py      # Oracle-in-the-loop AL variant
├── blend_datasets.py              # Merge synthetic + real data
├── build_pool.py                  # Candidate catalyst pool builder
├── catalyst_gan.py                # Catalyst space WGAN-GP
├── compute_chg_descriptors.py     # CHGNet adsorption energy computation
├── config.py                      # Global configuration and hyperparameters
├── download_fairchem_model.py     # FairChem model download utility
├── download_oc20.py               # OC20 dataset download
├── extract_oc20_descriptors.py    # OC20 descriptor extraction
├── fetch_catalysis_hub.py         # CatalysisHub data fetch
├── generate_calibrated_dataset.py # Calibrated dataset generation
├── generate_synthetic_runs.py     # Synthetic FT run generation
├── plant_calibrated.py            # Physics-informed plant model
├── process_chg_descriptors.py     # CHGNet descriptor post-processing
├── process_descriptors.py         # General descriptor processing
├── process_gan.py                 # Process condition GAN
├── sample_catalyst_gan.py         # Sample from catalyst GAN
├── sample_process_gan.py          # Sample from process GAN
├── score_candidates.py            # Score GAN candidates via oracle
├── train_oc20_model.py            # Fine-tune OC20 model
├── train_oracle.py                # Train oracle (Random Forest)
├── train_schnet_oc20.py           # SchNet training on OC20
├── train_surrogates.py            # Surrogate model training
├── verify_data.py                 # Data integrity checks
├── visualize.py                   # Generate all result figures
├── environment.yml                # Conda environment spec
└── README.md
```

---

## Known Limitations & Future Work

- **Constant CO conversion (0.51):** A dynamic conversion model (CSTR/PFR) would improve realism.
- **Linear α–temperature model:** Replacing with a microkinetics-derived model (e.g., Anderson–Schulz–Flory with site coverage) is a natural next step.
- **Pressure insensitivity:** The pressure plots show near-zero variance — the plant model likely needs a pressure-dependent rate term.
- **Oracle calibration leakage:** R² = 1.000 on parity suggests the oracle may be memorising the plant model; adding independent held-out real runs would validate generalisation.
- **GAN mode collapse:** Introduce a Pareto-aware diversity metric (e.g., crowding distance) in the conditioning strategy.
- **Uncertainty quantification:** Replace the RF oracle with a Gaussian Process or Deep Ensemble for principled UQ in the EI acquisition.
- **Multi-fidelity loop:** Couple DFT-level validation for top-10 GAN candidates.

---

## Citation

If you use this work, please cite:

```bibtex
@software{co2_saf_dt_2026,
  title  = {CO2-to-SAF Digital Twin},
  author = {Flacko-07},
  year   = {2026},
  url    = {https://github.com/Flacko-07/co2-saf-dt}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
