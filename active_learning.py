"""
active_learning.py
Active learning loop for catalyst/process optimisation.
"""
import pandas as pd
import numpy as np
import joblib
from scipy.stats import norm
from sklearn.preprocessing import OrdinalEncoder
from plant_calibrated import CO2_to_SAF_Plant
from config import PROC_DIR, PROMOTER_MULTIPLIER

# Load surrogates
model_rate = joblib.load("surrogate_rate.pkl")
model_sel  = joblib.load("surrogate_sel.pkl")

# Dataframe to accumulate all evaluated points
data = pd.read_csv(PROC_DIR / "plant_calibrated_dataset.csv")

# Candidate pool (discrete grid) – can be reduced for speed
FE_FACETS = ["110", "100", "111"]
PROMOTERS = list(PROMOTER_MULTIPLIER.keys())
T_grid = np.arange(473, 678, 5)
P_grid = np.arange(1, 41, 2)

def build_pool():
    import itertools
    rows = []
    for f, p, t, P in itertools.product(FE_FACETS, PROMOTERS, T_grid, P_grid):
        rows.append([f, p, t, P])
    pool_df = pd.DataFrame(rows, columns=["facet","promoter","temperature_K","pressure_bar"])
    return pool_df

def oracle(row):
    """Evaluate plant model for one candidate."""
    plant = CO2_to_SAF_Plant(
        T_rwgs=673,
        P_bar=row["pressure_bar"],
        T_ft=row["temperature_K"],
        catalyst_mass_g=1.0,
        ft_co_conversion=0.51,
        promoter_multiplier=PROMOTER_MULTIPLIER[row["promoter"]],
    )
    fresh = {"CO2": 1.0, "H2": 3.0}
    res = plant.run_simulation(fresh, simulation_hours=1.0)
    rate_kg_h_g = res["STY_mg_gcat_h"] / 1e6
    sel = res["per_step_metrics"][-1]["selectivity"]
    return rate_kg_h_g, sel

def acq_ei(X, best_y):
    """Expected Improvement over log‑rate * selectivity."""
    mu_rate, _ = model_rate.predict(X, return_std=True)
    mu_sel, _ = model_sel.predict(X, return_std=True)
    # Use a simple combined score: mu_rate * mu_sel
    score = mu_rate * mu_sel
    # Use rate standard deviation for EI (largest uncertainty typically in rate)
    _, std_rate = model_rate.predict(X, return_std=True)
    # Clip std
    std_rate = np.clip(std_rate, 1e-6, None)
    imp = score - best_y
    Z = imp / std_rate
    ei = imp * norm.cdf(Z) + std_rate * norm.pdf(Z)
    ei[std_rate <= 1e-6] = 0.0
    return ei

# Initial best score from training data
data["score"] = np.log10(data["SAF_kg_h_per_g"] + 1e-12) * data["SAF_selectivity"]
best_y = data["score"].max()
print(f"Initial best score: {best_y:.4f}")

pool = build_pool()
for iteration in range(10):
    print(f"\n--- Iteration {iteration+1} ---")
    # Transform pool
    X_pool = model_rate.named_steps["preproc"].transform(pool)
    ei_vals = acq_ei(X_pool, best_y)
    idx = np.argmax(ei_vals)
    proposed = pool.iloc[idx]
    print(f"Proposed: facet={proposed['facet']}, promoter={proposed['promoter']}, "
          f"T={proposed['temperature_K']} K, P={proposed['pressure_bar']} bar")

    # Query oracle
    rate_kg, sel = oracle(proposed)
    new_score = np.log10(rate_kg + 1e-12) * sel
    print(f"Oracle: rate={rate_kg:.6e} kg/h/g, sel={sel:.3f}, score={new_score:.4f}")

    # Append to dataset
    new_row = proposed.to_dict()
    new_row["SAF_kg_h_per_g"] = rate_kg
    new_row["SAF_selectivity"] = sel
    new_row["CO_conversion"] = np.nan  # not stored but optional
    data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
    best_y = max(best_y, new_score)

    # Retrain surrogates
    X = data[["facet","promoter","temperature_K","pressure_bar"]]
    y_rate = np.log10(data["SAF_kg_h_per_g"].values + 1e-12)
    y_sel  = data["SAF_selectivity"].values
    model_rate.fit(X, y_rate)
    model_sel.fit(X, y_sel)

    print(f"Dataset size: {len(data)}, current best score: {best_y:.4f}")

# Save updated dataset and retrained models
data.to_csv(PROC_DIR / "plant_calibrated_dataset.csv", index=False)
joblib.dump(model_rate, "surrogate_rate.pkl")
joblib.dump(model_sel, "surrogate_sel.pkl")
print("Active learning complete. Updated dataset and surrogates saved.")