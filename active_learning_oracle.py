"""
active_learning_oracle.py
Active learning loop over the oracle‑evaluated pool using Expected Improvement.
Uses product score (STY * selectivity) and stops early when no progress is made.
"""
import pandas as pd
import numpy as np
import joblib
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from config import POOL_CSV

# ── Load the pre‑evaluated pool ────────────────────────────────────────────
pool = pd.read_csv(POOL_CSV)

# ── Initial training set (random 200 points from the pool) ─────────────────
np.random.seed(42)
init_idx = np.random.choice(len(pool), size=min(200, len(pool)), replace=False)
train = pool.iloc[init_idx]

X_train = train[["facet", "promoter", "temperature_K", "pressure_bar"]]
y_rate = np.log10(train["STY_mg_gcat_h"].values / 1000.0 + 1e-12)   # log10(g/g/h)
y_sel  = train["SAF_selectivity"].values

# ── Preprocessor (standardise numerics, ordinal encode categories) ─────────
preproc = ColumnTransformer([
    ("num", StandardScaler(), ["temperature_K", "pressure_bar"]),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
     ["facet", "promoter"])
])

# ── Gaussian Process surrogates ────────────────────────────────────────────
gp_rate = Pipeline([
    ("prep", preproc),
    ("gp", GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01),
        n_restarts_optimizer=5, normalize_y=True, alpha=1e-6))
])
gp_sel = Pipeline([
    ("prep", preproc),
    ("gp", GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01),
        n_restarts_optimizer=5, normalize_y=True, alpha=1e-6))
])

gp_rate.fit(X_train, y_rate)
gp_sel.fit(X_train, y_sel)

# ── Active learning loop ───────────────────────────────────────────────────
best_score = train["score"].max()
candidates = pool[["facet", "promoter", "temperature_K", "pressure_bar"]]

no_improve = 0                     # counter for early stopping

for it in range(1, 501):           # up to 500 iterations
    # Predict on all remaining candidates
    X_cand = gp_rate.named_steps["prep"].transform(candidates)
    mu_rate, std_rate = gp_rate.named_steps["gp"].predict(X_cand, return_std=True)
    mu_sel, _ = gp_sel.named_steps["gp"].predict(X_cand, return_std=True)

    # Reconstruct predicted STY (mg/g/h) and selectivity → product score
    sty_pred = 10**mu_rate * 1000.0
    score_surr = sty_pred * mu_sel

    # Expected Improvement on product score (using std of rate as uncertainty proxy)
    imp = score_surr - best_score
    with np.errstate(divide='ignore'):
        Z = imp / np.maximum(std_rate, 1e-6)
        ei = imp * norm.cdf(Z) + std_rate * norm.pdf(Z)
        ei[std_rate <= 1e-6] = 0.0

    # Pick the point with highest EI
    next_idx = np.argmax(ei)
    proposed = candidates.iloc[next_idx]
    true_row = pool.iloc[next_idx]         # actual oracle evaluation
    true_score = true_row["score"]

    print(f"Iter {it}: proposed {proposed.to_dict()}, true score = {true_score:.2f}")

    # Add to training set
    X_train = pd.concat([X_train, pd.DataFrame([proposed])], ignore_index=True)
    y_rate = np.append(y_rate, np.log10(true_row["STY_mg_gcat_h"] / 1000.0 + 1e-12))
    y_sel  = np.append(y_sel, true_row["SAF_selectivity"])

    # Update best score and early stopping counter
    if true_score > best_score:
        best_score = true_score
        no_improve = 0
    else:
        no_improve += 1

    if no_improve >= 50:
        print(f"Early stopping – no improvement for 50 iterations. (best = {best_score:.2f})")
        break

    # Retrain surrogates
    gp_rate.fit(X_train, y_rate)
    gp_sel.fit(X_train, y_sel)

# ── Save refined surrogates ────────────────────────────────────────────────
joblib.dump({"gp_rate": gp_rate, "gp_sel": gp_sel}, "active_surrogates.pkl")
print("Active learning finished. Surrogates saved.")