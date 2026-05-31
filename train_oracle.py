"""
train_oracle.py (predicts log STY and selectivity)
"""
import pandas as pd, numpy as np, joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from config import BLENDED_RUNS_CSV, DESCRIPTORS_CSV, ORACLE_MODEL

runs = pd.read_csv(BLENDED_RUNS_CSV)
desc = pd.read_csv(DESCRIPTORS_CSV)
runs["facet"] = runs["facet"].astype(str)
desc["facet"] = desc["facet"].astype(str)

merged = runs.merge(desc, on=["facet","promoter"], how="left")
e_cols = [c for c in merged.columns if c.startswith("E_")]
feature_cols = e_cols + ["temperature_K", "pressure_bar"]

merged = merged.dropna(subset=feature_cols)
X = merged[feature_cols]
y_sty = np.log10(merged["STY_mg_gcat_h"].values / 1000 + 1e-12)
y_sel = merged["SAF_selectivity"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model_sty = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_scaled, y_sty)
model_sel = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_scaled, y_sel)

joblib.dump({
    "model_sty": model_sty, "model_sel": model_sel,
    "scaler": scaler, "feature_cols": feature_cols,
}, ORACLE_MODEL)
print("Oracle saved.")