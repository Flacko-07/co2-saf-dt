"""
train_surrogates.py
Train GP surrogates on the calibrated dataset.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from config import PROC_DIR

# Load dataset
df = pd.read_csv(PROC_DIR / "plant_calibrated_dataset.csv")
X = df[["facet", "promoter", "temperature_K", "pressure_bar"]]
y_rate = np.log10(df["SAF_kg_h_per_g"].values + 1e-12)
y_sel  = df["SAF_selectivity"].values

# Train/validation split
X_train, X_val, yr_train, yr_val, ys_train, ys_val = train_test_split(
    X, y_rate, y_sel, test_size=0.2, random_state=42
)

# Preprocessor
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["temperature_K", "pressure_bar"]),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
     ["facet", "promoter"])
])

# GP kernel
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01)

# Rate GP
gp_rate = Pipeline([
    ("preproc", preprocessor),
    ("gp", GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                    normalize_y=True, alpha=1e-6))
])
gp_rate.fit(X_train, yr_train)

# Selectivity GP
gp_sel = Pipeline([
    ("preproc", preprocessor),
    ("gp", GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                    normalize_y=True, alpha=1e-6))
])
gp_sel.fit(X_train, ys_train)

# Validation metrics
pred_yr_val = gp_rate.predict(X_val)
pred_ys_val = gp_sel.predict(X_val)

print("Rate GP: R² = %.4f, MAE = %.4f" % (r2_score(yr_val, pred_yr_val), mean_absolute_error(yr_val, pred_yr_val)))
print("Sel GP:  R² = %.4f, MAE = %.4f" % (r2_score(ys_val, pred_ys_val), mean_absolute_error(ys_val, pred_ys_val)))

# Save
joblib.dump(gp_rate, "surrogate_rate.pkl")
joblib.dump(gp_sel, "surrogate_sel.pkl")
print("Surrogates saved.")