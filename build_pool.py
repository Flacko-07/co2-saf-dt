import pandas as pd, numpy as np, joblib, itertools
from config import DESCRIPTORS_CSV, ORACLE_MODEL, POOL_CSV

bundle = joblib.load(ORACLE_MODEL)
model_sty = bundle["model_sty"]; model_sel = bundle["model_sel"]
scaler    = bundle["scaler"];   feat_cols = bundle["feature_cols"]

desc = pd.read_csv(DESCRIPTORS_CSV); desc["facet"] = desc["facet"].astype(str)

facets = ["110","100","111"]
proms  = ["none","K","Co","Pt","Pd","Ru","Rh","Re"]
T_grid = np.arange(473, 678, 5); P_grid = np.arange(1, 41, 2)

rows = []
for f,p,T,P in itertools.product(facets, proms, T_grid, P_grid):
    cat = desc[(desc.facet==f) & (desc.promoter==p)]
    if cat.empty: continue
    feat = {}
    for col in feat_cols:
        if col in cat.columns: feat[col] = cat.iloc[0][col]
        elif col == "temperature_K": feat[col] = T
        elif col == "pressure_bar": feat[col] = P
    vec = np.array([feat[col] for col in feat_cols]).reshape(1,-1)
    vec_sc = scaler.transform(vec)
    log_sty = model_sty.predict(vec_sc)[0]
    sel = model_sel.predict(vec_sc)[0]
    sty_mg = 10**log_sty * 1000
    rows.append({"facet":f,"promoter":p,"temperature_K":T,"pressure_bar":P,
                 "STY_mg_gcat_h":sty_mg,"SAF_selectivity":sel,"score":sty_mg*sel})
    if len(rows)%500==0: print(f"{len(rows)} points")

pool = pd.DataFrame(rows)
pool.to_csv(POOL_CSV, index=False)
print(f"Pool ({len(pool)} points) saved.")