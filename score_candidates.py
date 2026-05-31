"""
score_candidates.py
Rank GAN‑proposed catalysts using product score (STY * selectivity).
No imports from process_gan/catalyst_gan – never triggers training.
"""
import torch, torch.nn as nn, numpy as np, joblib, pandas as pd

# ── Minimal generator definitions (same architectures as the trained GANs) ──
class ProcessGenerator(nn.Module):
    def __init__(self, latent_dim=16, input_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Linear(128, input_dim),
        )
    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=1))

class CatalystGenerator(nn.Module):
    def __init__(self, latent_dim=64, cont_dim=6, n_facets=3, n_proms=8):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(latent_dim + 1, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
        )
        self.cont_out  = nn.Linear(256, cont_dim)
        self.facet_out = nn.Linear(256, n_facets)
        self.prom_out  = nn.Linear(256, n_proms)

    def forward(self, z, c):
        h = self.shared(torch.cat([z, c], dim=1))
        return self.cont_out(h), self.facet_out(h), self.prom_out(h)

# ── Load saved GANs and encoders ────────────────────────────────────────────
facet_le, prom_le = joblib.load("catalyst_encoders.pkl")
cont_scaler = joblib.load("catalyst_scaler.pkl")
G_cat = CatalystGenerator(64, 6, len(facet_le.classes_), len(prom_le.classes_))
G_cat.load_state_dict(torch.load("catalyst_gan_g.pth", map_location="cpu"))
G_cat.eval()

proc_scaler = joblib.load("process_scaler.pkl")
G_proc = ProcessGenerator(16, 2)
G_proc.load_state_dict(torch.load("process_gan_g.pth", map_location="cpu"))
G_proc.eval()

# ── Load oracle ────────────────────────────────────────────────────────────
bundle = joblib.load("oracle.pkl")
model_sty = bundle["model_sty"]
model_sel = bundle["model_sel"]
scaler_oracle = bundle["scaler"]
feat_cols = bundle["feature_cols"]

desc = pd.read_csv("data/processed/catalyst_descriptors.csv")
desc["facet"] = desc["facet"].astype(str)

N = 5000
z_cat = torch.randn(N, 64)
c_cat = torch.ones(N, 1)
with torch.no_grad():
    cont_norm, fl, pl = G_cat(z_cat, c_cat)
    cont = cont_scaler.inverse_transform(cont_norm.numpy())
    facets = facet_le.inverse_transform(fl.argmax(1).numpy())
    proms  = prom_le.inverse_transform(pl.argmax(1).numpy())

z_proc = torch.randn(N, 16)
c_proc = torch.ones(N, 1)
with torch.no_grad():
    proc_norm = G_proc(z_proc, c_proc).numpy()
proc = proc_scaler.inverse_transform(proc_norm)

rows = []
for i in range(N):
    f, p = str(facets[i]), str(proms[i])
    cat_row = desc[(desc.facet == f) & (desc.promoter == p)]
    if cat_row.empty:
        continue
    T, P = proc[i, 0], proc[i, 1]
    feat = {}
    for col in feat_cols:
        if col in cat_row.columns:
            feat[col] = cat_row.iloc[0][col]
        elif col == "temperature_K":
            feat[col] = T
        elif col == "pressure_bar":
            feat[col] = P
    vec = np.array([feat[col] for col in feat_cols]).reshape(1, -1)
    vec_sc = scaler_oracle.transform(vec)
    log_sty = model_sty.predict(vec_sc)[0]
    sel = model_sel.predict(vec_sc)[0]
    sty_mg = 10**log_sty * 1000
    score = sty_mg * sel
    rows.append((f, p, T, P, sty_mg, sel, score))

df = pd.DataFrame(rows, columns=["facet","promoter","T_K","P_bar","STY_mg_gcat_h","SAF_selectivity","score"])
df = df.sort_values("score", ascending=False)

print("Top 10 candidates (STY × selectivity):")
print(df.head(10).to_string(index=False))

print("\nBest unique catalyst:")
best = df.iloc[0]
print(f"  Facet={best['facet']}, Promoter={best['promoter']}, "
      f"T={best['T_K']:.0f} K, P={best['P_bar']:.1f} bar, "
      f"STY={best['STY_mg_gcat_h']:.1f} mg/g/h, selectivity={best['SAF_selectivity']:.3f}")