"""
catalyst_gan.py
Stable property‑guided GAN for catalyst formulations.
Uses GP surrogates for reliable score evaluation.
"""
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from config import POOL_CSV, DESCRIPTORS_CSV

# ── Load data ──────────────────────────────────────────────────────────────
pool = pd.read_csv(POOL_CSV)
desc = pd.read_csv(DESCRIPTORS_CSV)
desc["facet"] = desc["facet"].astype(str)
pool["facet"] = pool["facet"].astype(str)
merged = pool.merge(desc, on=["facet", "promoter"], how="left")
e_cols = [c for c in merged.columns if c.startswith("E_")]

facet_le = LabelEncoder()
prom_le  = LabelEncoder()
facet_idx = facet_le.fit_transform(merged["facet"])
prom_idx  = prom_le.fit_transform(merged["promoter"])
n_facets = len(facet_le.classes_)
n_proms  = len(prom_le.classes_)

cont_cols = ["temperature_K", "pressure_bar"] + e_cols
cont = merged[cont_cols].values
cont_scaler = StandardScaler()
cont_norm = cont_scaler.fit_transform(cont)

score = merged["score"].values
threshold = np.quantile(score, 0.8)
labels = (score >= threshold).astype(np.float32)

cont_t  = torch.tensor(cont_norm, dtype=torch.float32)
facet_t = torch.tensor(facet_idx, dtype=torch.long)
prom_t  = torch.tensor(prom_idx, dtype=torch.long)
c_t     = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(cont_t, facet_t, prom_t, c_t)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# ── Networks ────────────────────────────────────────────────────────────────
latent_dim = 64
cont_dim = cont_norm.shape[1]

class Gen(nn.Module):
    def __init__(self):
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

class Disc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cont_dim + n_facets + n_proms + 1, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, cont, f_oh, p_oh, c):
        inp = torch.cat([cont, f_oh, p_oh, c], dim=1)
        return self.net(inp)

# ── Load GP surrogates as frozen evaluator ─────────────────────────────────
surrogates = joblib.load("active_surrogates.pkl")
gp_rate = surrogates["gp_rate"]
gp_sel  = surrogates["gp_sel"]

def evaluate_batch(cont_tensor, facet_idx, prom_idx):
    """Return predicted score (STY * selectivity)."""
    cont_np = cont_tensor.detach().cpu().numpy()
    cont_real = cont_scaler.inverse_transform(cont_np)
    facets = facet_le.inverse_transform(facet_idx.detach().cpu().numpy())
    proms  = prom_le.inverse_transform(prom_idx.detach().cpu().numpy())
    df = pd.DataFrame({
        "facet": facets,
        "promoter": proms,
        "temperature_K": cont_real[:, 0],
        "pressure_bar": cont_real[:, 1],
    })
    X_gp = gp_rate.named_steps["prep"].transform(df[["facet","promoter","temperature_K","pressure_bar"]])
    mu_rate, _ = gp_rate.named_steps["gp"].predict(X_gp, return_std=True)
    mu_sel, _  = gp_sel.named_steps["gp"].predict(X_gp, return_std=True)
    sty_pred = 10**mu_rate * 1000.0
    score = sty_pred * mu_sel
    return torch.tensor(score, dtype=torch.float32).to(cont_tensor.device).unsqueeze(1)

# ── Training setup ──────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Gen().to(device)
D = Disc().to(device)

opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.9))

lambda_gp = 10.0
lambda_prop = 0.5
tau = 1.0

for epoch in range(500):
    for cont_real, f_real, p_real, c_real in loader:
        cont_real, c_real = cont_real.to(device), c_real.to(device)
        f_oh_real = F.one_hot(f_real, n_facets).float().to(device)
        p_oh_real = F.one_hot(p_real, n_proms).float().to(device)
        bs = cont_real.size(0)

        # ── Discriminator ──────────────────────────────────────────────────
        z = torch.randn(bs, latent_dim, device=device)
        cont_fake, fl, pl = G(z, c_real)
        f_soft = F.gumbel_softmax(fl, tau=tau, hard=False)
        p_soft = F.gumbel_softmax(pl, tau=tau, hard=False)

        d_real = D(cont_real, f_oh_real, p_oh_real, c_real)
        d_fake = D(cont_fake.detach(), f_soft.detach(), p_soft.detach(), c_real)
        loss_D = (nn.ReLU()(1.0 - d_real)).mean() + (nn.ReLU()(1.0 + d_fake)).mean()

        # Gradient penalty
        alpha = torch.rand(bs, 1, device=device)
        int_cont = alpha * cont_real + (1 - alpha) * cont_fake.detach()
        int_f = alpha * f_oh_real + (1 - alpha) * f_soft.detach()
        int_p = alpha * p_oh_real + (1 - alpha) * p_soft.detach()
        int_cont.requires_grad_(True); int_f.requires_grad_(True); int_p.requires_grad_(True)
        d_int = D(int_cont, int_f, int_p, c_real)
        grads = torch.autograd.grad(outputs=d_int, inputs=[int_cont, int_f, int_p],
                                    grad_outputs=torch.ones_like(d_int),
                                    create_graph=True, retain_graph=True)
        gp = lambda_gp * sum((g.norm(2, dim=1) - 1) ** 2 for g in grads).mean()
        loss_D_total = loss_D + gp

        opt_D.zero_grad()
        loss_D_total.backward()
        opt_D.step()

        # ── Generator ──────────────────────────────────────────────────────
        z = torch.randn(bs, latent_dim, device=device)
        cont_fake, fl, pl = G(z, c_real)
        f_soft = F.gumbel_softmax(fl, tau=tau, hard=False)
        p_soft = F.gumbel_softmax(pl, tau=tau, hard=False)
        d_fake = D(cont_fake, f_soft, p_soft, c_real)
        loss_G_adv = -d_fake.mean()

        # Property guidance (use hard choices for evaluation)
        f_hard = fl.argmax(dim=1)
        p_hard = pl.argmax(dim=1)
        prop_score = evaluate_batch(cont_fake, f_hard, p_hard)
        prop_score_norm = prop_score / 150.0    # normalise
        loss_G_prop = -prop_score_norm.mean()
        loss_G = loss_G_adv + lambda_prop * loss_G_prop

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    if epoch % 100 == 0:
        with torch.no_grad():
            z = torch.randn(500, latent_dim, device=device)
            c = torch.ones(500, 1, device=device)
            cf, fl, pl = G(z, c)
            facets = facet_le.inverse_transform(fl.argmax(dim=1).cpu().numpy())
            proms  = prom_le.inverse_transform(pl.argmax(dim=1).cpu().numpy())
            unique = len(set(zip(facets, proms)))
            print(f"Epoch {epoch:3d} | D: {loss_D.item():.3f}  G_adv: {loss_G_adv.item():.3f}  "
                  f"G_prop: {loss_G_prop.item():.3f}  |  unique catalysts: {unique}")

torch.save(G.state_dict(), "catalyst_gan_g.pth")
joblib.dump(cont_scaler, "catalyst_scaler.pkl")
joblib.dump((facet_le, prom_le), "catalyst_encoders.pkl")
print("Catalyst GAN trained and saved.")