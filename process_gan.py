"""
process_gan.py
Stable property‑guided GAN for operating conditions.
Uses GP surrogate (from active learning) as the property guide – bounded & reliable.
"""
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from config import POOL_CSV

# ── Load pool and active surrogates ────────────────────────────────────────
pool = pd.read_csv(POOL_CSV)
X = pool[["temperature_K", "pressure_bar"]].values
y_score = pool["score"].values   # STY * selectivity

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# Condition: top 20 % = 1, rest = 0
threshold = np.quantile(y_score, 0.8)
labels = (y_score >= threshold).astype(np.float32)

X_t = torch.tensor(X_norm, dtype=torch.float32)
c_t = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
loader = DataLoader(TensorDataset(X_t, c_t), batch_size=64, shuffle=True)

# ── Networks ────────────────────────────────────────────────────────────────
latent_dim = 16
input_dim = 2

class Gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Linear(128, input_dim),
        )
    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=1))

class Disc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1),   # no sigmoid – we'll use hinge loss
        )
    def forward(self, x, c):
        return self.net(torch.cat([x, c], dim=1))

# ── Load GP surrogates as the property evaluator (frozen) ──────────────────
surrogates = joblib.load("active_surrogates.pkl")
gp_rate = surrogates["gp_rate"]
gp_sel  = surrogates["gp_sel"]

def evaluate_batch(x_tensor):
    """Return predicted score (STY * selectivity) for normalised inputs."""
    x_np = x_tensor.detach().cpu().numpy()
    x_real = scaler.inverse_transform(x_np)
    # Build a DataFrame for the GP (needs categorical columns too – we use dummy '110'/'none')
    df = pd.DataFrame(x_real, columns=["temperature_K", "pressure_bar"])
    df["facet"] = "110"
    df["promoter"] = "none"
    X_gp = gp_rate.named_steps["prep"].transform(df[["facet", "promoter", "temperature_K", "pressure_bar"]])
    mu_rate, _ = gp_rate.named_steps["gp"].predict(X_gp, return_std=True)
    mu_sel, _  = gp_sel.named_steps["gp"].predict(X_gp, return_std=True)
    sty_pred = 10**mu_rate * 1000.0
    score = sty_pred * mu_sel
    return torch.tensor(score, dtype=torch.float32).to(x_tensor.device).unsqueeze(1)

# ── Training setup ──────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Gen().to(device)
D = Disc().to(device)

opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.9))

lambda_gp = 10.0          # gradient penalty weight
lambda_prop = 1.0         # property guidance weight (much smaller)

for epoch in range(200):
    for x_real, c_real in loader:
        x_real, c_real = x_real.to(device), c_real.to(device)
        bs = x_real.size(0)

        # ── Train Discriminator (with gradient penalty) ─────────────────────
        z = torch.randn(bs, latent_dim, device=device)
        x_fake = G(z, c_real)

        # Hinge loss
        d_real = D(x_real, c_real)
        d_fake = D(x_fake.detach(), c_real)
        loss_D = (nn.ReLU()(1.0 - d_real)).mean() + (nn.ReLU()(1.0 + d_fake)).mean()

        # Gradient penalty
        alpha = torch.rand(bs, 1, device=device)
        interpolates = alpha * x_real + (1 - alpha) * x_fake.detach()
        interpolates.requires_grad_(True)
        d_interp = D(interpolates, c_real)
        grads = torch.autograd.grad(outputs=d_interp, inputs=interpolates,
                                    grad_outputs=torch.ones_like(d_interp),
                                    create_graph=True, retain_graph=True)[0]
        gp = lambda_gp * ((grads.norm(2, dim=1) - 1) ** 2).mean()
        loss_D_total = loss_D + gp

        opt_D.zero_grad()
        loss_D_total.backward()
        opt_D.step()

        # ── Train Generator (every iteration, not every 5) ──────────────────
        z = torch.randn(bs, latent_dim, device=device)
        x_fake = G(z, c_real)
        d_fake = D(x_fake, c_real)
        loss_G_adv = -d_fake.mean()

        # Property guidance: evaluate with GP, clip contribution
        prop_score = evaluate_batch(x_fake)
        # Normalise to ~0-1 range (max pool score ≈ 150)
        prop_score_norm = prop_score / 150.0
        loss_G_prop = -prop_score_norm.mean()
        loss_G = loss_G_adv + lambda_prop * loss_G_prop

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    if epoch % 50 == 0:
        with torch.no_grad():
            z = torch.randn(1000, latent_dim, device=device)
            c = torch.ones(1000, 1, device=device)
            fake = G(z, c).cpu().numpy()
            real_fake = scaler.inverse_transform(fake)
            print(f"Epoch {epoch:3d} | D loss: {loss_D.item():.3f}  G_adv: {loss_G_adv.item():.3f}  "
                  f"G_prop: {loss_G_prop.item():.3f}  |  T range: [{real_fake[:,0].min():.0f}, {real_fake[:,0].max():.0f}]")

torch.save(G.state_dict(), "process_gan_g.pth")
joblib.dump(scaler, "process_scaler.pkl")
print("Process GAN trained and saved.")