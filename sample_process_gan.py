"""
sample_process_gan.py
Generate top operating conditions from the trained process GAN.
"""
import torch
import numpy as np
import joblib
from process_gan import Generator  # Now works, no training on import

# Load scaler
scaler = joblib.load("process_scaler.pkl")
input_dim = 2
latent_dim = 16

# Load model
G = Generator(latent_dim=latent_dim, input_dim=input_dim)
G.load_state_dict(torch.load("process_gan_g.pth", map_location="cpu"))
G.eval()

# Sample many points, decode, and filter by constraints
z = torch.randn(2000, latent_dim)
c = torch.ones(2000, 1)   # condition = "top" class

with torch.no_grad():
    x_norm = G(z, c).numpy()

x_real = scaler.inverse_transform(x_norm)  # columns: temperature_K, pressure_bar

# Filter realistic bounds
valid = (x_real[:, 0] >= 473) & (x_real[:, 0] <= 673) & (x_real[:, 1] >= 1) & (x_real[:, 1] <= 40)
x_valid = x_real[valid]

print(f"Generated {len(x_valid)} valid process points.")
print("Top 10 (sorted by temperature):")
top = x_valid[np.argsort(x_valid[:, 0])[::-1][:10]]
for t, p in top:
    print(f"  T = {t:.1f} K, P = {p:.1f} bar")