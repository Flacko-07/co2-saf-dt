"""
sample_catalyst_gan.py
Generate top catalyst formulations from the trained catalyst GAN.
"""
import torch
import torch.nn.functional as F
import numpy as np
import joblib
from catalyst_gan import Generator  # Now works, no training on import

# Load encoders and scaler
facet_le, prom_le = joblib.load("catalyst_encoders.pkl")
cont_scaler = joblib.load("catalyst_scaler.pkl")

# Load model
cont_dim = 6  # Adjust based on your actual continuous features: T, P, E_CO, E_H, E_O, E_OH
n_facets = len(facet_le.classes_)
n_proms = len(prom_le.classes_)
latent_dim = 64

G = Generator(latent_dim=latent_dim, cont_dim=cont_dim, n_facets=n_facets, n_proms=n_proms)
G.load_state_dict(torch.load("catalyst_gan_g.pth", map_location="cpu"))
G.eval()

# Sample
z = torch.randn(2000, latent_dim)
c = torch.ones(2000, 1)

with torch.no_grad():
    cont_norm, facet_logits, prom_logits = G(z, c)

cont = cont_scaler.inverse_transform(cont_norm.numpy())
facets = facet_le.inverse_transform(facet_logits.argmax(dim=1).numpy())
proms  = prom_le.inverse_transform(prom_logits.argmax(dim=1).numpy())

# Build candidates
candidates = []
for i in range(len(cont)):
    candidates.append((facets[i], proms[i], cont[i, 0], cont[i, 1]))

unique = list(set(candidates))[:20]
print("Top 20 unique catalyst candidates:")
for f, p, t, P in unique:
    print(f"  Facet={f}, Promoter={p}, T={t:.1f} K, P={P:.1f} bar")