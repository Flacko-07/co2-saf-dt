"""
train_schnet_oc20.py
Train a PyG SchNet model on the OC20 IS2RE 10k split.
No OCP/fairchem required – pure PyTorch Geometric.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import SchNet
from torch_geometric.loader import DataLoader
import lmdb
import pickle
import numpy as np
from ase import Atoms
from tqdm import tqdm
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
LMDB_PATH = "data/raw/oc20/is2res_train_val_test_lmdbs/data/is2re/10k/train"
SAVE_PATH = "checkpoints/schnet_is2re_10k.pt"
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Data loading from LMDB (OC20 format) ────────────────────────────────────
def oc20_lmdb_to_graphs(lmdb_path, max_samples=2000):
    """Convert OC20 IS2RE LMDB to a list of PyG Data objects."""
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    graphs = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for idx, (key, value) in enumerate(cursor):
            if max_samples and idx >= max_samples:
                break
            data = pickle.loads(value)
            # data is a dict with 'atoms' (ASE Atoms) and 'y' (energy)
            atoms = data["atoms"]
            energy = data["y"]  # relaxed energy in eV
            # Convert ASE atoms to PyG Data (simple version: atom types and positions)
            atomic_numbers = atoms.get_atomic_numbers()
            pos = torch.tensor(atoms.positions, dtype=torch.float32)
            z = torch.tensor(atomic_numbers, dtype=torch.long)
            # We'll use SchNet which needs atom positions, atomic numbers, and target energy
            from torch_geometric.data import Data
            graph = Data(z=z, pos=pos, y=torch.tensor([energy], dtype=torch.float32))
            graphs.append(graph)
    env.close()
    return graphs

# ── Train ───────────────────────────────────────────────────────────────────
def main():
    print("Loading training data (2000 samples for speed) …")
    graphs = oc20_lmdb_to_graphs(LMDB_PATH, max_samples=2000)
    loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=True)

    model = SchNet(
        hidden_channels=128,
        num_filters=128,
        num_interactions=3,
        num_gaussians=50,
        cutoff=6.0,
        max_num_neighbors=50,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.L1Loss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            batch = batch.to(DEVICE)
            pred = model(batch.z, batch.pos, batch.batch)
            loss = loss_fn(pred, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: MAE = {total_loss/len(loader):.4f} eV")

    Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
