"""
train_schnet_oc20.py
Train a PyG SchNet on OC20 IS2RE 10k – handles old PyG serialisation.
"""
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import SchNet
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import lmdb
import pickle
from tqdm import tqdm
from pathlib import Path

LMDB_PATH = "data/raw/oc20/is2res_train_val_test_lmdbs/data/is2re/10k/train"
SAVE_PATH = "checkpoints/schnet_is2re_10k.pt"
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
MAX_SAMPLES = 2000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def oc20_to_graphs(lmdb_path, max_samples=2000):
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    graphs = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for idx, (key, value) in enumerate(cursor):
            if max_samples and idx >= max_samples:
                break
            # Load old PyG Data without triggering version errors
            old_data = pickle.loads(value)

            # Patch the internal store version to match current PyG
            if hasattr(old_data, '_store'):
                old_data._store._version = torch_geometric.__version__

            # Now safe to extract atoms and energy
            atoms = old_data.atoms
            energy = float(old_data.y)

            # Build a fresh Data object (no version issues)
            z   = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
            pos = torch.tensor(atoms.positions, dtype=torch.float32)
            data = Data(z=z, pos=pos, y=torch.tensor([energy], dtype=torch.float32))
            graphs.append(data)
    env.close()
    return graphs

def main():
    print(f"Loading up to {MAX_SAMPLES} training graphs …")
    graphs = oc20_to_graphs(LMDB_PATH, max_samples=MAX_SAMPLES)
    print(f"Loaded {len(graphs)} samples.")

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
        total_loss = 0.0
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