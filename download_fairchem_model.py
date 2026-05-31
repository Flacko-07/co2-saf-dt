"""
download_fairchem_model.py
Download the GemNet-OC IS2RE checkpoint from Hugging Face fairchem.
"""
from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id = "fairchem/GemNet-OC-IS2RE-OC20-All"
filename = "gemnet-oc_is2re_all.pt"   # exact filename in the repo

target_dir = Path("checkpoints")
target_dir.mkdir(parents=True, exist_ok=True)

local_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=target_dir,
    local_dir_use_symlinks=False,
)
print(f"Checkpoint downloaded to {local_path}")
