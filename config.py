"""
config.py
Central configuration for CO2-to-SAF GNN pipeline.
All paths, hyperparameters, and constants live here.
"""
from pathlib import Path
import torch

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR  = ROOT / "data" / "raw"
PROC_DIR  = ROOT / "data" / "processed"
CKPT_DIR  = ROOT / "checkpoints"
DESC_DIR  = ROOT / "descriptors"
RESULT_DIR = ROOT / "results"
BLENDED_RUNS_CSV = DATA_DIR / "blended_plant_runs.csv"
ORIGINAL_RUNS_CSV = DATA_DIR / "original_plant_runs.csv"   # your real experiments
DESCRIPTORS_CSV   = PROC_DIR / "catalyst_descriptors.csv"  # OC20 pivot
ORACLE_MODEL      = ROOT / "oracle.pkl"                    # trained oracle
POOL_CSV          = PROC_DIR / "oracle_pool.csv"  
for d in [DATA_DIR, PROC_DIR, CKPT_DIR, DESC_DIR, RESULT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── OC20 data download URLs ────────────────────────────────────────────────────
# IS2RE = Initial Structure → Relaxed Energy  (best for pretraining descriptors)
OC20_URLS = {
    # 10 k systems – quick smoke-test (~180 MB)
    "is2re_10k":     "https://dl.fbaipublicfiles.com/opencatalystproject/data/is2re/is2re_train_10k.tar.gz",
    # 100 k systems – good pretraining baseline (~2 GB)
    "is2re_100k":    "https://dl.fbaipublicfiles.com/opencatalystproject/data/is2re/is2re_train_100k.tar.gz",
    # Validation (in-distribution), ~24 k systems (~500 MB)
    "is2re_val_id":  "https://dl.fbaipublicfiles.com/opencatalystproject/data/is2re/is2re_val_id.tar.gz",
}
# Pretrained OCP GemNet-OC checkpoint (IS2RE, MIT license)
OCP_PRETRAINED_URL = (
    "https://dl.fbaipublicfiles.com/opencatalystproject/models/"
    "2022_09/is2re/gemnet-OC_is2re_all.pt"
)

# ── Catalysis-Hub API ─────────────────────────────────────────────────────────
CATHUB_GRAPHQL = "https://api.catalysis-hub.org/graphql"

# ── Materials Project API ─────────────────────────────────────────────────────
# Set your free API key at https://materialsproject.org/api
MP_API_KEY = "0QEuBNwixrzsbd1q1395k1bI1TUNGDR6"

# ── Surface / adsorbate space for Fe catalysts ────────────────────────────────
FE_FACETS     = ["110", "100", "111", "211"]          # Miller indices as strings
PROMOTERS     = ["none", "K", "Co", "Pt", "Pd", "Ru", "Rh", "Re"]
ADSORBATES    = ["CO", "H", "O", "OH", "CH2", "CH3"]  # Key FT intermediates
SLAB_LAYERS   = 4     # Number of atomic layers in surface slab
SLAB_VACUUM   = 15.0  # Ångström vacuum above slab
SLAB_SUPERCELL = (2, 2)  # Supercell expansion in x, y

# ── GNN hyperparameters (SchNet) ──────────────────────────────────────────────
GNN_CONFIG = dict(
    hidden_channels   = 256,
    num_filters       = 256,
    num_interactions  = 6,
    num_gaussians     = 50,
    cutoff            = 6.0,   # Å  – interaction cutoff
    max_num_neighbors = 50,
    readout           = "add", # "add" or "mean"
    dipole            = False,
    mean              = None,  # will be set after dataset stats
    std               = None,
)

# ── Training hyperparameters ──────────────────────────────────────────────────
TRAIN_CONFIG = dict(
    batch_size        = 32,
    lr                = 1e-4,
    weight_decay      = 0.0,
    max_epochs        = 100,
    patience          = 15,    # early-stopping patience
    grad_clip_norm    = 10.0,
    scheduler         = "plateau",   # "plateau" or "cosine"
    loss              = "mae",       # "mae" or "mse"
    val_fraction      = 0.1,
    seed              = 42,
    num_workers       = 4,
    pin_memory        = True,
)

# ── Descriptor bridge to plant model ─────────────────────────────────────────
# Plant reference (Shanghai SARI pilot)
PLANT_REFERENCE = dict(
    T_ft_K            = 603,
    P_bar             = 20,
    ft_co_conversion  = 0.51,
    saf_selectivity   = 0.375,
    sty_mg_gcat_h     = 252.7,
)

# Promoter CO-conversion multipliers (empirical baseline;
# will be replaced/refined by GNN-derived regression)
PROMOTER_MULTIPLIER = {
    "none": 1.00,
    "K":    1.15,
    "Co":   1.08,
    "Pt":   1.10,
    "Pd":   1.12,
    "Ru":   1.18,
    "Rh":   1.20,
    "Re":   1.25,
}

# ── GAN hyperparameters ───────────────────────────────────────────────────────
PROCESS_GAN_CONFIG = dict(
    latent_dim        = 64,
    hidden_dim        = 256,
    n_layers          = 4,
    lr_g              = 2e-4,
    lr_d              = 2e-4,
    n_critic          = 5,        # WGAN-GP: critic steps per generator step
    lambda_gp         = 10.0,     # Gradient penalty weight
    lambda_prop       = 5.0,      # Property guidance weight
    batch_size        = 64,
    max_epochs        = 500,
    top_percentile    = 20,       # Use top-20% as "high-quality" class
)

CATALYST_GAN_CONFIG = dict(
    latent_dim        = 128,
    hidden_dim        = 512,
    n_layers          = 5,
    lr_g              = 1e-4,
    lr_d              = 1e-4,
    n_critic          = 5,
    lambda_gp         = 10.0,
    lambda_prop       = 10.0,
    temp_gumbel       = 1.0,      # Gumbel-Softmax temperature for categoricals
    batch_size        = 64,
    max_epochs        = 1000,
    top_percentile    = 20,
)
