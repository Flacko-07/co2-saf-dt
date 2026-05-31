"""
download_oc20.py  –  FIXED v2
═══════════════════════════════════════════════════════════════════════════════
Root cause of the 403:  Meta moved to a single IS2RE bundle.  The old
per-split URLs (is2re_train_10k.tar.gz, etc.) no longer exist.  The correct
URL is:
    https://dl.fbaipublicfiles.com/opencatalystproject/data/
        is2res_train_val_test_lmdbs.tar.gz          ← ONE file, ~8 GB

You already have this unpacked at:
    data/raw/oc20/is2res_train_val_test_lmdbs/data/is2re/{10k,100k,all}/

This script now:
  1.  Maps the existing data to the canonical split names the rest of the
      pipeline uses (resolve_lmdb_path).
  2.  Has a --verify flag to check what you already have.
  3.  Can download any missing split via the single correct URL.

Usage
─────
python download_oc20.py --verify            # check existing data
python download_oc20.py --download          # download bundle if not present
python download_oc20.py --split 10k         # print path for a specific split
"""

from __future__ import annotations
import argparse
import os
import sys
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

# ── Locate config ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, CKPT_DIR

# ── Correct single-bundle URL  ─────────────────────────────────────────────────
IS2RE_BUNDLE_URL = (
    "https://dl.fbaipublicfiles.com/opencatalystproject/data/"
    "is2res_train_val_test_lmdbs.tar.gz"
)
IS2RE_BUNDLE_DEST = DATA_DIR / "oc20" / "is2res_train_val_test_lmdbs.tar.gz"

# OCP GemNet-OC pretrained checkpoint – still alive
PRETRAINED_URL = (
    "https://dl.fbaipublicfiles.com/opencatalystproject/models/"
    "2022_09/is2re/gemnet-OC_is2re_all.pt"
)

# ── Canonical path map ────────────────────────────────────────────────────────
# After extracting is2res_train_val_test_lmdbs.tar.gz the layout is:
#   is2res_train_val_test_lmdbs/data/is2re/{10k,100k,all}/{split}/data.lmdb
OC20_BASE = DATA_DIR / "oc20" / "is2res_train_val_test_lmdbs" / "data" / "is2re"

SPLIT_MAP: dict[str, Path] = {
    # training splits
    "10k":          OC20_BASE / "10k"  / "train",
    "100k":         OC20_BASE / "100k" / "train",
    "all_train":    OC20_BASE / "all"  / "train",
    # validation
    "val_id":       OC20_BASE / "all"  / "val_id",
    "val_ood_ads":  OC20_BASE / "all"  / "val_ood_ads",
    "val_ood_cat":  OC20_BASE / "all"  / "val_ood_cat",
    "val_ood_both": OC20_BASE / "all"  / "val_ood_both",
    # test
    "test_id":      OC20_BASE / "all"  / "test_id",
    "test_ood_ads": OC20_BASE / "all"  / "test_ood_ads",
    "test_ood_cat": OC20_BASE / "all"  / "test_ood_cat",
    "test_ood_both":OC20_BASE / "all"  / "test_ood_both",
}


def resolve_lmdb_path(split: str) -> Path:
    """
    Return the Path to the data.lmdb file for *split*.
    Raises FileNotFoundError if not present on disk.
    """
    if split not in SPLIT_MAP:
        raise ValueError(f"Unknown split '{split}'. Choose from: {list(SPLIT_MAP)}")
    p = SPLIT_MAP[split] / "data.lmdb"
    if not p.exists():
        raise FileNotFoundError(
            f"LMDB not found at {p}\n"
            f"  Run:  python download_oc20.py --download   to get the bundle."
        )
    return p


def verify_splits() -> dict[str, bool]:
    """
    Check which splits are present on disk and print a table.
    Returns a dict {split_name: exists}.
    """
    status = {}
    print(f"\n{'Split':<20} {'LMDB path':<65} {'Status'}")
    print("─" * 100)
    for name, folder in SPLIT_MAP.items():
        lmdb = folder / "data.lmdb"
        exists = lmdb.exists()
        size_mb = lmdb.stat().st_size / 1e6 if exists else 0
        flag = f"✓  {size_mb:>8.1f} MB" if exists else "✗  MISSING"
        print(f"{name:<20} {str(folder):<65} {flag}")
        status[name] = exists
    present = sum(status.values())
    print(f"\n{present}/{len(status)} splits present on disk.\n")
    return status


# ── Download helpers ───────────────────────────────────────────────────────────

def _stream_download(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest.name} already downloaded ({dest.stat().st_size/1e9:.2f} GB)")
        return dest
    print(f"Downloading {url}\n  → {dest}")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            fh.write(chunk)
            bar.update(len(chunk))
    return dest


def _extract_bundle(tar_path: Path, out_dir: Path) -> None:
    marker = out_dir / ".bundle_extracted"
    if marker.exists():
        print(f"[skip] Bundle already extracted to {out_dir}")
        return
    print(f"Extracting {tar_path.name} → {out_dir}  (this can take 15–30 min)")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(path=out_dir)
    marker.touch()
    print("Extraction complete.")


def download_bundle() -> None:
    """Download the IS2RE bundle (all splits in one file) and extract it."""
    tar = _stream_download(IS2RE_BUNDLE_URL, IS2RE_BUNDLE_DEST)
    _extract_bundle(tar, DATA_DIR / "oc20")
    verify_splits()


def download_pretrained_checkpoint() -> Path:
    dest = CKPT_DIR / "gemnet_oc_is2re_all.pt"
    if dest.exists():
        print(f"[skip] Pretrained checkpoint already at {dest}")
        return dest
    _stream_download(PRETRAINED_URL, dest)
    print(f"Checkpoint: {dest}")
    return dest


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OC20 IS2RE data manager (fixed for current Meta URLs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--verify",     action="store_true",
                        help="Print a table of which splits are present on disk.")
    parser.add_argument("--download",   action="store_true",
                        help="Download the full IS2RE bundle (~8 GB) and extract it.")
    parser.add_argument("--pretrained", action="store_true",
                        help="Download GemNet-OC pretrained checkpoint (~700 MB).")
    parser.add_argument("--split",      choices=list(SPLIT_MAP),
                        help="Print the resolved LMDB path for a specific split.")
    args = parser.parse_args()

    if args.verify:
        verify_splits()

    if args.download:
        download_bundle()

    if args.pretrained:
        download_pretrained_checkpoint()

    if args.split:
        try:
            p = resolve_lmdb_path(args.split)
            print(f"{args.split}: {p}")
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)

    if not any([args.verify, args.download, args.pretrained, args.split]):
        # Default: just verify what's there
        verify_splits()


if __name__ == "__main__":
    main()