"""
common.py

Small helpers shared by every script, so all of them load config, seeds and
images in exactly the same way.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.datasets import STL10

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = REPO_ROOT / "task1"
CONFIG_PATH = TASK_DIR / "configs" / "task1.yaml"
FEATURE_DIR = TASK_DIR / "features"      # gitignored: cached features
CKPT_DIR = TASK_DIR / "checkpoints"      # gitignored: trained heads
RESULTS_DIR = TASK_DIR / "results"       # committed: tables and figures


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(relative_path):
    """Load a JSON file relative to the task1/ folder, e.g. 'data/subset_ids.json'."""
    with open(TASK_DIR / relative_path, "r") as f:
        return json.load(f)


def get_stl10(split, cfg):
    return STL10(root=str(REPO_ROOT / cfg["data_root"]), split=split, download=True)


def load_images(ds, indices, size=224):
    """
    Returns (n, size, size, 3) uint8 RGB images for the given dataset indices.
    STL-10 is stored 96x96, so we upsample ONCE here to the common 224x224 that
    every intervention and every model then shares.
    """
    arr = ds.data[np.asarray(indices)]  # (n, 3, 96, 96) uint8
    out = np.empty((len(arr), size, size, 3), dtype=np.uint8)
    for k, a in enumerate(arr):
        img = Image.fromarray(a.transpose(1, 2, 0))
        out[k] = np.asarray(img.resize((size, size), Image.BICUBIC))
    return out