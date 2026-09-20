"""
make_subset.py

1. Downloads STL-10 (if needed).
2. Makes a stratified 80/20 train/val split of the OFFICIAL TRAIN partition.
3. Selects a class-balanced subset of official TEST images.
4. Saves all indices to JSON so every later script uses the exact same images.

Run from the repo root (PA1):
    python task1/data/make_subset.py
"""

import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from torchvision.datasets import STL10

# Repo root = PA1/  (this file lives in PA1/task1/data/)
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "task1" / "configs" / "task1.yaml"
OUT_DIR = REPO_ROOT / "task1" / "data"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def stratified_train_val(labels, seed, val_fraction=0.2):
    """Stratified split of the official training partition. Returns sorted index lists."""
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        stratify=labels,
        random_state=seed,
    )
    return sorted(train_idx.tolist()), sorted(val_idx.tolist())


def balanced_test_subset(labels, n_total, seed, num_classes):
    """
    Pick n_total // num_classes images per class from the test set.
    If a class has too few images, use all of them and record the shortfall.
    """
    rng = np.random.RandomState(seed)
    per_class = n_total // num_classes

    selected = []
    per_class_counts = {}
    imbalance_notes = []

    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        if len(class_indices) >= per_class:
            chosen = rng.choice(class_indices, size=per_class, replace=False)
        else:
            chosen = class_indices
            imbalance_notes.append(
                f"class {c}: only {len(class_indices)} available, wanted {per_class}"
            )
        per_class_counts[c] = int(len(chosen))
        selected.extend(chosen.tolist())

    return sorted(selected), per_class_counts, imbalance_notes


def main():
    cfg = load_config()
    seed = cfg["seed"]
    n_test = cfg["n_test_images"]
    data_root = REPO_ROOT / cfg["data_root"]
    data_root.mkdir(parents=True, exist_ok=True)

    # --- Load datasets (labels only are needed here, but this downloads the images) ---
    train_ds = STL10(root=str(data_root), split="train", download=True)
    test_ds = STL10(root=str(data_root), split="test", download=True)

    class_names = list(train_ds.classes)
    num_classes = len(class_names)
    train_labels = np.array(train_ds.labels)
    test_labels = np.array(test_ds.labels)

    print(f"Train (official): {len(train_ds)} images | Test (official): {len(test_ds)} images")
    print(f"Classes: {class_names}")

    # --- 1. Stratified 80/20 train/val split ---
    train_idx, val_idx = stratified_train_val(train_labels, seed)
    split = {
        "seed": seed,
        "class_names": class_names,
        "train_indices": train_idx,
        "val_indices": val_idx,
    }
    with open(OUT_DIR / "train_val_split.json", "w") as f:
        json.dump(split, f)
    print(f"Train/val split: {len(train_idx)} train, {len(val_idx)} val")

    # --- 2. Class-balanced test subset ---
    test_idx, per_class_counts, notes = balanced_test_subset(
        test_labels, n_test, seed, num_classes
    )
    subset = {
        "seed": seed,
        "class_names": class_names,
        "source_split": "official STL-10 test",
        "n_images": len(test_idx),
        "per_class_counts": {class_names[c]: n for c, n in per_class_counts.items()},
        "imbalance_notes": notes,
        "test_indices": test_idx,
        "test_labels": [int(test_labels[i]) for i in test_idx],
    }
    with open(OUT_DIR / "subset_ids.json", "w") as f:
        json.dump(subset, f, indent=2)

    print(f"Test subset: {len(test_idx)} images")
    print("Per-class counts:", subset["per_class_counts"])
    if notes:
        print("Imbalance (document this in your README):")
        for n in notes:
            print("  -", n)
    else:
        print("Subset is perfectly class-balanced.")

    print(f"\nSaved:\n  {OUT_DIR / 'train_val_split.json'}\n  {OUT_DIR / 'subset_ids.json'}")


if __name__ == "__main__":
    main()