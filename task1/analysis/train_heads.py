"""
train_heads.py  (Task Step 1: clean baseline)

For each frozen backbone (ResNet-50, ViT-B/16, CLIP ViT-B/32):
  1. Extract + cache features for train / val / clean test subset.
  2. Train ONE linear head on the cached features:
       AdamW, lr 1e-3, weight decay 1e-4, <= 50 epochs,
       early stopping after 5 epochs without improved val accuracy, seed 6304.
  3. Evaluate on the clean 500-image test subset:
       top-1 accuracy, macro-F1, mean maximum confidence.
Also evaluates CLIP zero-shot with the fixed prompt "a photo of a {class}."

Run from the repo root:
    python task1/analysis/train_heads.py
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on path

from task1.data.common import (CKPT_DIR, FEATURE_DIR, RESULTS_DIR, get_stl10,
                               load_config, load_images, load_json, set_seed)
from task1.models.backbones import BACKBONE_NAMES, load_backbone

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- features
def get_features(bb, ds, indices, cache_name, chunk=500, batch_size=64):
    """Extract features in chunks (keeps RAM small) and cache them to disk."""
    path = FEATURE_DIR / f"{bb.name}_{cache_name}.npy"
    if path.exists():
        return np.load(path)
    feats = []
    for i in range(0, len(indices), chunk):
        imgs = load_images(ds, indices[i:i + chunk])
        feats.append(bb.extract_array(imgs, batch_size=batch_size))
    feats = np.concatenate(feats, axis=0)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, feats)
    return feats


# ---------------------------------------------------------------- head
def train_head(Xtr, ytr, Xva, yva, n_classes, seed, cfg):
    set_seed(seed)
    lr = cfg.get("head_lr", 1e-3)
    wd = cfg.get("head_weight_decay", 1e-4)
    max_epochs = cfg.get("head_max_epochs", 50)
    patience = cfg.get("head_patience", 5)
    bs = cfg.get("head_batch_size", 128)

    Xtr, Xva = torch.from_numpy(Xtr).float().to(DEVICE), torch.from_numpy(Xva).float().to(DEVICE)
    ytr, yva = torch.from_numpy(ytr).long().to(DEVICE), torch.from_numpy(yva).long().to(DEVICE)

    head = nn.Linear(Xtr.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    best_acc, best_state, best_epoch, bad = -1.0, None, 0, 0
    log = []
    for epoch in range(1, max_epochs + 1):
        head.train()
        perm = torch.randperm(len(Xtr), generator=gen).to(DEVICE)
        total = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(head(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

        head.eval()
        with torch.no_grad():
            val_acc = (head(Xva).argmax(1) == yva).float().mean().item()
        log.append({"epoch": epoch, "train_loss": total / len(Xtr), "val_acc": val_acc})

        if val_acc > best_acc:
            best_acc, best_state, best_epoch, bad = val_acc, copy.deepcopy(head.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break

    head.load_state_dict(best_state)  # restore best-validation weights
    head.eval()
    return head, {"best_val_acc": best_acc, "best_epoch": best_epoch,
                  "epochs_run": len(log), "log": log}


# ---------------------------------------------------------------- metrics
def compute_metrics(logits, y):
    """logits: (N, C) tensor/array, y: (N,) labels."""
    logits = torch.as_tensor(logits).float().cpu()
    probs = logits.softmax(dim=1)
    pred = probs.argmax(1).numpy()
    return {
        "top1_acc": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "mean_max_conf": float(probs.max(1).values.mean()),
    }


# ---------------------------------------------------------------- main
def main():
    cfg = load_config()
    seed = cfg["seed"]

    split = load_json("data/train_val_split.json")
    subset = load_json("data/subset_ids.json")
    class_names = subset["class_names"]
    n_classes = len(class_names)

    train_ds, test_ds = get_stl10("train", cfg), get_stl10("test", cfg)
    train_idx, val_idx = split["train_indices"], split["val_indices"]
    test_idx = subset["test_indices"]

    ytr = np.asarray(train_ds.labels)[train_idx]
    yva = np.asarray(train_ds.labels)[val_idx]
    yte = np.asarray(subset["test_labels"])

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows, train_logs = [], {}
    for name in BACKBONE_NAMES:
        print(f"\n=== {name} ===")
        bb = load_backbone(name, DEVICE)

        Xtr = get_features(bb, train_ds, train_idx, "train")
        Xva = get_features(bb, train_ds, val_idx, "val")
        Xte = get_features(bb, test_ds, test_idx, "test_clean")
        print(f"features: train {Xtr.shape}, val {Xva.shape}, test {Xte.shape}")

        head, info = train_head(Xtr, ytr, Xva, yva, n_classes, seed, cfg)
        print(f"best val acc {info['best_val_acc']:.4f} at epoch {info['best_epoch']} "
              f"({info['epochs_run']} epochs run)")
        torch.save(head.state_dict(), CKPT_DIR / f"{name}_head.pt")
        train_logs[name] = info

        with torch.no_grad():
            logits = head(torch.from_numpy(Xte).float().to(DEVICE))
        m = compute_metrics(logits, yte)
        rows.append({"model": f"{name}_linear_head", **m})
        print("clean test:", m)

        if name == "clip":
            zs = bb.zero_shot_logits_from_features(torch.from_numpy(Xte), class_names,
                                                   prompt="a photo of a {}.")
            mz = compute_metrics(zs, yte)
            rows.append({"model": "clip_zero_shot", **mz})
            print("clip zero-shot:", mz)

        del bb
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "clean_baseline.csv", index=False)
    with open(RESULTS_DIR / "head_training_log.json", "w") as f:
        json.dump(train_logs, f, indent=2)
    print("\n", df.to_string(index=False))
    print(f"\nSaved {RESULTS_DIR / 'clean_baseline.csv'}")


if __name__ == "__main__":
    main()