import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from task4.data.datasets import get_cifar_loaders
from task4.models.resnet_cifar import ResNet18CIFAR
from task4.methods.proser import PROSER

def extract_features(model, loader, device, is_proser=False):
    model.eval()
    all_logits = []
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if is_proser:
                logits, features, _ = model(x, return_features=True)
            else:
                logits, features, _ = model(x, return_features=True)
                
            all_logits.append(logits.cpu().numpy())
            all_features.append(features.cpu().numpy())
            all_labels.append(y.numpy())
            
    return np.concatenate(all_logits), np.concatenate(all_features), np.concatenate(all_labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["vanilla", "gcsc", "proser"])
    parser.add_argument("--data_root", type=str, default="data")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaders = get_cifar_loaders(args.data_root, method="vanilla") # Use vanilla transforms for evaluation extraction
    
    base_model = ResNet18CIFAR(num_classes=10).to(device)
    
    ckpt_path = Path("task4/checkpoints") / f"{args.method}.pt"
    if args.method == "proser":
        ckpt_path = Path("task4/checkpoints") / "proser_full.pt"
        
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return
        
    if args.method == "proser":
        model = PROSER(base_model, num_dummy=5).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        model = base_model
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
    out_dir = Path(f"task4/cache/{args.method}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    splits_to_extract = ["train_unaug", "val", "test", "near", "far"]
    
    for split in splits_to_extract:
        print(f"Extracting {split} for {args.method}...")
        logits, features, labels = extract_features(model, loaders[split], device, is_proser=(args.method=="proser"))
        np.save(out_dir / f"{split}_logits.npy", logits)
        np.save(out_dir / f"{split}_features.npy", features)
        np.save(out_dir / f"{split}_labels.npy", labels)
        
    print(f"Extraction for {args.method} complete.")

if __name__ == "__main__":
    main()
