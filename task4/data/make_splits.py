import os
import json
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path

def main():
    seed = 6304
    data_root = "data"
    os.makedirs(data_root, exist_ok=True)
    
    # CIFAR-10 90/10 Split
    c10 = CIFAR10(root=data_root, train=True, download=True)
    targets = np.array(c10.targets)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))
    
    c10_splits = {
        "train": train_idx.tolist(),
        "val": val_idx.tolist()
    }
    
    out_dir = Path("task4/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cifar10_splits.json", "w") as f:
        json.dump(c10_splits, f)
        
    # CIFAR-100 Unknowns
    c100 = CIFAR100(root=data_root, train=False, download=True)
    c100_classes = c100.classes
    
    near_classes = ["bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel"]
    far_classes = ["bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe"]
    
    near_idx = []
    far_idx = []
    
    for i, target in enumerate(c100.targets):
        cls_name = c100_classes[target]
        if cls_name in near_classes:
            near_idx.append(i)
        elif cls_name in far_classes:
            far_idx.append(i)
            
    c100_splits = {
        "near": near_idx,
        "far": far_idx
    }
    
    with open(out_dir / "cifar100_unknowns.json", "w") as f:
        json.dump(c100_splits, f)
        
    print(f"Saved CIFAR-10 splits (train: {len(train_idx)}, val: {len(val_idx)})")
    print(f"Saved CIFAR-100 unknowns (near: {len(near_idx)}, far: {len(far_idx)})")

if __name__ == "__main__":
    main()
