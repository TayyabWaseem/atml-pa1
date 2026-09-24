import json
import random
from pathlib import Path
import os

def create_pacs_splits(data_root="data/PACS", seed=6304, out_path="shared/splits/pacs_sketch_seed6304.json"):
    """
    Creates stratified 80/20 train/val splits for the source domains (Photo, Art_Painting, Cartoon)
    using seed 6304. Leaves Sketch as target (no split needed, all target examples used).
    """
    random.seed(seed)
    
    domains = ["photo", "art_painting", "cartoon"]
    splits = {"photo": {"train": [], "val": []},
              "art_painting": {"train": [], "val": []},
              "cartoon": {"train": [], "val": []}}
    
    root = Path(data_root)
    if not root.exists():
        print(f"Warning: {root} does not exist. Please place PACS dataset there.")
        return
        
    for domain in domains:
        domain_dir = root / domain
        if not domain_dir.exists():
            continue
            
        classes = sorted([d.name for d in domain_dir.iterdir() if d.is_dir()])
        
        for cls in classes:
            cls_dir = domain_dir / cls
            images = sorted([f.name for f in cls_dir.iterdir() if f.is_file()])
            
            # Shuffle with the fixed seed
            random.shuffle(images)
            
            # Stratified 80/20 split
            split_idx = int(len(images) * 0.8)
            train_imgs = images[:split_idx]
            val_imgs = images[split_idx:]
            
            # Store relative paths: class/image.jpg
            splits[domain]["train"].extend([f"{cls}/{img}" for img in train_imgs])
            splits[domain]["val"].extend([f"{cls}/{img}" for img in val_imgs])
            
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Splits saved to {out_file}")

if __name__ == "__main__":
    create_pacs_splits()
