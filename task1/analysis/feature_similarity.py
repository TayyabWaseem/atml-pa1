import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from task1.data.common import get_stl10, load_images, load_json, load_config, RESULTS_DIR
from task1.data.transforms import apply_grayscale, apply_translation, apply_patch_shuffle
from task1.models.backbones import BACKBONE_NAMES, load_backbone

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def cosine_stability(f1, f2):
    """
    f1, f2: (N, D) numpy arrays
    Returns mean cosine stability.
    """
    f1_t = torch.from_numpy(f1).float()
    f2_t = torch.from_numpy(f2).float()
    cos_sim = F.cosine_similarity(f1_t, f2_t, dim=1)
    return cos_sim.mean().item()

def main():
    cfg = load_config()
    subset = load_json("data/subset_ids.json")
    test_idx = subset["test_indices"]
    
    ds = get_stl10("test", cfg)
    clean_images = load_images(ds, test_idx)
    
    print("Generating transformed images for representation stability...")
    gray_imgs = apply_grayscale(clean_images)
    shuffle_imgs = apply_patch_shuffle(clean_images, seed=cfg["seed"])
    
    # For translation, we'll average stability over the 16 pixel shifts (4 directions) to keep it simple,
    # or just use 16px shift to the right as a representative example.
    # The spec says: "For grayscale, cue conflict, translation, patch shuffling, pair each transformed image with its clean counterpart and measure the cosine stability"
    # Let's use shift=16, direction='right' for the translation stability metric
    trans_imgs = apply_translation(clean_images, shift=16, direction='right')
    
    # Load cue conflict images
    conflicts_path = REPO_ROOT / "task1" / "data" / "interventions" / "cue_conflicts" / "conflicts.json"
    has_conflicts = conflicts_path.exists()
    
    if has_conflicts:
        with open(conflicts_path, "r") as f:
            conflicts_info = json.load(f)
            
        from PIL import Image
        conflict_imgs = []
        conflict_clean_imgs = []
        
        for info in conflicts_info:
            # Transformed image
            img_path = REPO_ROOT / "task1" / "data" / "interventions" / "cue_conflicts" / info["filename"]
            conflict_imgs.append(np.array(Image.open(img_path).convert("RGB")))
            
            # The clean counterpart is the CONTENT image
            clean_idx = info["content_idx"]
            conflict_clean_imgs.append(load_images(ds, [clean_idx])[0])
            
        conflict_imgs = np.array(conflict_imgs)
        conflict_clean_imgs = np.array(conflict_clean_imgs)
    
    rows = []
    
    for name in BACKBONE_NAMES:
        print(f"\nComputing stability for {name}...")
        bb = load_backbone(name, DEVICE)
        
        # Clean features
        f_clean = bb.extract_array(clean_images, batch_size=64)
        
        # Transformed features
        f_gray = bb.extract_array(gray_imgs, batch_size=64)
        f_shuf = bb.extract_array(shuffle_imgs, batch_size=64)
        f_trans = bb.extract_array(trans_imgs, batch_size=64)
        
        stab_gray = cosine_stability(f_clean, f_gray)
        stab_shuf = cosine_stability(f_clean, f_shuf)
        stab_trans = cosine_stability(f_clean, f_trans)
        
        row = {
            "model": name,
            "grayscale_stability": stab_gray,
            "shuffle_stability": stab_shuf,
            "translation_stability": stab_trans
        }
        
        if has_conflicts:
            f_conf_clean = bb.extract_array(conflict_clean_imgs, batch_size=64)
            f_conf_trans = bb.extract_array(conflict_imgs, batch_size=64)
            stab_conf = cosine_stability(f_conf_clean, f_conf_trans)
            row["cue_conflict_stability"] = stab_conf
            
        rows.append(row)
        
        del bb
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "representation_stability.csv", index=False)
    print("\n--- Representation Stability ---")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()

