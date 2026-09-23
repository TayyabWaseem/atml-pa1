import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# UMAP requires `pip install umap-learn`
try:
    import umap
except ImportError:
    print("UMAP not found. Please `pip install umap-learn`")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from task1.data.common import get_stl10, load_images, load_json, load_config, RESULTS_DIR
from task1.data.transforms import apply_patch_shuffle
from task1.models.backbones import BACKBONE_NAMES, load_backbone

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def plot_umap(bb_name, f_clean, f_trans, labels, class_names, title, save_path, seed=6304):
    """
    f_clean: (N, D)
    f_trans: (N, D)
    labels: (N,)
    """
    N = len(f_clean)
    X = np.vstack([f_clean, f_trans])
    
    print(f"Fitting UMAP for {bb_name}...")
    reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(X)
    
    emb_clean = embedding[:N]
    emb_trans = embedding[N:]
    
    plt.figure(figsize=(10, 8))
    
    # We have up to 10 classes in STL-10, use a categorical colormap
    cmap = plt.get_cmap("tab10")
    
    for c_idx, c_name in enumerate(class_names):
        idx = (labels == c_idx)
        # Plot clean as circles
        plt.scatter(emb_clean[idx, 0], emb_clean[idx, 1], c=[cmap(c_idx)], 
                    marker='o', label=f"{c_name} (Clean)", alpha=0.7, edgecolors='w', s=50)
        # Plot transformed as crosses/X
        plt.scatter(emb_trans[idx, 0], emb_trans[idx, 1], c=[cmap(c_idx)], 
                    marker='X', label=f"{c_name} (Transformed)", alpha=0.7, s=50)
                    
    plt.title(title)
    
    # Put legend outside
    handles, lgd_labels = plt.gca().get_legend_handles_labels()
    # To avoid huge legend, just show one pair for marker reference, and colors for classes
    # Actually, let's just make a smaller legend
    by_label = dict(zip(lgd_labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
    cfg = load_config()
    subset = load_json("data/subset_ids.json")
    test_idx = subset["test_indices"]
    yte = np.asarray(subset["test_labels"])
    class_names = subset["class_names"]
    
    ds = get_stl10("test", cfg)
    clean_images = load_images(ds, test_idx)
    
    # We will visualize Patch Shuffle as the main representative transform, 
    # as it radically changes global structure
    print("Generating Patch Shuffle images for UMAP...")
    trans_imgs = apply_patch_shuffle(clean_images, seed=cfg["seed"])
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    for name in BACKBONE_NAMES:
        print(f"\nExtracting features for {name}...")
        bb = load_backbone(name, DEVICE)
        
        f_clean = bb.extract_array(clean_images, batch_size=64)
        f_trans = bb.extract_array(trans_imgs, batch_size=64)
        
        del bb
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            
        save_path = RESULTS_DIR / f"umap_{name}_patch_shuffle.png"
        plot_umap(name, f_clean, f_trans, yte, class_names, 
                  title=f"UMAP: Clean (o) vs Patch Shuffle (X) [{name}]", 
                  save_path=save_path, 
                  seed=cfg["seed"])
        print(f"Saved {save_path}")

if __name__ == "__main__":
    main()

