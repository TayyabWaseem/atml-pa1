import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from task4.scores import scores
from sklearn.metrics import roc_curve, auc

def load_scores(method, score_fn, score_name):
    cache_dir = Path(f"task4/cache/{method}")
    if not cache_dir.exists(): return None, None
    
    test_logits = np.load(cache_dir / "test_logits.npy")
    near_logits = np.load(cache_dir / "near_logits.npy")
    far_logits = np.load(cache_dir / "far_logits.npy")
    
    unknown_logits = np.concatenate([near_logits, far_logits])
    
    if score_name == "Mahalanobis":
        train_feats = np.load(cache_dir / "train_unaug_features.npy")
        train_labels = np.load(cache_dir / "train_unaug_labels.npy")
        means, inv_cov = scores.fit_mahalanobis(train_feats, train_labels)
        
        test_feats = np.load(cache_dir / "test_features.npy")
        near_feats = np.load(cache_dir / "near_features.npy")
        far_feats = np.load(cache_dir / "far_features.npy")
        unknown_feats = np.concatenate([near_feats, far_feats])
        
        known_scores = score_fn(test_feats, means, inv_cov)
        unknown_scores = score_fn(unknown_feats, means, inv_cov)
    else:
        known_scores = score_fn(test_logits)
        unknown_scores = score_fn(unknown_logits)
        
    return known_scores, unknown_scores

def plot_roc_curves():
    methods_scores = [
        ("Vanilla", "MSP", scores.score_msp),
        ("Vanilla", "MLS", scores.score_mls),
        ("Vanilla", "Energy", scores.score_energy),
        ("Vanilla", "Mahalanobis", scores.score_mahalanobis),
        ("GCSC", "MLS", scores.score_mls),
        ("PROSER", "Placeholder", scores.score_proser_placeholder)
    ]
    
    plt.figure(figsize=(10, 8))
    
    for method, score_name, fn in methods_scores:
        try:
            known, unknown = load_scores(method, fn, score_name)
            if known is None: continue
            
            y_true = np.concatenate([np.zeros(len(known)), np.ones(len(unknown))])
            y_scores = np.concatenate([known, unknown])
            
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            label = f"{method} - {score_name} (AUC = {roc_auc:.3f})"
            plt.plot(fpr, tpr, lw=2, label=label)
        except Exception as e:
            print(f"Skipping plot for {method} {score_name}: {e}")
            
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Open-Set Recognition (All Unknowns)')
    plt.legend(loc="lower right")
    
    out_dir = Path("task4/results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "roc_curves.png", dpi=300)
    plt.close()

def plot_score_distributions():
    # Let's plot histograms for Vanilla MLS (Known vs Near vs Far)
    method = "Vanilla"
    cache_dir = Path(f"task4/cache/{method}")
    if not cache_dir.exists(): return
    
    test_logits = np.load(cache_dir / "test_logits.npy")
    near_logits = np.load(cache_dir / "near_logits.npy")
    far_logits = np.load(cache_dir / "far_logits.npy")
    
    known = scores.score_mls(test_logits)
    near = scores.score_mls(near_logits)
    far = scores.score_mls(far_logits)
    
    plt.figure(figsize=(10, 6))
    plt.hist(known, bins=50, alpha=0.5, label='Known (CIFAR-10)', density=True)
    plt.hist(near, bins=50, alpha=0.5, label='Near Unknowns', density=True)
    plt.hist(far, bins=50, alpha=0.5, label='Far Unknowns', density=True)
    
    plt.xlabel('Unknownness Score (MLS)')
    plt.ylabel('Density')
    plt.title('Score Distributions (Vanilla - MLS)')
    plt.legend(loc='upper right')
    
    out_dir = Path("task4/results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "score_distributions_mls.png", dpi=300)
    plt.close()

def main():
    print("Generating OSR plots...")
    plot_roc_curves()
    plot_score_distributions()
    print("Plots saved to task4/results/figures/")

if __name__ == "__main__":
    main()
