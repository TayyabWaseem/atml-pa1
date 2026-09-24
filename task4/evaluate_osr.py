import numpy as np
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from task4.scores import scores

def get_auroc(known_scores, unknown_scores):
    # known = 0, unknown = 1
    y_true = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    y_scores = np.concatenate([known_scores, unknown_scores])
    return roc_auc_score(y_true, y_scores)

def get_fpr95(known_scores, unknown_scores):
    tau = np.percentile(known_scores, 95)
    # FPR = FP / (FP + TN)
    # FP = unknowns incorrectly accepted = unknown_scores <= tau
    # Total unknowns = len(unknown_scores)
    fpr = np.mean(unknown_scores <= tau)
    return fpr, tau

def evaluate_method_score(method, score_name, get_score_fn):
    cache_dir = Path(f"task4/cache/{method}")
    
    val_logits = np.load(cache_dir / "val_logits.npy")
    test_logits = np.load(cache_dir / "test_logits.npy")
    near_logits = np.load(cache_dir / "near_logits.npy")
    far_logits = np.load(cache_dir / "far_logits.npy")
    
    if score_name == "Mahalanobis":
        train_feats = np.load(cache_dir / "train_unaug_features.npy")
        train_labels = np.load(cache_dir / "train_unaug_labels.npy")
        means, inv_cov = scores.fit_mahalanobis(train_feats, train_labels)
        
        val_feats = np.load(cache_dir / "val_features.npy")
        test_feats = np.load(cache_dir / "test_features.npy")
        near_feats = np.load(cache_dir / "near_features.npy")
        far_feats = np.load(cache_dir / "far_features.npy")
        
        val_u = get_score_fn(val_feats, means, inv_cov)
        test_u = get_score_fn(test_feats, means, inv_cov)
        near_u = get_score_fn(near_feats, means, inv_cov)
        far_u = get_score_fn(far_feats, means, inv_cov)
    else:
        val_u = get_score_fn(val_logits)
        test_u = get_score_fn(test_logits)
        near_u = get_score_fn(near_logits)
        far_u = get_score_fn(far_logits)
        
    auroc_near = get_auroc(test_u, near_u)
    auroc_far = get_auroc(test_u, far_u)
    auroc_all = get_auroc(test_u, np.concatenate([near_u, far_u]))
    
    _, tau = get_fpr95(val_u, val_u) # Threshold based on val
    acc_rate = np.mean(test_u <= tau)
    fpr_near = np.mean(near_u <= tau)
    fpr_far = np.mean(far_u <= tau)
    
    # Closed-set Accuracy (CSA) on test (using first 10 logits)
    test_labels = np.load(cache_dir / "test_labels.npy")
    preds = np.argmax(test_logits[:, :10], axis=1)
    csa = accuracy_score(test_labels, preds)
    
    return {
        "AUROC_Near": auroc_near * 100,
        "AUROC_Far": auroc_far * 100,
        "AUROC_All": auroc_all * 100,
        "Acceptance_Rate": acc_rate * 100,
        "FPR95_Near": fpr_near * 100,
        "FPR95_Far": fpr_far * 100,
        "CSA": csa * 100,
        "tau": float(tau)
    }

def main():
    results = {}
    
    # 1. Compare MSP, MLS, Energy, Mahalanobis on Vanilla
    methods_to_score = [
        ("Vanilla", "MSP", scores.score_msp),
        ("Vanilla", "MLS", scores.score_mls),
        ("Vanilla", "Energy", scores.score_energy),
        ("Vanilla", "Mahalanobis", scores.score_mahalanobis)
    ]
    
    for method, score_name, fn in methods_to_score:
        try:
            res = evaluate_method_score(method, score_name, fn)
            key = f"{method}_{score_name}"
            results[key] = res
            print(f"[{key}] AUROC All: {res['AUROC_All']:.2f}% | FPR95 All: {(res['FPR95_Near']+res['FPR95_Far'])/2:.2f}% | CSA: {res['CSA']:.2f}%")
        except Exception as e:
            print(f"Skipping {method} {score_name}: {e}")
            
    # 2. Compare Vanilla, GCSC, PROSER on MLS
    for method in ["GCSC", "PROSER"]:
        try:
            res = evaluate_method_score(method, "MLS", scores.score_mls)
            key = f"{method}_MLS"
            results[key] = res
            print(f"[{key}] AUROC All: {res['AUROC_All']:.2f}% | FPR95 All: {(res['FPR95_Near']+res['FPR95_Far'])/2:.2f}% | CSA: {res['CSA']:.2f}%")
            
            if method == "PROSER":
                res = evaluate_method_score("PROSER", "Placeholder", scores.score_proser_placeholder)
                key = "PROSER_Placeholder"
                results[key] = res
                print(f"[{key}] AUROC All: {res['AUROC_All']:.2f}% | FPR95 All: {(res['FPR95_Near']+res['FPR95_Far'])/2:.2f}% | CSA: {res['CSA']:.2f}%")
        except Exception as e:
            print(f"Skipping {method}: {e}")
            
    out_dir = Path("task4/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
