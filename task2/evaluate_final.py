import argparse
import json
import torch
import sys
from pathlib import Path

# Add the repository root to sys.path so 'shared' and 'task2' can be imported
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from shared.pacs import get_task2_loaders
from task2.models.backbone import PACSRelatedResNet
from task2.models.classifier_head import ClassifierHead
from task2.evaluation.domain_separability import compute_domain_separability

def eval_model(model_name, ckpt_path, data_root, device):
    print(f"\nEvaluating {model_name}...")
    
    _, val_loaders, _, target_eval_loader = get_task2_loaders(data_root)
    
    backbone = PACSRelatedResNet().to(device)
    head = ClassifierHead().to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone.load_state_dict(ckpt['backbone'])
    head.load_state_dict(ckpt['head'])
    
    backbone.eval()
    head.eval()
    
    # 1. Source val domains performance
    source_accs = {}
    source_f1s = {}
    
    for domain, loader in val_loaders.items():
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                preds = head(backbone(x)).argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y.cpu().tolist())
                
        source_accs[domain] = accuracy_score(all_labels, all_preds) * 100
        source_f1s[domain] = f1_score(all_labels, all_preds, average='macro') * 100
        
    mean_source_acc = np.mean(list(source_accs.values()))
    mean_source_f1 = np.mean(list(source_f1s.values()))
    
    # 2. Target performance
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in target_eval_loader:
            x, y = x.to(device), y.to(device)
            preds = head(backbone(x)).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
            
    target_acc = accuracy_score(all_labels, all_preds) * 100
    target_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    
    # Per-class target accuracy
    classes = sorted(["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"])
    cm = confusion_matrix(all_labels, all_preds, labels=range(7))
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    per_class_dict = {classes[i]: acc for i, acc in enumerate(per_class_acc)}
    
    # 3. Domain Separability
    domain_sep = compute_domain_separability(backbone, val_loaders, target_eval_loader, device)
    
    return {
        "source_accs": source_accs,
        "source_f1s": source_f1s,
        "mean_source_acc": mean_source_acc,
        "mean_source_f1": mean_source_f1,
        "target_acc": target_acc,
        "target_f1": target_f1,
        "domain_separability": domain_sep,
        "per_class_target_acc": per_class_dict,
        "confusion_matrix": cm.tolist()
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/PACS")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    methods = [
        ("Source-only", "task2/checkpoints/source_only_grl1.0.pt"),
        ("DAN", "task2/checkpoints/dan_grl1.0.pt"),
        ("DANN", "task2/checkpoints/dann_grl1.0.pt"),
        ("CDAN", "task2/checkpoints/cdan_grl1.0.pt"),
    ]
    
    # Also evaluate controlled study DANNs if they exist
    for grl in [0.25, 0.5]:
        path = f"task2/checkpoints/dann_grl{grl}.pt"
        if Path(path).exists():
            methods.append((f"DANN_grl{grl}", path))
            
    results = {}
    
    for name, path in methods:
        if not Path(path).exists():
            print(f"Skipping {name}, checkpoint {path} not found.")
            continue
            
        res = eval_model(name, path, args.data_root, device)
        
        if "Source-only" in results:
            res["target_acc_change"] = res["target_acc"] - results["Source-only"]["target_acc"]
            
            # Print class-specific improvement/degradation relative to source-only
            print(f"\n--- {name} vs Source-only Class Differences ---")
            diffs = {cls: res["per_class_target_acc"][cls] - results["Source-only"]["per_class_target_acc"][cls] 
                     for cls in res["per_class_target_acc"]}
            print("Largest Improvements:", sorted(diffs.items(), key=lambda x: x[1], reverse=True)[:2])
            print("Largest Degradations:", sorted(diffs.items(), key=lambda x: x[1])[:2])
            
        else:
            res["target_acc_change"] = 0.0
            
        print(f"\n{name} Results:")
        print(f"  Mean Source Acc: {res['mean_source_acc']:.2f}% | Target Acc: {res['target_acc']:.2f}%")
        print(f"  Domain Separability: {res['domain_separability']:.2f}%")
        
        results[name] = res
        
    out_dir = Path("task2/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
