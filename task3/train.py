import argparse
import json
import math
import os
import sys
from pathlib import Path
import random

# Add the repository root to sys.path so 'shared' and 'task2' can be imported
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from shared.pacs import get_task2_loaders
from task2.models.backbone import PACSRelatedResNet
from task2.models.classifier_head import ClassifierHead
from task3.methods import dan_dg
from task3.methods.sam import SAM

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def evaluate(backbone, head, val_loaders, device):
    backbone.eval()
    head.eval()
    macro_f1s = []
    
    with torch.no_grad():
        for domain, loader in val_loaders.items():
            all_preds, all_labels = [], []
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = head(backbone(x))
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y.cpu().tolist())
            
            f1 = f1_score(all_labels, all_preds, average='macro')
            macro_f1s.append(f1)
            
    return np.mean(macro_f1s)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["dan_dg", "sam"])
    parser.add_argument("--data_root", type=str, default="data/PACS")
    parser.add_argument("--lambda_dg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=6304)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--limit_batches", type=int, default=None)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # We use get_task2_loaders but IGNORE the target loaders completely for training
    combined_source_loader, val_loaders, _, _ = get_task2_loaders(args.data_root)
    
    if args.limit_batches is not None:
        combined_source_loader.max_batches = args.limit_batches
        
    backbone = PACSRelatedResNet().to(device)
    head = ClassifierHead().to(device)
    
    # Load the ERM checkpoint as initialization
    erm_ckpt = torch.load("task2/checkpoints/source_only_grl1.0.pt", map_location=device)
    backbone.load_state_dict(erm_ckpt['backbone'])
    head.load_state_dict(erm_ckpt['head'])
    
    modules = [backbone, head]
    
    if args.method == "sam":
        # $\rho=0.05$ for standard SAM
        optimizer = SAM(
            [{'params': m.parameters()} for m in modules],
            torch.optim.AdamW,
            rho=0.05,
            lr=1e-4, weight_decay=1e-4
        )
    else:
        optimizer = torch.optim.AdamW(
            [{'params': m.parameters()} for m in modules],
            lr=1e-4, weight_decay=1e-4
        )
        
    best_f1 = -1.0
    patience_counter = 0
    patience = 5
    
    out_dir = Path("task3/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_name = f"{args.method}.pt"
    if args.method == "dan_dg":
        ckpt_name = f"dan_dg_lambda{args.lambda_dg}.pt"
        
    for epoch in range(args.max_epochs):
        backbone.train()
        head.train()
        
        epoch_losses = []
        
        for source_x, source_y in combined_source_loader:
            source_x, source_y = source_x.to(device), source_y.to(device)
            
            optimizer.zero_grad()
            
            if args.method == "dan_dg":
                feats = backbone(source_x)
                logits = head(feats)
                loss, info = dan_dg.compute_loss(feats, logits, source_y, args.lambda_dg)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                
            elif args.method == "sam":
                # First pass
                feats = backbone(source_x)
                logits = head(feats)
                loss = F.cross_entropy(logits, source_y)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                # Second pass
                feats = backbone(source_x)
                logits = head(feats)
                loss2 = F.cross_entropy(logits, source_y)
                loss2.backward()
                optimizer.second_step(zero_grad=True)
                
                epoch_losses.append(loss.item())
                
        mean_f1 = evaluate(backbone, head, val_loaders, device)
        print(f"Epoch {epoch}: Loss = {np.mean(epoch_losses):.4f}, Mean Source Val F1 = {mean_f1:.4f}")
        
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            patience_counter = 0
            torch.save({
                'backbone': backbone.state_dict(),
                'head': head.state_dict()
            }, out_dir / ckpt_name)
            print("  New best! Saved checkpoint.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    main()
