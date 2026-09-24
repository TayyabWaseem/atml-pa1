import argparse
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from shared.pacs import get_task2_loaders
from task2.models.backbone import PACSRelatedResNet
from task2.models.classifier_head import ClassifierHead
from task2.models.domain_discriminator import DomainDiscriminator
from task2.methods import source_only, dan, dann, cdan

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
            all_preds = []
            all_labels = []
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                feats = backbone(x)
                logits = head(feats)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y.cpu().tolist())
            
            f1 = f1_score(all_labels, all_preds, average='macro')
            macro_f1s.append(f1)
            
    return np.mean(macro_f1s)

def get_alpha(epoch, max_epochs, max_grl):
    # p = training progress from 0 to 1
    # We'll use epoch / max_epochs
    p = epoch / max_epochs
    alpha = (2.0 / (1.0 + math.exp(-10 * p))) - 1.0
    return alpha * max_grl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["source_only", "dan", "dann", "cdan"])
    parser.add_argument("--data_root", type=str, default="data/PACS")
    parser.add_argument("--max_grl", type=float, default=1.0, help="For controlled design study (Option B)")
    parser.add_argument("--lambda_mmd", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=6304)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    combined_source_loader, val_loaders, target_loader, _ = get_task2_loaders(args.data_root)
    
    backbone = PACSRelatedResNet().to(device)
    head = ClassifierHead().to(device)
    
    modules = [backbone, head]
    
    domain_disc = None
    if args.method == "dann":
        domain_disc = DomainDiscriminator(in_features=512).to(device)
        modules.append(domain_disc)
    elif args.method == "cdan":
        domain_disc = DomainDiscriminator(in_features=3584).to(device)
        modules.append(domain_disc)
        
    optimizer = torch.optim.AdamW([
        {'params': m.parameters()} for m in modules
    ], lr=1e-4, weight_decay=1e-4)
    
    best_f1 = -1.0
    patience_counter = 0
    max_epochs = 30
    patience = 5
    
    out_dir = Path("task2/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"{args.method}_grl{args.max_grl}.pt"
    
    for epoch in range(max_epochs):
        backbone.train() # This handles freezing BN running stats internally
        head.train()
        if domain_disc:
            domain_disc.train()
            
        target_iter = iter(target_loader)
        
        alpha = get_alpha(epoch, max_epochs, args.max_grl)
        
        epoch_losses = []
        
        for source_x, source_y in combined_source_loader:
            source_x, source_y = source_x.to(device), source_y.to(device)
            
            try:
                target_x, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_x, _ = next(target_iter)
            target_x = target_x.to(device)
            
            optimizer.zero_grad()
            
            source_feats = backbone(source_x)
            source_logits = head(source_feats)
            
            if args.method == "source_only":
                loss, info = source_only.compute_loss(source_feats, source_logits, source_y)
            elif args.method == "dan":
                target_feats = backbone(target_x)
                loss, info = dan.compute_loss(source_feats, source_logits, source_y, target_feats, args.lambda_mmd)
            elif args.method == "dann":
                target_feats = backbone(target_x)
                loss, info = dann.compute_loss(source_feats, source_logits, source_y, domain_disc, alpha, target_feats)
            elif args.method == "cdan":
                target_feats = backbone(target_x)
                target_logits = head(target_feats)
                loss, info = cdan.compute_loss(source_feats, source_logits, source_y, domain_disc, alpha, target_feats, target_logits)
                
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
        mean_f1 = evaluate(backbone, head, val_loaders, device)
        print(f"Epoch {epoch}: Loss = {np.mean(epoch_losses):.4f}, Mean Source Val F1 = {mean_f1:.4f}")
        
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            patience_counter = 0
            
            state = {
                'backbone': backbone.state_dict(),
                'head': head.state_dict(),
            }
            if domain_disc:
                state['domain_disc'] = domain_disc.state_dict()
            torch.save(state, out_dir / ckpt_name)
            print("  New best! Saved checkpoint.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
                
if __name__ == "__main__":
    main()
