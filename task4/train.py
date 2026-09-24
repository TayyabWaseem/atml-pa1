import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from task4.data.datasets import get_cifar_loaders
from task4.models.resnet_cifar import ResNet18CIFAR
from task4.methods.vanilla import compute_loss as compute_vanilla_loss
from task4.methods.proser import PROSER, compute_classifier_placeholder_loss, compute_data_placeholder_loss

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def evaluate(model, loader, device, is_proser=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if is_proser:
                logits = logits[:, :10] # Only evaluate on known classes
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total * 100.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["vanilla", "gcsc", "proser"])
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--seed", type=int, default=6304)
    parser.add_argument("--max_epochs", type=int, default=100) # Vanilla and GCSC use 100, PROSER uses 50
    parser.add_argument("--limit_batches", type=int, default=None)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    loaders = get_cifar_loaders(args.data_root, method=args.method)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    
    out_dir = Path("task4/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    base_model = ResNet18CIFAR(num_classes=10).to(device)
    
    if args.method == "proser":
        args.max_epochs = 50 # Override for PROSER
        # Initialize PROSER from Vanilla checkpoint
        vanilla_ckpt_path = out_dir / "vanilla.pt"
        if not vanilla_ckpt_path.exists():
            print("Error: PROSER requires the Vanilla checkpoint. Run vanilla first.")
            return
        base_model.load_state_dict(torch.load(vanilla_ckpt_path, map_location=device))
        
        model = PROSER(base_model, num_dummy=5).to(device)
        
        # PROSER specific hyperparameters
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    else:
        model = base_model
        # Vanilla / GCSC hyperparameters
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        
    best_acc = -1.0
    ckpt_name = f"{args.method}.pt"
    
    for epoch in range(args.max_epochs):
        model.train()
        epoch_losses = []
        
        for i, (x, y) in enumerate(train_loader):
            if args.limit_batches and i >= args.limit_batches: break
            
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            if args.method == "proser":
                # Split batch in half
                half = x.size(0) // 2
                x_cp, y_cp = x[:half], y[:half]
                x_dp, y_dp = x[half:], y[half:]
                
                # Classifier placeholder loss
                logits_cp = model(x_cp)
                loss_cp = compute_classifier_placeholder_loss(logits_cp, y_cp, beta=1.0)
                
                # Data placeholder loss (manifold mixup between layer2 and layer3)
                # We need to run x_dp up to layer2, mix, and run the rest.
                # In ResNet18CIFAR, layer2 output is 'h'
                _, _, h_dp = model(x_dp, return_features=True)
                
                # Mix representations belonging to different classes
                # For simplicity, we just shift the batch by 1 to mix with a different example
                # (Assuming batch is shuffled, mostly different classes)
                idx = torch.randperm(h_dp.size(0)).to(device)
                
                # Sample lambda from Beta(2, 2)
                lam = np.random.beta(2, 2)
                
                h_mixed = lam * h_dp + (1 - lam) * h_dp[idx]
                
                # Pass h_mixed through the rest of the network
                out = model.base_model.model.layer3(h_mixed)
                out = model.base_model.model.layer4(out)
                out = model.base_model.model.avgpool(out)
                feats_mixed = torch.flatten(out, 1)
                logits_mixed = model.base_model.model.fc(feats_mixed)
                
                # gamma = 0.1 for data placeholder loss
                loss_dp = compute_data_placeholder_loss(logits_mixed) * 0.1
                
                loss = loss_cp + loss_dp
            else:
                logits = model(x)
                loss, _ = compute_vanilla_loss(logits, y)
                
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            
        scheduler.step()
        
        val_acc = evaluate(model, val_loader, device, is_proser=(args.method=="proser"))
        print(f"Epoch {epoch}: Loss = {np.mean(epoch_losses):.4f}, Val Acc = {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            if args.method == "proser":
                torch.save(model.base_model.state_dict(), out_dir / ckpt_name) # Save base model for apples-to-apples load
                torch.save(model.state_dict(), out_dir / "proser_full.pt") # Save full model for placeholder evaluation
            else:
                torch.save(model.state_dict(), out_dir / ckpt_name)
            print("  New best! Saved checkpoint.")

if __name__ == "__main__":
    main()
