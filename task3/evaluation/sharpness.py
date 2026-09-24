import torch
import torch.nn.functional as F
import numpy as np

def compute_sharpness(backbone, head, source_val_loaders, device, seed=6304):
    """
    Selects a fixed validation batch containing 32 examples from each source (total 96).
    Places model in eval mode.
    Computes \Delta_sharp = L_val(theta + epsilon) - L_val(theta)
    where epsilon = 0.05 * (grad / ||grad||_2)
    """
    backbone.eval()
    head.eval()
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    x_batch = []
    y_batch = []
    
    for domain, loader in source_val_loaders.items():
        # Just grab the first batch (which is 64 elements) and take 32
        for x, y in loader:
            x_batch.append(x[:32])
            y_batch.append(y[:32])
            break
            
    x_batch = torch.cat(x_batch, dim=0).to(device)
    y_batch = torch.cat(y_batch, dim=0).to(device)
    
    # Enable gradients to compute the perturbation
    backbone.zero_grad()
    head.zero_grad()
    
    # Need requires_grad=True internally for parameters, they are already True.
    # Just need to compute loss and backward.
    logits = head(backbone(x_batch))
    loss_val = F.cross_entropy(logits, y_batch)
    loss_val.backward()
    
    # Compute epsilon and apply perturbation
    radius = 0.05
    grad_norm = torch.norm(
        torch.stack([p.grad.norm(p=2) for m in (backbone, head) for p in m.parameters() if p.grad is not None]),
        p=2
    )
    
    scale = radius / (grad_norm + 1e-12)
    
    # Store original params and perturb
    with torch.no_grad():
        for m in (backbone, head):
            for p in m.parameters():
                if p.grad is not None:
                    p.add_(p.grad * scale)
                    
    # Compute new loss
    backbone.zero_grad()
    head.zero_grad()
    with torch.no_grad():
        logits_adv = head(backbone(x_batch))
        loss_adv = F.cross_entropy(logits_adv, y_batch)
        
    delta_sharp = loss_adv.item() - loss_val.item()
    
    # Restore original params
    with torch.no_grad():
        for m in (backbone, head):
            for p in m.parameters():
                if p.grad is not None:
                    p.sub_(p.grad * scale)
                    
    return delta_sharp
