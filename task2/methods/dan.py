import torch
import torch.nn.functional as F

def rbf_kernel(x, y, bandwidths):
    # x: (N, D), y: (M, D)
    dist = torch.cdist(x, y, p=2) ** 2
    kernel_val = 0
    for bw in bandwidths:
        kernel_val += torch.exp(-dist / bw)
    return kernel_val

def compute_mmd(source_features, target_features):
    combined = torch.cat([source_features, target_features], dim=0)
    
    # Compute median pairwise squared feature distance in the current combined batch
    with torch.no_grad():
        dist_sq = torch.cdist(combined, combined, p=2) ** 2
        # Use only upper triangle without diagonal to calculate median
        upper_tri_indices = torch.triu_indices(dist_sq.size(0), dist_sq.size(1), offset=1)
        pairwise_dists = dist_sq[upper_tri_indices[0], upper_tri_indices[1]]
        median_dist = torch.median(pairwise_dists)
        
    bandwidths = [0.5 * median_dist, 1.0 * median_dist, 2.0 * median_dist]
    
    xx = rbf_kernel(source_features, source_features, bandwidths)
    yy = rbf_kernel(target_features, target_features, bandwidths)
    xy = rbf_kernel(source_features, target_features, bandwidths)
    
    return xx.mean() + yy.mean() - 2 * xy.mean()

def compute_loss(features, logits, labels, target_features=None, lambda_mmd=1.0, **kwargs):
    source_len = labels.size(0)
    source_logits = logits[:source_len]
    loss_cls = F.cross_entropy(source_logits, labels)
    
    if target_features is None or len(target_features) == 0:
        return loss_cls, {"loss_cls": loss_cls.item(), "loss_mmd": 0.0}
        
    source_feats = features[:source_len]
    
    loss_mmd = compute_mmd(source_feats, target_features)
    
    total_loss = loss_cls + lambda_mmd * loss_mmd
    return total_loss, {"loss_cls": loss_cls.item(), "loss_mmd": loss_mmd.item()}
