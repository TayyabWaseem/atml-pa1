import torch
import torch.nn.functional as F

def rbf_kernel(x, y, bandwidths):
    dist = torch.cdist(x, y, p=2) ** 2
    kernel_val = 0
    for bw in bandwidths:
        kernel_val += torch.exp(-dist / bw)
    return kernel_val

def compute_mmd(features_a, features_b):
    combined = torch.cat([features_a, features_b], dim=0)
    
    with torch.no_grad():
        dist_sq = torch.cdist(combined, combined, p=2) ** 2
        upper_tri_indices = torch.triu_indices(dist_sq.size(0), dist_sq.size(1), offset=1)
        pairwise_dists = dist_sq[upper_tri_indices[0], upper_tri_indices[1]]
        median_dist = torch.median(pairwise_dists)
        
    bandwidths = [0.5 * median_dist, 1.0 * median_dist, 2.0 * median_dist]
    
    xx = rbf_kernel(features_a, features_a, bandwidths)
    yy = rbf_kernel(features_b, features_b, bandwidths)
    xy = rbf_kernel(features_a, features_b, bandwidths)
    
    return xx.mean() + yy.mean() - 2 * xy.mean()

def compute_loss(features, logits, labels, lambda_dg=1.0, **kwargs):
    loss_cls = F.cross_entropy(logits, labels)
    
    # We know the batch contains 8 examples from each source domain in order: Photo, Art, Cartoon
    # Total features = 24.
    chunk_size = features.size(0) // 3
    
    feat_P = features[0 * chunk_size : 1 * chunk_size]
    feat_A = features[1 * chunk_size : 2 * chunk_size]
    feat_C = features[2 * chunk_size : 3 * chunk_size]
    
    mmd_PA = compute_mmd(feat_P, feat_A)
    mmd_PC = compute_mmd(feat_P, feat_C)
    mmd_AC = compute_mmd(feat_A, feat_C)
    
    loss_mmd = (mmd_PA + mmd_PC + mmd_AC) / 3.0
    
    total_loss = loss_cls + lambda_dg * loss_mmd
    return total_loss, {"loss_cls": loss_cls.item(), "loss_mmd": loss_mmd.item()}
