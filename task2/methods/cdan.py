import torch
import torch.nn.functional as F

def compute_loss(features, logits, labels, domain_discriminator, alpha, target_features=None, target_logits=None, **kwargs):
    source_len = labels.size(0)
    source_logits = logits[:source_len]
    loss_cls = F.cross_entropy(source_logits, labels)
    
    if target_features is None or len(target_features) == 0:
        return loss_cls, {"loss_cls": loss_cls.item(), "loss_domain": 0.0}
        
    source_feats = features[:source_len]
    
    combined_feats = torch.cat([source_feats, target_features], dim=0)
    combined_logits = torch.cat([source_logits, target_logits], dim=0)
    
    # g(x) = vec(f ⊗ p)
    # p = C(f) = softmax(logits)
    p = F.softmax(combined_logits, dim=1) # (N, C)
    
    # Outer product: f (N, D), p (N, C) -> (N, D, 1) * (N, 1, C) -> (N, D, C) -> (N, D*C)
    # Wait, the instruction says "Feed g(x) to a discriminator with the SAME hidden width (256)".
    # But D * C = 512 * 7 = 3584. 
    # Yes, the discriminator in CDAN needs to take 3584 features if implemented directly this way.
    # We will adjust the CDAN discriminator initialization in the main script to expect 3584 in_features.
    f_expanded = combined_feats.unsqueeze(2) # (N, 512, 1)
    p_expanded = p.unsqueeze(1)              # (N, 1, 7)
    g_x = torch.bmm(f_expanded, p_expanded).view(combined_feats.size(0), -1) # (N, 3584)
    
    domain_preds = domain_discriminator(g_x, alpha)
    
    # Domain labels: 0 for source, 1 for target
    domain_labels_source = torch.zeros(source_feats.size(0), dtype=torch.long, device=features.device)
    domain_labels_target = torch.ones(target_features.size(0), dtype=torch.long, device=features.device)
    domain_labels = torch.cat([domain_labels_source, domain_labels_target], dim=0)
    
    loss_domain = F.cross_entropy(domain_preds, domain_labels)
    
    total_loss = loss_cls + loss_domain
    return total_loss, {"loss_cls": loss_cls.item(), "loss_domain": loss_domain.item()}
