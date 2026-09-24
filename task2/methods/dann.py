import torch
import torch.nn.functional as F
import math

def compute_loss(features, logits, labels, domain_discriminator, alpha, target_features=None, **kwargs):
    source_len = labels.size(0)
    source_logits = logits[:source_len]
    loss_cls = F.cross_entropy(source_logits, labels)
    
    if target_features is None or len(target_features) == 0:
        return loss_cls, {"loss_cls": loss_cls.item(), "loss_domain": 0.0}
        
    source_feats = features[:source_len]
    
    combined_feats = torch.cat([source_feats, target_features], dim=0)
    domain_preds = domain_discriminator(combined_feats, alpha)
    
    # Domain labels: 0 for source, 1 for target
    domain_labels_source = torch.zeros(source_feats.size(0), dtype=torch.long, device=features.device)
    domain_labels_target = torch.ones(target_features.size(0), dtype=torch.long, device=features.device)
    domain_labels = torch.cat([domain_labels_source, domain_labels_target], dim=0)
    
    loss_domain = F.cross_entropy(domain_preds, domain_labels)
    
    # Optimization minimizes source classification loss plus domain-classification loss with unit weight.
    total_loss = loss_cls + loss_domain
    return total_loss, {"loss_cls": loss_cls.item(), "loss_domain": loss_domain.item()}
