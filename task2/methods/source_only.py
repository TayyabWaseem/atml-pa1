import torch.nn.functional as F

def compute_loss(features, logits, labels, **kwargs):
    """
    Source-only ERM loss.
    Only computes cross-entropy on the labeled source examples.
    """
    # Assuming the first N examples in the batch are from the source domains
    source_len = labels.size(0)
    source_logits = logits[:source_len]
    loss_cls = F.cross_entropy(source_logits, labels)
    return loss_cls, {"loss_cls": loss_cls.item()}
