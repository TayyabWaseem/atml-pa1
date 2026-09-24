import torch.nn.functional as F

def compute_loss(logits, labels):
    loss = F.cross_entropy(logits, labels)
    return loss, {"loss": loss.item()}
