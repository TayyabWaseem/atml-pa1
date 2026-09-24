import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PROSER(nn.Module):
    def __init__(self, base_model, num_dummy=5):
        super().__init__()
        self.base_model = base_model
        
        # Add dummy classifiers
        # The base model's fc has shape (num_classes, in_features)
        in_features = self.base_model.model.fc.in_features
        num_known = self.base_model.model.fc.out_features
        
        self.num_dummy = num_dummy
        self.num_known = num_known
        
        # Create a new linear layer that includes the dummy classes
        self.fc_extended = nn.Linear(in_features, num_known + num_dummy)
        
        # Initialize the known class weights from the base model
        with torch.no_grad():
            self.fc_extended.weight[:num_known] = self.base_model.model.fc.weight
            self.fc_extended.bias[:num_known] = self.base_model.model.fc.bias
            
            # The dummy weights are initialized randomly (default behavior of nn.Linear)
            
        # Replace the base model's fc
        self.base_model.model.fc = self.fc_extended
        
    def forward(self, x, return_features=False):
        return self.base_model(x, return_features=return_features)

def compute_classifier_placeholder_loss(logits, labels, beta=1.0):
    # logits shape: (N, num_known + num_dummy)
    num_known = logits.size(1) - 5 # Assuming 5 dummy
    
    # 1. Standard cross-entropy on known classes
    logits_known = logits[:, :num_known]
    loss_ce = F.cross_entropy(logits_known, labels)
    
    # 2. Classifier placeholder loss (push second max to be a dummy class)
    # Exclude the true class from the logits to find the "remaining" responses
    mask = torch.ones_like(logits).scatter_(1, labels.unsqueeze(1), 0.0)
    logits_masked = logits * mask - (1 - mask) * 1e9 # Mask out true class
    
    # We want the max of the remaining to be one of the dummy classes
    # Following Zhou et al. (2021) and the assignment's instructions
    # Max over known classes (excluding true class)
    max_known = logits_masked[:, :num_known].max(dim=1)[0]
    # Max over dummy classes
    max_dummy = logits_masked[:, num_known:].max(dim=1)[0]
    
    # We want max_dummy > max_known -> minimize max_known - max_dummy, bounded
    loss_cp = F.softplus(max_known - max_dummy).mean()
    
    return loss_ce + beta * loss_cp

def compute_data_placeholder_loss(logits_mixed):
    # The mixed representations should be classified as one of the dummy classes
    num_known = logits_mixed.size(1) - 5
    
    # Target distribution: uniform over dummy classes, 0 for known classes
    # Or just cross entropy with a target that encourages predicting ANY dummy class
    # The paper uses log_softmax and maximizes probability of dummy classes
    log_probs = F.log_softmax(logits_mixed, dim=1)
    
    # Probability assigned to dummy classes
    dummy_log_probs = log_probs[:, num_known:]
    
    # We want to maximize the sum of probabilities of dummy classes
    # which is equivalent to minimizing -log(\sum exp(dummy_log_probs))
    # logsumexp gives log(sum(exp(x)))
    loss_dp = -torch.logsumexp(dummy_log_probs, dim=1).mean()
    
    return loss_dp
