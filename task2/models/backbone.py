import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class PACSRelatedResNet(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        
        # Remove original classifier
        self.features = nn.Sequential(*list(model.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1) # 512-d feature

    def train(self, mode=True):
        """
        Batch-normalization policy: freeze all BatchNorm running means and variances 
        at their pretrained ImageNet values for every method in Tasks 2 and 3. 
        The BatchNorm scale and bias parameters (gamma and beta) remain trainable.
        """
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval() # Freezes running mean and var, but affine params still receive gradients if requires_grad=True
