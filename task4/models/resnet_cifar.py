import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        model = resnet18(weights=None)
        
        # Replace the ImageNet 7x7, stride-2 conv with a 3x3, stride-1 conv
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Remove the initial max-pooling layer (replace with Identity)
        model.maxpool = nn.Identity()
        
        # Replace the classifier
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        self.model = model
        
    def forward(self, x, return_features=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        h = self.model.layer2(x) # We'll need this for PROSER's manifold mixup
        x = self.model.layer3(h)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.model.fc(features)
        
        if return_features:
            return logits, features, h
        return logits
