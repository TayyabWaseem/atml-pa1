import torch
import torch.nn as nn
from torch.autograd import Function

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)

class DomainDiscriminator(nn.Module):
    def __init__(self, in_features=512):
        super().__init__()
        # Discriminator consisting of a 256-unit hidden layer, ReLU, dropout 0.5, and a two-class output layer
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
        
    def forward(self, x, alpha):
        x = grad_reverse(x, alpha)
        return self.net(x)
