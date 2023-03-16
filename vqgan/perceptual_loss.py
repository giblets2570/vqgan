import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


class PerceptualLoss(nn.Module):
    def __init__(self, layer_weights=[0.2, 0.2, 0.2, 0.2, 0.2]):
        super().__init__()
        self.layer_weights = layer_weights
        self.vgg = models.vgg16(pretrained=True).features
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_features = []
        y_features = []
        for i, module in enumerate(self.vgg):
            x = module(x)
            y = module(y)
            if i in {3, 8, 15, 22, 29}:
                x_features.append(x)
                y_features.append(y)
        loss = 0
        for i in range(len(x_features)):
            loss += self.layer_weights[i] * F.mse_loss(x_features[i], y_features[i].detach())
        return loss
