import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from vqgan.mobilenet import mobilenet_v2
from vqgan.vgg import vgg19_bn


class PerceptualLoss(nn.Module):
    def __init__(self, layers=[3, 5, 9, 11, 15], layer_weights=[0.2, 0.2, 0.2, 0.2, 0.2], model='mobilenet_v2'):
        super().__init__()
        self.layers = layers
        self.layer_weights = layer_weights
        if model == 'mobilenet_v2':
            self.feature_extractor = mobilenet_v2(pretrained=True).features
        elif model == 'vgg19_bn':
            self.feature_extractor = vgg19_bn(pretrained=True).features
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_features = []
        y_features = []
        for i, module in enumerate(self.feature_extractor):
            x = module(x)
            y = module(y)
            if i in self.layers:
                x_features.append(x)
                y_features.append(y)
        loss = 0
        for i in range(len(x_features)):
            loss += self.layer_weights[i] * F.mse_loss(x_features[i], y_features[i].detach())
        return loss
