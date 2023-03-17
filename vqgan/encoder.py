import torch
import torch.nn as nn
from vqgan.basic_block import BasicBlock


class CNNEncoder(nn.Module):

    def __init__(self, dropout_prob=0.5):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.layers = self.__make_layers(3, 8, 128, 3)
        self.to_z = nn.Linear(128, 128)

    def __make_layers(self, in_channels, spacing, out_channels, n_pools):
        blocks = [(in_channels, spacing)]
        while blocks[-1][-1] < out_channels:
            in_channels = blocks[-1][-1]
            blocks.append((in_channels, in_channels + spacing))
        pool_spacing = len(blocks) // (n_pools + 1)
        layers = []
        n_pools_added = 0
        for i, block in enumerate(blocks):
            layers.append(BasicBlock(*block, dropout_prob=self.dropout_prob))
            if (i + 1) % pool_spacing == 0 and n_pools_added < n_pools:
                layers.append(nn.MaxPool2d(2))
                n_pools_added += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    from torchvision.datasets import CIFAR100
    from torchvision import transforms

    dataset_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = CIFAR100('data/', download=True, transform=dataset_transform)
    model = CNNEncoder()

    data = torch.stack((train_dataset[0][0], train_dataset[1][0]))

    output = model(data)
    print(output)