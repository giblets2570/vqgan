import torch
import torch.nn as nn
from vqgan.basic_block import BasicBlock


class CNNEncoder(nn.Module):

    def __init__(self, dropout_prob=0.5):
        super().__init__()


        self.layers = nn.ModuleList([
            BasicBlock(3, 8),
            BasicBlock(8, 16),
            nn.MaxPool2d(2),
            BasicBlock(16, 24),
            BasicBlock(24, 32),
            nn.MaxPool2d(2),
            BasicBlock(32, 40),
            BasicBlock(40, 48),
            nn.MaxPool2d(2),
            BasicBlock(48, 56),
            BasicBlock(56, 64, out_activation=False)
        ])

    def forward(self, x):
        
        for layer in self.layers:
            x = layer(x)
        return x




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