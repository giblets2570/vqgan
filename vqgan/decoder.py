import torch
import torch.nn as nn
from vqgan.basic_block import BasicBlock

class CNNDecoder(nn.Module):

    def __init__(self, dropout_prob=0.5):
        super().__init__()

        self.layers = nn.ModuleList([
            BasicBlock(64, 56),
            BasicBlock(56, 48),
            nn.ConvTranspose2d(48, 48, kernel_size=4, stride=2, padding=1),
            BasicBlock(48, 40),
            BasicBlock(40, 32),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=1),
            BasicBlock(32, 24),
            BasicBlock(24, 16),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),
            BasicBlock(16, 16),
            BasicBlock(16, 8)
        ])

        self.output_layer = nn.Conv2d(8, 3, kernel_size=3)


    def forward(self, x):
        
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)



if __name__ == '__main__':
    model = CNNDecoder()

    input = torch.randn(2, 64, 4, 4)


    output = model(input)


    print(output.shape)