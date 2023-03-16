import torch
import torch.nn as nn
from vqgan.basic_block import BasicBlock

class CNNDecoder(nn.Module):

    def __init__(self, dropout_prob=0.5):
        super().__init__()

        self.layers = nn.Sequential(
            BasicBlock(64, 56, dropout_prob=dropout_prob),
            BasicBlock(56, 48, dropout_prob=dropout_prob),
            nn.ConvTranspose2d(48, 48, kernel_size=4, stride=2, padding=1),
            BasicBlock(48, 40, dropout_prob=dropout_prob),
            BasicBlock(40, 32, dropout_prob=dropout_prob),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=1),
            BasicBlock(32, 24, dropout_prob=dropout_prob),
            BasicBlock(24, 16, dropout_prob=dropout_prob),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),
            BasicBlock(16, 16, dropout_prob=dropout_prob),
            BasicBlock(16, 8, dropout_prob=dropout_prob)
        )

        self.output_layer = nn.Conv2d(8, 3, kernel_size=3)


    def forward(self, x):
        x = self.layers(x)
        return self.output_layer(x)



if __name__ == '__main__':
    model = CNNDecoder()

    input = torch.randn(2, 64, 4, 4)


    output = model(input)


    print(output.shape)