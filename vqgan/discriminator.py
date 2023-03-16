import torch
import torch.nn as nn
from vqgan.basic_block import BasicBlock
from einops import rearrange


class CNNDiscriminator(nn.Module):

    def __init__(self, dropout_prob=0.5):
        super().__init__()
        # This will break the image into chunks of 4x4 to tell if it is real
        self.conv_layers = nn.Sequential(
            BasicBlock(3, 8, dropout_prob=dropout_prob),
            BasicBlock(8, 16, dropout_prob=dropout_prob),
            nn.MaxPool2d(2),
            BasicBlock(16, 24, dropout_prob=dropout_prob),
            BasicBlock(24, 32, dropout_prob=dropout_prob),
            nn.MaxPool2d(2),
            BasicBlock(32, 40, dropout_prob=dropout_prob),
            BasicBlock(40, 48, dropout_prob=dropout_prob),
            nn.MaxPool2d(2),
            BasicBlock(48, 56, dropout_prob=dropout_prob),
            BasicBlock(56, 64, dropout_prob=dropout_prob)
        )

        self.output_layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):

        squeeze = False
        if x.ndim == 3:
            x = x.unsqueeze(0)
            squeeze = True
        
        x = self.conv_layers(x)  # this will be shape 64 x 4 x 4

        x = rearrange(x, 'b x y z -> b y z x')  # 4 x 4 x 64

        x = self.output_layers(x).squeeze(-1)  # 4 x 4

        if squeeze:
            x = x.squeeze(0)

        return x



if __name__ == '__main__':
    image = torch.randn(3, 32, 32)

    disc = CNNDiscriminator()

    out = disc(image)

    print(out)