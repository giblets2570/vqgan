import torch.nn as nn
from vqgan.residual_block import ResidualBlock
from vqgan.non_local_block import NonLocalBlock
from vqgan.downsample_block import DownSampleBlock
import math


class CNNEncoder(nn.Module):

    def __init__(
        self,
        dropout_prob=0.5,
        spacing=8,
        out_channels=128,
        m=3
    ):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.cs = self.__find_cs(m, out_channels)
        self.first_conv = nn.Conv2d(3, self.cs[0], kernel_size=3, padding=1)

        res_downsample_layers = []
        for i, channels in enumerate(self.cs[:-1]):
            res_downsample_layers.append(
                ResidualBlock(channels, self.cs[i + 1]))
            res_downsample_layers.append(
                DownSampleBlock(self.cs[i + 1], self.cs[i + 1]))

        self.res_downsample_layers = nn.Sequential(*res_downsample_layers)

        self.non_local_block = nn.Sequential(
            ResidualBlock(self.cs[-1], self.cs[-1], dropout_prob=dropout_prob),
            NonLocalBlock(self.cs[-1]),
            ResidualBlock(self.cs[-1], self.cs[-1], dropout_prob=dropout_prob),
        )
        n_groups = math.gcd(24, self.cs[-1])
        self.group_norm = nn.GroupNorm(n_groups, self.cs[-1])
        self.swish = nn.SiLU()
        self.out_conv = nn.Conv2d(
            self.cs[-1], out_channels, kernel_size=3, padding=1)

    def __find_cs(self, m, out_channels):
        return [(out_channels * i) // (m + 2) for i in range(1, m + 2)]

    def forward(self, x):
        x = self.first_conv(x)
        x = self.res_downsample_layers(x)
        x = self.non_local_block(x)
        x = self.group_norm(x)
        x = self.swish(x)
        return self.out_conv(x)


if __name__ == '__main__':
    import torch
    encoder = CNNEncoder()

    inp = torch.randn(4, 3, 32, 32)

    out = encoder(inp)

    print(out)
