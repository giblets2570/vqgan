import torch
import torch.nn as nn
from vqgan.residual_block import ResidualBlock
from vqgan.non_local_block import NonLocalBlock
from vqgan.upsample_block import UpSampleBlock


class CNNDecoder(nn.Module):

    def __init__(
        self,
        dropout_prob=0.5,
        in_channels=128,
        m=3
    ):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.cs = self.__find_cs(m, in_channels)

        self.first_conv = nn.Conv2d(
            in_channels,
            self.cs[0],
            kernel_size=3,
            padding=1
        )

        self.non_local_block = nn.Sequential(
            ResidualBlock(self.cs[0], self.cs[0], dropout_prob=dropout_prob),
            NonLocalBlock(self.cs[0]),
            ResidualBlock(self.cs[0], self.cs[0], dropout_prob=dropout_prob),
        )

        res_upsample_layers = []
        for i, channels in enumerate(self.cs[:-1]):
            res_upsample_layers.append(
                ResidualBlock(channels, self.cs[i + 1]))
            res_upsample_layers.append(
                UpSampleBlock(self.cs[i + 1], self.cs[i + 1]))

        self.res_upsample_layers = nn.Sequential(*res_upsample_layers)

        n_groups = self.__find_n_groups(self.cs[-1])
        print(f'Using {n_groups} groups for group norm in decoder')
        self.group_norm = nn.GroupNorm(n_groups, self.cs[-1])
        self.swish = nn.SiLU()

        self.output_layer = nn.Sequential(
            nn.Conv2d(self.cs[-1], 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def __find_cs(self, m, in_channels):
        return [(in_channels * i) // (m + 2) for i in range(1, m + 2)][::-1]

    def __find_n_groups(self, in_channels):
        # Find an n groups between 4 and 12
        for i in range(4, 13):
            if in_channels % i == 0:
                return i
        return 1

    def forward(self, x):
        x = self.first_conv(x)
        x = self.non_local_block(x)
        x = self.res_upsample_layers(x)
        x = self.group_norm(x)
        x = self.swish(x)
        return self.output_layer(x)


if __name__ == '__main__':
    model = CNNDecoder()

    input = torch.randn(2, 128, 4, 4)

    output = model(input)

    print(output.shape)
