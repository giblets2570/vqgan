import torch
import torch.nn as nn
from vqgan.basic_block import BasicBlock
from vqgan.non_local_block import NonLocalBlock
from vqgan.upsample_block import UpSampleBlock


class CNNDecoder(nn.Module):

    def __init__(
        self,
        dropout_prob=0.5,
        spacing=8,
        in_channels=128,
        n_transposes=3
    ):
        super().__init__()
        self.dropout_prob = dropout_prob

        b_channels = in_channels - spacing

        self.first_conv = nn.Conv2d(
            in_channels,
            b_channels,
            kernel_size=3,
            padding=1
        )

        self.non_local_block = nn.Sequential(
            BasicBlock(b_channels, b_channels, dropout_prob=dropout_prob),
            NonLocalBlock(b_channels),
            BasicBlock(b_channels, b_channels, dropout_prob=dropout_prob),
        )

        self.residual_layers = self.__make_residual_layers(
            b_channels,
            spacing,
            spacing,
            n_transposes
        )

        n_channels = self.residual_layers[-1].conv1.out_channels
        self.group_norm = nn.GroupNorm(4, n_channels)
        self.swish = nn.SiLU()

        self.output_layer = nn.Conv2d(n_channels, 3, kernel_size=3, padding=1)

    def __make_residual_layers(self, in_channels, spacing, out_channels, n_transposes):
        blocks = [(in_channels, in_channels - spacing)]
        while blocks[-1][-1] > out_channels:
            in_channels = blocks[-1][-1]
            blocks.append((in_channels, in_channels - spacing))
        transpose_spacing = len(blocks) // (n_transposes + 1)
        layers = []
        n_transposes_added = 0
        for i, block in enumerate(blocks):
            layers.append(BasicBlock(*block, dropout_prob=self.dropout_prob))
            if (i + 1) % transpose_spacing == 0 and n_transposes_added < n_transposes:
                channels = block[-1]
                layers.append(UpSampleBlock(channels, channels))
                n_transposes_added += 1
        assert n_transposes_added == n_transposes, 'not enough conv transposes added'
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.non_local_block(x)
        x = self.residual_layers(x)
        x = self.group_norm(x)
        x = self.swish(x)
        return self.output_layer(x)


if __name__ == '__main__':
    model = CNNDecoder()

    input = torch.randn(2, 128, 4, 4)

    output = model(input)

    print(output.shape)
