import torch.nn as nn
from vqgan.basic_block import BasicBlock
from vqgan.non_local_block import NonLocalBlock


class CNNEncoder(nn.Module):

    def __init__(
        self,
        dropout_prob=0.5,
        spacing=8,
        out_channels=128,
        n_pools=3
    ):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.first_conv = nn.Conv2d(3, spacing, kernel_size=3, padding=1)
        self.residual_layers = self.__make_residual_layers(
            spacing,
            spacing,
            out_channels,
            n_pools
        )

        num_channels = self.residual_layers[-1].conv1.out_channels
        self.non_local_block = nn.Sequential(
            BasicBlock(num_channels, num_channels, dropout_prob=dropout_prob),
            NonLocalBlock(num_channels),
            BasicBlock(num_channels, num_channels, dropout_prob=dropout_prob),
        )
        self.group_norm = nn.GroupNorm(4, num_channels)
        self.swish = nn.SiLU()
        self.out_conv = nn.Conv2d(
            num_channels, out_channels, kernel_size=3, padding=1)

    def __make_residual_layers(
        self,
        in_channels,
        spacing,
        out_channels,
        n_pools
    ):
        blocks = [(in_channels, spacing)]
        while blocks[-1][-1] < out_channels - spacing:
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
        x = self.first_conv(x)
        x = self.residual_layers(x)
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