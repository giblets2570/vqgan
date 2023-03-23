import torch
import torch.nn as nn
from vqgan.residual_block import ResidualBlock
from vqgan.downsample_block import DownSampleBlock


class CNNDiscriminator(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, m=3, dropout_prob=0.5, channels=32):
        super(CNNDiscriminator, self).__init__()
        self.dropout_prob = dropout_prob
        self.first_conv = nn.Conv2d(
            in_channels, channels, kernel_size=3, padding=1)
        res_downsample_layers = []
        for i in range(m):
            in_c = channels * (i + 1)
            out_c = in_c + channels
            res_downsample_layers.append(
                ResidualBlock(in_c, out_c, dropout_prob=dropout_prob))
            res_downsample_layers.append(
                DownSampleBlock(out_c, out_c))

        self.res_downsample_layers = nn.Sequential(*res_downsample_layers)

        in_c = (m + 1) * channels
        self.output_layer = nn.Conv2d(
            in_c, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.res_downsample_layers(x)
        x = self.output_layer(x).squeeze(-3)
        return x


if __name__ == '__main__':
    image = torch.randn(1, 3, 32, 32)

    discriminator = CNNDiscriminator(3, 1)

    output = discriminator(image)

    print(output)
