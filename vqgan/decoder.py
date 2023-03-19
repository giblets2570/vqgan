import torch
import torch.nn as nn
from vqgan.basic_block import BasicBlock


class CNNDecoder(nn.Module):

    def __init__(self, dropout_prob=0.5, spacing=8, in_channels=128, n_transposes=3):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.layers = self.__make_layers(in_channels, spacing, spacing, n_transposes)
        self.output_layer = nn.Conv2d(3, 3, kernel_size=3, padding='same')

    def __make_layers(self, in_channels, spacing, out_channels, n_transposes):
        blocks = [(in_channels, in_channels - spacing)]
        while blocks[-1][-1] > out_channels:
            in_channels = blocks[-1][-1]
            blocks.append((in_channels, in_channels - spacing))
        blocks.append((blocks[-1][-1], 3))
        transpose_spacing = len(blocks) // (n_transposes + 1)
        layers = []
        n_transposes_added = 0
        for i, block in enumerate(blocks):
            layers.append(BasicBlock(*block, dropout_prob=self.dropout_prob))
            if (i + 1) % transpose_spacing == 0 and n_transposes_added < n_transposes:
                channels = block[-1]
                layers.append(nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2, padding=0))
                n_transposes_added += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.output_layer(self.layers(x))


if __name__ == '__main__':
    model = CNNDecoder()

    input = torch.randn(2, 128, 4, 4)

    output = model(input)

    print(output.shape)
