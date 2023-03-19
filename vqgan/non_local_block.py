import torch
import torch.nn as nn
from einops import rearrange


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.theta = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.g = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)

        self.out = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)

    def forward(self, inp):

        squeeze = False
        if inp.ndim == 3:
            inp = inp.unsqueeze(0)
            squeeze = True

        theta = self.theta(inp)
        phi = self.phi(inp)
        g = self.g(inp)

        b, x, y, z = theta.shape

        theta = rearrange(theta, 'b x y z -> b (x y) z')
        phi = rearrange(phi, 'b x y z -> b z (x y)')
        g = rearrange(g, 'b x y z -> b (x y) z')

        theta_phi = theta.bmm(phi).softmax(-1)
        theta_phi_g = theta_phi.bmm(g)

        theta_phi_g = rearrange(theta_phi_g, 'b (x y) z -> b x y z', x=x, y=y)

        output = self.out(theta_phi_g) + inp  # residual

        if squeeze:
            output = output.squeeze(0)
        return output


if __name__ == '__main__':
    block = NonLocalBlock(in_channels=64)

    inp = torch.randn((64, 16, 16))

    output = block(inp)

    nn.GroupNorm(32, 64)

    print(output)