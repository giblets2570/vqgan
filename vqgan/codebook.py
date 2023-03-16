import torch
import torch.nn as nn
from einops import rearrange

class CodeBook(nn.Module):

    def __init__(self, n_codes=256):
        super().__init__()

        self.embedding = nn.Embedding(n_codes, 64)

    def forward(self, z):
        """This will compute the z_q, z quantized.

        Args:
            z (_type_): _description_
        """

        squeeze = False
        if z.ndim == 3:
            z = z.unsqueeze(0)
            squeeze = True
        a, b, c = z.shape[-3:]

        x = rearrange(z, 't a b c -> t (b c) a')

        distances = torch.cdist(x, self.embedding.weight)
        codes = distances.argmin(-1)
        codes = rearrange(codes, 't (b c) -> t b c', b=b, c=c)
        if squeeze:
            codes = codes.squeeze(0)
        return codes

    def decode(self, z_q):
        squeeze = False
        if z_q.ndim == 2:
            z_q = z_q.unsqueeze(0)
            squeeze = True
        result = self.embedding(z_q)
        result = rearrange(result, 't a b c -> t c a b')
        if squeeze:
            result = result.squeeze(0)
        return result

if __name__ == "__main__":
    codebook = CodeBook()

    x = torch.randn(64, 4, 4)

    z_q = codebook(x)

    print(z_q)