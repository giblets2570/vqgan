import torch.nn as nn


class Head(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.proj = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        # x has shape (batch_size, seq_len, embedding_dim)
        x = self.layer_norm(x)
        x = self.linear(x)
        logits = self.proj(x)
        # logits has shape (batch_size, seq_len, vocab_size)
        return logits


class CodeGenerator(nn.Module):
    def __init__(self, n_codes=256, embedding_dim=128):
        super().__init__()
        # embedding layer
        self.embedding = nn.Embedding(n_codes, embedding_dim)
        # transformer decoder
        decoder_layer = nn.TransformerEncoderLayer(embedding_dim, nhead=8)
        norm = nn.LayerNorm(embedding_dim)
        self.decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=1, norm=norm)
        self.output_layer = Head(embedding_dim, n_codes)

    def forward(self, codes):
        # embed the codes
        embeddings = self.embedding(codes)
        # create the masks
        transformer_output = self.decoder(embeddings, is_causal=True)
        # return the output
        return self.output_layer(transformer_output)


if __name__ == '__main__':
    import torch
    codes = torch.randint(0, 255, (32, 4, 4))
    code_generator = CodeGenerator()

    output = code_generator(codes)

    print(output)
