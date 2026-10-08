import tiktoken
import torch.nn as nn
import torch
from typing import Any


class LayerNormalization(nn.Module):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(cfg["emb_dim"]))
        self.bias = nn.Parameter(torch.zeros(cfg["emb_dim"]))

    def forward(self, x):
        m = torch.mean(x, dim=-1, keepdim=True)
        v = torch.var(x, dim=-1, keepdim=True)
        norm_x = (x - m) / torch.sqrt(v + self.eps)
        return self.scale * norm_x + self.bias


class MaskedMultiheadAttention(nn.Module):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.W_query = nn.Linear(cfg["emb_dim"], cfg["emb_dim"])
        self.W_key = nn.Linear(cfg["emb_dim"], cfg["emb_dim"])
        self.W_value = nn.Linear(cfg["emb_dim"], cfg["emb_dim"])

    def forward(self, x):
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)

        att_score = queries @ keys.transpose(1, 2)

        k_shape = keys.shape
        a = att_score / (k_shape[-1] ** 0.5)
        att_weights = torch.softmax(a, dim=-1)
        context_vector = att_weights @ values

        return context_vector


class Transformer(nn.Module):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer_norm1 = LayerNormalization(cfg)
        pass

    def forward(self, x):
        x = self.layer_norm1(x)
        return x


class Gpt2(nn.Module):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_out = nn.Dropout(cfg["drop_rate"])
        self.transformer = Transformer(cfg)

    def forward(self, x):
        batch_size, seq_len = x.shape
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(seq_len))
        x = tok_emb + pos_emb
        x = self.drop_out(x)
        x = self.transformer(x)

        return x


def main(cfg):
    test = torch.arange(5)
    prompt = "The capital city of France"
    enc = tiktoken.get_encoding("gpt2")

    x = enc.encode(prompt)
    gpt = Gpt2(cfg)
    x = torch.tensor(x, dtype=torch.long).unsqueeze(0)
    x = gpt(x)


if __name__ == "__main__":
    BASE_CONFIG = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,  # Dropout rate
        "qkv_bias": True,  # Query-key-value bias
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12,
    }
    main(BASE_CONFIG)

    # todo -> Page 87-108 Efficient multi-head attention class understand!
