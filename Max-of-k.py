from numpy import dtype

from Gpt2_interference import MultiHeadAttention
import sys
import tiktoken
import torch.nn as nn
import torch
from typing import Any


class MaxOfKTransformer(nn.Module):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attention = MultiHeadAttention(cfg)

    def forward(self, x):
        return self.attention(x)


class MaxOfKGpt(nn.Module):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.trf_blocks = MaxOfKTransformer(cfg)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x):
        batch, seq_length = x.shape
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)

        x = tok_emb + pos_emb
        x = self.trf_blocks(x)
        logits = self.out_head(x)
        return logits


if __name__ == "__main__":
    BASE_CONFIG = {
        "vocab_size": 64,  # Vocabulary size
        "context_length": 8,  # Context length
        "drop_rate": 0.0,  # Dropout rate
        "qkv_bias": True,  # Query-key-value bias
        "emb_dim": 32,
        "n_layers": 1,
        "n_heads": 1,
    }

    maxOfKGpt = MaxOfKGpt(BASE_CONFIG)

    input = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long).unsqueeze(0)

    logits = maxOfKGpt(input)

    print(logits.shape)
    i = torch.argmax(logits, dim=-1)
    print(i)
    pass
