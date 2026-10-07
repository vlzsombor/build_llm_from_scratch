import torch
import torch.nn as nn
from typing import Any


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Use a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Use a placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 3072),
            nn.ReLU(),
            nn.Linear(
                3072,
                768,
            ),
        )

    def forward(self, x):
        return self.layers(x)


class DummyAtt(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.W_query = nn.Linear(768, 768)
        self.W_key = nn.Linear(768, 768)
        self.W_value = nn.Linear(768, 768)
        self.out_proj = nn.Linear(768, 768)  # Linear layer to combine head outputs


class DummyLayerNorm(nn.Module):
    def __init__(self,a=0, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.scale = nn.Parameter(torch.ones(768))
        self.shift = nn.Parameter(torch.zeros(768))
    def forward(self, x):
        return x


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = DummyAtt()
        self.ff = FeedForward()
        self.norm1 = DummyLayerNorm()
        self.norm2 = DummyLayerNorm()
        self.drop_shortcut = nn.Dropout(False)
        # A simple placeholder

    def forward(self, x):
        # This block does nothing and just returns its input.
        return x
