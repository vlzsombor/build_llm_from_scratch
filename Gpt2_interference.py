import sys
import tiktoken
import torch.nn as nn
import torch
from typing import Any


class LayerNormalization(nn.Module):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(cfg["emb_dim"]))
        self.shift = nn.Parameter(torch.zeros(cfg["emb_dim"]))

    def forward(self, x):
        m = torch.mean(x, dim=-1, keepdim=True)
        v = torch.var(x, dim=-1, keepdim=True)
        norm_x = (x - m) / torch.sqrt(v + self.eps)
        return self.scale * norm_x + self.shift


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        d_in = cfg["emb_dim"]
        d_out = cfg["emb_dim"]
        qkv_bias = cfg["qkv_bias"]
        # self.att = MultiHeadAttention(
        #     d_in=cfg["emb_dim"],
        #     d_out=cfg["emb_dim"],
        #     context_length=cfg["context_length"],
        #     num_heads=cfg["n_heads"],
        #     dropout=cfg["drop_rate"],
        #     qkv_bias=cfg["qkv_bias"])
        n_heads = cfg["n_heads"]
        assert d_out % n_heads == 0, "d out must be divisible by num_heads"
        context_length = cfg["context_length"]

        self.d_out = d_out
        self.num_heads = n_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.head_dim = d_out // n_heads
        self.out_proj = nn.Linear(d_out, d_out)  # ???
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch, num_tokens, emb_dim = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        #######################################################

        keys = keys.view(batch, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch, num_tokens, self.num_heads, self.head_dim)

        # (batch, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        attn_weights = self.dropout(attn_weights)

        # (batch, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(batch, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


# class MaskedMultiheadAttention(nn.Module):
#     def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
#         super().__init__(*args, **kwargs)
#         self.W_query = nn.Linear(cfg["emb_dim"], cfg["emb_dim"])
#         self.W_key = nn.Linear(cfg["emb_dim"], cfg["emb_dim"])
#         self.W_value = nn.Linear(cfg["emb_dim"], cfg["emb_dim"])
#
#     def forward(self, x):
#         keys = self.W_key(x)
#         values = self.W_value(x)
#         queries = self.W_query(x)
#
#         att_score = queries @ keys.transpose(1, 2)
#
#         k_shape = keys.shape
#         a = att_score / (k_shape[-1] ** 0.5)
#         att_weights = torch.softmax(a, dim=-1)
#         context_vector = att_weights @ values
#
#         return context_vector
#


class FeedForward(nn.Module):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(cfg["emb_dim"] * 4, cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class Transformer(nn.Module):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.norm1 = LayerNormalization(cfg)
        self.norm2 = LayerNormalization(cfg)
        self.att = MultiHeadAttention(cfg)
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
        self.ff = FeedForward(cfg)

        pass

    def forward(self, x):

        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class Gpt2(nn.Module):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_out = nn.Dropout(cfg["drop_rate"])
        # self.transformer = Transformer(cfg)
        self.trf_blocks = nn.Sequential(
            *[Transformer(cfg) for _ in range(cfg["n_heads"])]
        )
        self.final_norm = LayerNormalization(cfg)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x):
        batch_size, seq_len = x.shape
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(seq_len))
        x = tok_emb + pos_emb
        x = self.drop_out(x)
        x = self.trf_blocks(x)

        x = self.final_norm(x)
        x = self.out_head(x)
        return x


def main(cfg, prompt):

    print(prompt)
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
    main(BASE_CONFIG, sys.argv[1])

    # todo -> Backpropagation of Attention/transformers
