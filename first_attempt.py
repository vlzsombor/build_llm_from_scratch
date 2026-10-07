import enum
from gpt_download import download_and_load_gpt2
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from main import get_text
from typing import Any
import torch.nn.functional as F
import math
import numpy as np


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride) -> None:
        super().__init__()
        token_ids: list[int] = tokenizer.encode(txt)
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk: list[int] = token_ids[i : i + max_length]
            target_chunk: list[int] = token_ids[i + 1 : i + 1 + max_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return self.input_ids[index], self.target_ids[index]

    @staticmethod
    def create_data_loader(
        txt,
        batch_size=4,
        max_length=256,
        stride=128,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    ):
        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )


# class InputEncoder:
#     def __init__(
#         self,
#         text: str,
#         vocab_size: int,
#         output_dim: int,
#         batch_size=4,
#         max_length=256,
#         stride=128,
#         shuffle=True,
#         drop_last=True,
#         num_workers=0,
#     ) -> None:
#         self.dataset = iter(
#             GPTDatasetV1.create_data_loader(
#                 text,
#                 batch_size=batch_size,
#                 max_length=max_length,
#                 stride=stride,
#                 shuffle=shuffle,
#                 drop_last=drop_last,
#                 num_workers=num_workers,
#             )
#         )
#         self.token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
#         self.pos_embeddings_layer = torch.nn.Embedding(max_length, output_dim)
#         self.max_length = max_length
#
#     def get_embedding(self):
#         inputs, targets = next(self.dataset)
#         token_embeddings = self.token_embedding_layer(inputs)
#         pos_embeddings = self.pos_embeddings_layer(torch.arange(self.max_length))
#         return token_embeddings + pos_embeddings


# the embedding vector is a ‘look up table’ it says for every token id the corresponding vector representation (inner representation of the gpt2) ?
#
#
class InputEmbedding(torch.nn.Module):
    def __init__(self, vocab_size: int, output_dim: int, context_length: int) -> None:
        super().__init__()
        self.token_emb = torch.nn.Embedding(
            vocab_size, output_dim
        )  # vocab_size x vector -> 50257 x 768
        self.pos_emb = torch.nn.Embedding(context_length, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        seq_len = inputs.shape[1]
        positions = torch.arange(seq_len, device=inputs.device)
        return self.token_emb(inputs) + self.pos_emb(positions)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GPTModel(nn.Module):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.emb = InputEmbedding(
            cfg["vocab_size"], cfg["emb_dim"], cfg["context_length"]
        )
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[Transformer(cfg) for _ in range(cfg["n_layers"])]
        )
        self.norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # ...

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        x = self.emb(in_idx)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.norm(x)

        logits = self.out_head(x)
        return logits


class SimplifiedSelfAttention(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def step1(self, inputs):
        query = inputs[1]
        attn_scores_2 = torch.empty(inputs.shape[0])

        for i, x_i in enumerate(inputs):
            attn_scores_2[i] = torch.dot(x_i, query)
        # print(attn_scores_2)

        attn_weights_2 = torch.softmax(
            attn_scores_2, dim=0
        )  # attn_scores_2 / attn_scores_2.sum()

        print(attn_weights_2)

        query = inputs[1]
        context_vec_2 = torch.zeros(query.shape)
        print("debug")
        for i, x_i in enumerate(inputs):
            print(f"i: {i} x_i: {x_i}")
            print(f"attn_weights_2 {attn_weights_2[i]}")
            context_vec_2 += attn_weights_2[i] * x_i
            print(f"context vec: {context_vec_2}")

            print("end")

        print(context_vec_2)

    def step1_all(self, inputs):
        attention_scores = inputs @ inputs.T

        print("step2")
        # print(attention_scores)
        # print(torch.softmax(attention_scores, dim=0))
        att_weights = torch.softmax(attention_scores, dim=-1)
        context_vector = att_weights @ inputs
        print(context_vector)

    def forward(self, x):
        inputs = torch.tensor(
            [
                [0.43, 0.15, 0.89],  # Your (x^1)
                [0.55, 0.87, 0.66],  # journey (x^2)
                [0.57, 0.85, 0.64],  # starts (x^3)
                [0.22, 0.58, 0.33],  # with (x^4)
                [0.77, 0.25, 0.10],  # one (x^5)
                [0.05, 0.80, 0.55],
            ]  # step (x^6)
        )
        # self.step1(inputs)
        self.step1_all(inputs)
        return x


class SimplifiedSelfAttention_Weights(nn.Module):
    def __init__(
        self, d_in: int = 3, d_out: int = 2, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.w_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        self.w_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        self.w_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    def forward(self, x):

        inputs = torch.tensor(
            [
                [0.43, 0.15, 0.89],  # Your (x^1)
                [0.55, 0.87, 0.66],  # journey (x^2)
                [0.57, 0.85, 0.64],  # starts (x^3)
                [0.22, 0.58, 0.33],  # with (x^4)
                [0.77, 0.25, 0.10],  # one (x^5)
                [0.05, 0.80, 0.55],
            ]  # step (x^6)
        )

        x_2 = inputs[1]

        query_2 = x_2 @ self.w_query
        # key_2 = x_2 @ self.w_key
        # value_2 = x_2 @ self.w_value
        #
        keys = inputs @ self.w_key
        values = inputs @ self.w_value
        print(f"keys.shape: {keys.shape}")
        print(f"values.shape: {values.shape}")

        key_2 = keys[1]
        attn_score_22 = query_2.dot(key_2)
        print(attn_score_22)
        # print(query_2)
        # print(key_2)
        # print(value_2)
        attn_scores_2 = query_2 @ keys.T
        d_k = keys.shape[-1]
        attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
        print(attn_weights_2)

        context_vec_2 = attn_weights_2 @ values
        print("context_vec_2")
        print(context_vec_2)

        return x


class SelfAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.w_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        self.w_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        self.w_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    def forward(self, x):
        inputs = torch.tensor(
            [
                [0.43, 0.15, 0.89],  # Your (x^1)
                [0.55, 0.87, 0.66],  # journey (x^2)
                [0.57, 0.85, 0.64],  # starts (x^3)
                [0.22, 0.58, 0.33],  # with (x^4)
                [0.77, 0.25, 0.10],  # one (x^5)
                [0.05, 0.80, 0.55],
            ]  # step (x^6)
        )
        query = inputs @ self.w_query
        key = inputs @ self.w_key
        value = inputs @ self.w_value

        a_score = query @ key.T

        dimension = inputs.shape[-1]

        a_weight = F.softmax(a_score / math.sqrt(dimension), dim=-1)

        context_vector = a_weight @ value

        print(context_vector)
        return context_vector


#####################################
# Chapter 3,
#####################################
class MultiHeadAttention(nn.Module):
    mask: torch.Tensor

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = (
            d_out // num_heads
        )  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        print(x.shape)
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        print(context_vec.shape)
        return context_vec


class FeedForward(nn.Module):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class Transformer(nn.Module):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attention = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
        self.feed_forward = FeedForward(cfg)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.emb.pos_emb.weight = assign(gpt.emb.pos_emb.weight, params["wpe"])
    gpt.emb.token_emb.weight = assign(gpt.emb.token_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attention"]["c_attn"])["w"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T
        )

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b
        )
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b
        )

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"]
        )
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"]
        )

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def generate(
    model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None
):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if (
            idx_next == eos_id
        ):  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def main():
    torch.manual_seed(123)
    txt = get_text()

    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False,  # Query-Key-Value bias
    }

    model_name = "gpt2-small (124M)"
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])

    model = GPTModel(NEW_CONFIG)

    load_weights_into_gpt(model, params)
    model.to(device)

    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)
    input_prompt = "The capital of France is"
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_prompt, tokenizer),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=50,
        temperature=1.0,
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    # dl = GPTDatasetV1.create_data_loader(
    #     txt,
    #     batch_size=4,
    #     max_length=GPT_CONFIG_124M["context_length"],
    #     stride=128
    # )
    #
    # for epoch in range(1):
    #     for inputs, targets in dl:
    #         # print(inputs)
    #         res = model(inputs)
    #         print(res.shape)
    #         break

    return


if __name__ == "__main__":
    main()
