import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from main import get_text
from typing import Any


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


class InputEmbedding(torch.nn.Module):
    def __init__(self, vocab_size: int, output_dim: int, context_length: int) -> None:
        super().__init__()
        self.token_emb = torch.nn.Embedding(vocab_size, output_dim)
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
        self.transformer = Transformer(cfg)
        # ...

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        x = self.emb(in_idx)
        x = self.drop_emb(x)
        x = self.transformer(x)
        return x


class SimplifiedSelfAttention(nn.Module):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x



class Transformer(nn.Module):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attention = SimplifiedSelfAttention()
        self.layer1 = LayerNorm(cfg["emb_dim"])

    def forward(self, x):
        x = self.layer1(x)
        x = self.attention(x)
        return x


def main():
    txt = get_text()

    # GPT_CONFIG_124M = {
    #     "vocab_size": 50257, # Vocabulary size
    #     "context_length": 1024, # Context length
    #     "emb_dim": 768, # Embedding dimension
    #     "n_heads": 12, # Number of attention heads
    #     "n_layers": 12, # Number of layers
    #     "drop_rate": 0.1, # Dropout rate
    #     "qkv_bias": False # Query-Key-Value bias
    # }
    #
    max_length = 4
    vocab_size = 50257
    GPT_CONFIG_124M = {
        "vocab_size": vocab_size,  # Vocabulary size
        "context_length": max_length,  # Context length
        "emb_dim": 256,  # Embedding dimension
        "drop_rate": 0.1,  # Dropout rate
    }


    model = GPTModel(GPT_CONFIG_124M)

    dl = GPTDatasetV1.create_data_loader(
        txt, batch_size=4, max_length=max_length, stride=128
    )

    for epoch in range(1):
        for inputs, targets in dl:  # a fresh pass each epoch
            res = model(inputs)
            print(res.shape)
            break

    return


if __name__ == "__main__":
    main()
