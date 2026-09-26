import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

from main import get_text


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


def main():
    txt = get_text()

    max_length = 4
    vocab_size = 50257
    output_dim = 256
    # ie = InputEncoder(
    #     txt,vocab_size, output_dim, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
    # )
    # res = ie.get_embedding()
    #
    # print(res.shape)
    dl = GPTDatasetV1.create_data_loader(
        txt, batch_size=4, max_length=max_length, stride=128
    )
    embed = InputEmbedding(vocab_size=vocab_size, output_dim=output_dim, context_length=256)

    for epoch in range(1):
        for inputs, targets in dl:  # a fresh pass each epoch
            x = embed(inputs)  # (batch, seq_len, 256)
            print(x.shape)
    return
    # dataloader = GPTDatasetV1.create_data_loader(
    #     txt, batch_size=1, max_length=4, stride=1, shuffle=False
    # )
    # token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    #
    # data_iter = iter(dataloader)
    # first = next(data_iter)
    # dataloader = GPTDatasetV1.create_data_loader(
    #     txt, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
    # )
    # data_iter = iter(dataloader)
    # inputs, targets = next(data_iter)
    # print("Token IDs:\n", inputs)
    # print("\nInputs shape:\n", inputs.shape)
    #
    # token_embeddings = token_embedding_layer(inputs)
    # print(token_embeddings.shape)
    # # print(f'token_embedding_layer {token_embedding_layer.weight.shape}')
    #
    # context_length = max_length
    # pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    # pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    # print(pos_embeddings.shape)
    # # print(pos_embeddings)
    # input_embeddings = token_embeddings + pos_embeddings
    # print(input_embeddings.shape)


if __name__ == "__main__":
    main()
