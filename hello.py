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
        stide=128,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    ):
        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stide)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )


def main():
    print("barom")
    txt = get_text()
    dataloader = GPTDatasetV1.create_data_loader(
        txt, batch_size=1, max_length=4, stide=1, shuffle=False
    )
    data_iter = iter(dataloader)
    first = next(data_iter)
    print(first)


if __name__ == "__main__":
    main()
