import re
import argparse
import tiktoken
from main import get_text

DELIMITERS = r'([,.:;?_!"()\']|--|\s)'  # [",", ";", " ", ".", "-", "\n"]
UNKNOWN = "<|unk|>"




class SimpleTokenizer_V1:
    ###############################
    # 1) TOKENIZER
    ###############################
    @staticmethod
    def get_tokenized_text(input_text: str) -> list[str]:
        # pattern = "|".join(map(re.escape, DELIMITERS))
        return re.split(DELIMITERS, input_text.lower())
        # return re.split(pattern, input_text)

    ################################
    # VOCABULARY
    # ##############################
    @staticmethod
    def get_vocabulary(all_words: list[str]):
        all_words.extend(["<|endoftext|>", "<|unk|>"])
        vocab = {token: integer for integer, token in enumerate(all_words)}
        return vocab
        # for item in list(vocab.items())[:50]:
        #     print(item)

    def __init__(self, text: str) -> None:
        self.vocabulary = self.get_vocabulary(self.get_tokenized_text(text))
        self.decoder_dict = {i: s for s, i in self.vocabulary.items()}
        pass

    def encode(self, token: str) -> list[int]:
        tokenized = self.get_tokenized_text(token)
        return self.encode_by_list(tokenized)

    def encode_by_list(self, tokens: list[str]) -> list[int]:
        return list(
            map(
                lambda x: (
                    self.vocabulary[x]
                    if x in self.vocabulary
                    else self.vocabulary[UNKNOWN]
                ),
                tokens,
            )
        )

    def decode(self, ids: list[int]) -> str:
        return " ".join(map(lambda x: self.decode_single(x), ids))

    def decode_single(self, id: int) -> str:
        return self.decoder_dict[id]

    # def get_token(self, token_id:int) -> str:
    #     return self.vocabular[token_id]





def main(c: int):
    print("barom")

    text = get_text("the-verdict.txt")
    text = text

    prompt = "I had always thought"

    prompt = "It's the last he painted, you know"

    prompt = """"It's the last he painted, you know,"
    Mrs. Gisburn said with pardonable pride."""
    prompt = "hello zsombor"
    prompt = """Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."""
    tokenizer = SimpleTokenizer_V1(text)
    ids = tokenizer.encode(prompt)
    print(ids)
    print(tokenizer.decode(ids))
    # print(len(text))
    # print(get_tokenized_text(text))
    # print(get_vocabulary(get_tokenized_text(text)))
    print('###############################')
    tokenizer = tiktoken.get_encoding("gpt2")
    text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
    )
    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=int, default=50)
    args = parser.parse_args()
    main(args.c)
