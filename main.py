from gpt_download import download_and_load_gpt2


settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")


@staticmethod
def get_text(file_path: str = "the-verdict.txt") -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
