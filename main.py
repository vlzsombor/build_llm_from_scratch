@staticmethod
def get_text(file_path: str='the-verdict.txt') -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
