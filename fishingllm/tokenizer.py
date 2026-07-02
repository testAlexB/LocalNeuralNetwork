from tokenizers import ByteLevelBPETokenizer
from tokenizers.normalizers import NFC
from pathlib import Path


class FishingTokenizer:
    """BPE-токенизатор для FishingLLM (byte-level, NFC нормализация)."""

    def __init__(self, vocab_size: int = 16_000):
        self.vocab_size = vocab_size
        self.special_tokens = [
            "<unk>",
            "<s>",
            "</s>",
            "<|user|>",
            "<|assistant|>",
            "<|system|>",
        ]
        self._tokenizer = ByteLevelBPETokenizer()

    def train(self, files: list[str]):
        self._tokenizer.train(
            files=files,
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=self.special_tokens,
        )
        self._tokenizer._tokenizer.normalizer = NFC()

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        encoded = self._tokenizer.encode_batch(texts)
        return [e.ids for e in encoded]

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)

    def decode_batch(self, batch: list[list[int]]) -> list[str]:
        return [self.decode(ids) for ids in batch]

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save_model(str(path))

    def load(self, path: str | Path):
        path = Path(path)
        self._tokenizer = ByteLevelBPETokenizer(
            vocab=str(path / "vocab.json"),
            merges=str(path / "merges.txt"),
        )
        self._tokenizer._tokenizer.normalizer = NFC()

    def get_vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> int | None:
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> str | None:
        return self._tokenizer.id_to_token(token_id)
