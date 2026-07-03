import pytest
import tempfile
from pathlib import Path
from fishingllm.tokenizer import FishingTokenizer


@pytest.fixture
def sample_corpus():
    texts = [
        "Щука поймала леща на спиннинг в Тверской области.",
        "Окунь и лещ клюют на червя в июне на Верхней Волге.",
        "Для ловли щуки используйте блесну или воблер.",
        "Сом — самая крупная рыба в наших водоёмах.",
        "Рыбалка на фидер требует терпения и хорошей наживки.",
        "Зимой мормышка — лучший выбор для окуня.",
        "В Тверской области более 500 озёр и 100 рек.",
        "Карась ловится на поплавочную удочку у самого берега.",
        "Судак предпочитает глубокие ямы и каменистое дно.",
        "Налим активен ночью в холодной воде.",
    ]
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt", delete=False) as f:
        for t in texts:
            f.write(t + "\n")
        tmp_path = f.name
    yield tmp_path
    Path(tmp_path).unlink(missing_ok=True)


@pytest.fixture
def trained_tokenizer(sample_corpus):
    tok = FishingTokenizer(vocab_size=500)
    tok.train([sample_corpus])
    return tok


class TestFishingTokenizerInit:
    def test_default_vocab_size(self):
        tok = FishingTokenizer()
        assert tok.vocab_size == 16_000

    def test_custom_vocab_size(self):
        tok = FishingTokenizer(vocab_size=500)
        assert tok.vocab_size == 500

    def test_special_tokens_present(self):
        tok = FishingTokenizer(vocab_size=500)
        expected = ["<unk>", "<s>", "</s>", "<|user|>", "<|assistant|>", "<|system|>"]
        assert tok.special_tokens == expected


class TestTraining:
    def test_train_increases_vocab(self, trained_tokenizer):
        assert trained_tokenizer.get_vocab_size() > 256

    def test_vocab_size_respected(self, trained_tokenizer):
        assert trained_tokenizer.get_vocab_size() <= 500

    def test_special_tokens_in_vocab(self, trained_tokenizer):
        for tok in trained_tokenizer.special_tokens:
            assert trained_tokenizer.token_to_id(tok) is not None


class TestEncodeDecode:
    def test_encode_returns_ids(self, trained_tokenizer):
        ids = trained_tokenizer.encode("щука")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0

    def test_decode_roundtrip(self, trained_tokenizer):
        text = "щука поймала леща"
        ids = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(ids)
        assert decoded == text

    def test_decode_roundtrip_russian(self, trained_tokenizer):
        text = "В Тверской области более 500 озёр и 100 рек."
        ids = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(ids)
        assert decoded == text

    def test_decode_roundtrip_special_chars(self, trained_tokenizer):
        text = "Сом — самая крупная рыба!"
        ids = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(ids)
        assert decoded == text

    def test_empty_string(self, trained_tokenizer):
        ids = trained_tokenizer.encode("")
        decoded = trained_tokenizer.decode(ids)
        assert decoded == ""


class TestBatch:
    def test_encode_batch(self, trained_tokenizer):
        texts = ["щука", "окунь", "лещ"]
        batch = trained_tokenizer.encode_batch(texts)
        assert len(batch) == 3
        assert all(len(ids) > 0 for ids in batch)

    def test_decode_batch(self, trained_tokenizer):
        texts = ["щука", "окунь", "лещ"]
        ids_batch = trained_tokenizer.encode_batch(texts)
        decoded = trained_tokenizer.decode_batch(ids_batch)
        assert decoded == texts

    def test_encode_batch_empty(self, trained_tokenizer):
        batch = trained_tokenizer.encode_batch([])
        assert batch == []


class TestSaveLoad:
    def test_save_load_roundtrip_encode(self, trained_tokenizer):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trained_tokenizer.save(tmp_dir)
            new_tok = FishingTokenizer(vocab_size=500)
            new_tok.load(tmp_dir)
            text = "щука на спиннинг"
            ids_original = trained_tokenizer.encode(text)
            ids_loaded = new_tok.encode(text)
            assert ids_original == ids_loaded

    def test_save_load_roundtrip_decode(self, trained_tokenizer):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trained_tokenizer.save(tmp_dir)
            new_tok = FishingTokenizer(vocab_size=500)
            new_tok.load(tmp_dir)
            text = "щука на спиннинг"
            ids_loaded = new_tok.encode(text)
            decoded = new_tok.decode(ids_loaded)
            assert decoded == text

    def test_save_load_syncs_vocab_size(self, trained_tokenizer):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trained_tokenizer.save(tmp_dir)
            new_tok = FishingTokenizer(vocab_size=999)
            new_tok.load(tmp_dir)
            assert new_tok.vocab_size == new_tok.get_vocab_size()


class TestLookups:
    def test_token_to_id_known(self, trained_tokenizer):
        tid = trained_tokenizer.token_to_id("<unk>")
        assert tid is not None

    def test_token_to_id_unknown(self, trained_tokenizer):
        tid = trained_tokenizer.token_to_id("<несуществует_999>")
        assert tid is None

    def test_id_to_token_known(self, trained_tokenizer):
        tid = trained_tokenizer.token_to_id("<unk>")
        token = trained_tokenizer.id_to_token(tid)
        assert token == "<unk>"

    def test_id_to_token_unknown(self, trained_tokenizer):
        token = trained_tokenizer.id_to_token(999999)
        assert token is None


class TestNormalization:
    def test_nfc_after_training(self, trained_tokenizer):
        composed = "\u0401"
        decomposed = "\u0415\u0308"
        ids_composed = trained_tokenizer.encode(composed)
        ids_decomposed = trained_tokenizer.encode(decomposed)
        assert ids_composed == ids_decomposed

    def test_nfc_after_load(self, trained_tokenizer):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trained_tokenizer.save(tmp_dir)
            loaded = FishingTokenizer(vocab_size=500)
            loaded.load(tmp_dir)
            composed = "\u0401"
            decomposed = "\u0415\u0308"
            ids_composed = loaded.encode(composed)
            ids_decomposed = loaded.encode(decomposed)
            assert ids_composed == ids_decomposed
