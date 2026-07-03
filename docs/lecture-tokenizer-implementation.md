# Реализация BPE-токенизатора для FishingLLM

## Архитектура

Файл: `fishingllm/tokenizer.py`
Тесты: `tests/test_tokenizer.py`

### Класс `FishingTokenizer`

Обёртка над HuggingFace `ByteLevelBPETokenizer` с добавлением NFC-нормализации.

```python
class FishingTokenizer:
    def __init__(self, vocab_size: int = 16_000, min_frequency: int = 2)
    def train(self, files: list[str])
    def encode(self, text: str) -> list[int]
    def encode_batch(self, texts: list[str]) -> list[list[int]]
    def decode(self, ids: list[int]) -> str
    def decode_batch(self, batch: list[list[int]]) -> list[str]
    def save(self, path: str | Path)
    def load(self, path: str | Path)
    def get_vocab_size(self) -> int
    def token_to_id(self, token: str) -> int | None
    def id_to_token(self, token_id: int) -> str | None
```

### Внутреннее устройство

`ByteLevelBPETokenizer` (из библиотеки `tokenizers`) состоит из:

1. **BPE-модель** — хранит словарь токенов и правила слияния (merges)
2. **ByteLevel pre-tokenizer** — разбивает текст на слова по регулярному выражению (GPT-2 стиль), затем каждое слово превращает в последовательность GPT-2 byte-encoded символов
3. **ByteLevel decoder** — обратное преобразование: токены → byte-encoded символы → текст
4. **NFC normalizer** — нормализация Unicode до кодирования (например, «Ё» → «Ё»)

```
  ByteLevelBPETokenizer
  ┌─────────────────────────────────────┐
  │  NFC normalizer                     │
  │  ByteLevel pre-tokenizer (regex)    │
  │  BPE model (vocab + merges)         │
  │  ByteLevel decoder                  │
  └─────────────────────────────────────┘
```

### Special tokens

Шесть служебных токенов добавляются в словарь при тренировке:

| Токен | Назначение |
|-------|-----------|
| `<unk>` | Неизвестный токен. В штатной работе byte-level BPE практически не использует `<unk>`, поскольку любой Unicode-текст представим через последовательность байтов. Токен сохранён для совместимости с инфраструктурой HuggingFace и как защита при некорректной конфигурации. |
| `<s>` | Начало последовательности |
| `</s>` | Конец последовательности |
| `<\|user\|>` | Метка пользователя (ChatML) |
| `<\|assistant\|>` | Метка ассистента (ChatML) |
| `<\|system\|>` | Метка системы (ChatML) |

## Связь с теорией (Лекция 2)

| Концепт в лекции | Реализация |
|-----------------|-----------|
| Byte-level BPE (GPT-2) | `ByteLevelBPETokenizer` — 256 начальных токенов, все байты |
| Unicode нормализация (NFC) | `tokenizer.normalizer = NFC()` |
| Регулярное выражение GPT-2 | ByteLevel pre-tokenizer (встроенное regex) |
| BPE-слияния | `BpeTrainer` с `min_frequency=2` |
| vocab_size | Параметр конструктора, по умолчанию 16 000 |
| Special tokens (ChatML) | `special_tokens=[...]` при тренировке |

## Trade-offs и альтернативы

### 1. `ByteLevelBPETokenizer` vs `Tokenizer(BPE(...))` + ByteLevel вручную

**Выбрано:** `ByteLevelBPETokenizer`.

**Почему:**
- Автоматически включает все 256 байт в начальный алфавит. Ручная сборка (`Tokenizer(BPE(...))`) этого не делает — если байт не встретился в корпусе, он станет `<unk>`.
- Предоставляет единый интерфейс encode/decode/train/save/load.
- Сохраняется в стандартный формат HuggingFace (vocab.json + merges.txt).

**Альтернатива:** Ручная сборка через `Tokenizer(BPE(unk_token="<unk>"))` с добавлением ByteLevel претокенизатора и декодера. Это даёт больше контроля (например, можно добавить кастомный нормализатор), но требует ручного управления начальным алфавитом.

### 2. NFC vs другие формы нормализации

**Выбрано:** NFC.

**Trade-off:** NFC (Normalization Form C) — стандартная форма для веба и большинства текстов. NFD (Normalization Form D) используется в системах поиска для более гибкого сравнения. NFC достаточно для FishingLLM: редкие диакритические варианты (например, «Ё» как два кода) будут приведены к одной форме, что уменьшает дублирование токенов.

### Важно: NFC и сериализация

`ByteLevelBPETokenizer` не сериализует нормализатор. Метод `save_model()` сохраняет только BPE-словарь (vocab.json) и правила слияний (merges.txt). NFC normaliser — отдельный компонент поверх токенизатора. Поэтому после `load()` нормализатор всегда назначается повторно явно. Это сделано в `FishingTokenizer.load()`, что подтверждается тестом `test_save_load_roundtrip` (encode после save→load идентичен оригиналу).

### 3. Почему `ByteLevelBPETokenizer`, а не `AutoTokenizer`?

`AutoTokenizer` загружает готовую конфигурацию из предобученного токенизатора (GPT-2, LLaMA и т.д.). Мы обучаем собственный токенизатор на рыбацком корпусе, поэтому `ByteLevelBPETokenizer` — правильный выбор: он даёт контроль над каждым этапом pipeline (нормализация, пред-токенизация, BPE-словарь).

### 4. min_frequency = 2

**Выбрано:** `min_frequency=2`.

**Причина:** Пары, встречающиеся только один раз в корпусе, не дают статистически значимой информации. Слияние таких пар раздувает словарь редкими бесполезными токенами. Конкретное значение зависит от размера корпуса и желаемого размера словаря. Для FishingLLM с целевым корпусом 50+ МБ `min_frequency=2` — разумный минимум; при необходимости его можно вынести в конструктор.

## Обоснование тестов

### Основные группы

| Тест | Проверяет |
|------|----------|
| `TestFishingTokenizerInit` | Конструктор, vocab_size по умолчанию, special tokens |
| `TestTraining` | Рост словаря после тренировки, соблюдение лимита vocab_size, наличие special tokens |
| `TestEncodeDecode` | encode возвращает int IDs, roundtrip encode→decode, русский текст, пустая строка |
| `TestBatch` | encode_batch и decode_batch с несколькими текстами |
| `TestSaveLoad` | Сохранение и загрузка модели, идентичность encode после save→load |
| `TestLookups` | token_to_id/id_to_token для известных и неизвестных токенов |
| `TestNormalization` | NFC-нормализация: «Ё» (один код) = «Е» + «¨» (два кода) |

### Почему эти тесты важны

- **Roundtrip (encode→decode)**: гарантирует, что токенизатор сохраняет текст без потерь. Если encode→decode не identity, модель не сможет генерировать читаемый текст.
- **Save/Load**: токенизатор должен быть сериализуем — это критично для воспроизводимости тренировки и инференса.
- **NFC**: без нормализации редкие Unicode-варианты раздувают словарь бесполезными дубликатами.

## Использование

```python
from fishingllm.tokenizer import FishingTokenizer
from pathlib import Path

# Создание и тренировка
tokenizer = FishingTokenizer(vocab_size=16_000)
tokenizer.train(["fishing_corpus.txt"])

# Сохранение
tokenizer.save("models/tokenizer")

# Загрузка
loaded = FishingTokenizer(vocab_size=16_000)
loaded.load("models/tokenizer")

# Кодирование
ids = tokenizer.encode("Щука на спиннинг")
# [341, 890, 307, 56, ...]

# Декодирование
text = tokenizer.decode(ids)
# "Щука на спиннинг"

# Пакетная обработка
texts = ["Щука", "Окунь", "Лещ"]
batch = tokenizer.encode_batch(texts)
decoded = tokenizer.decode_batch(batch)
```

## Задел на будущее

При интеграции с моделью и батчингом понадобятся константы:

| ID | Токен | Назначение |
|----|-------|-----------|
| `pad_token_id` | `<s>` (или отдельный `<pad>`) | Выравнивание последовательностей в батче |
| `bos_token_id` | `<s>` | Начало последовательности |
| `eos_token_id` | `</s>` | Конец последовательности (стоп-генерация) |

Для ChatML-формата позже может понадобиться `<|end|>` как EOS-маркер диалога. Пока `</s>` достаточно, токен зарезервирован.

## Ссылки

- [HuggingFace tokenizers: ByteLevelBPETokenizer](https://huggingface.co/docs/tokenizers/api/tokenizer#tokenizers.implementations.ByteLevelBPETokenizer)
- [minbpe (Karpathy)](https://github.com/karpathy/minbpe) — каноническая educational реализация BPE
- [Лекция 2: Токенизация](lecture-02-tokenization.docx) — теория, лежащая в основе реализации
