"""Токенизация текстов DistilBERT. Функции дня 1 понадобятся в день 2."""

from transformers import AutoTokenizer, PreTrainedTokenizerBase

MODEL_NAME = "distilbert-base-uncased"


def load_tokenizer(model_name: str = MODEL_NAME) -> PreTrainedTokenizerBase:
    """Загружает токенизатор Hugging Face для выбранной модели."""
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_texts(texts, tokenizer, max_length: int = 128):
    """Токенизирует список текстов в батч с паддингом и обрезкой."""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def explain_tokenization(text: str, tokenizer) -> None:
    """Показывает разбиение текста на токены без специальных токенов."""
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    print(f"Исходный текст: {text}")
    print(f"Токены: {tokens}")
    print(f"IDs: {ids}")
    print(f"Количество: {len(tokens)}")
