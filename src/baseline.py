"""Baseline-классификация по CLS-эмбеддингам без дообучения модели."""

import numpy as np
import pandas as pd

from src.embeddings import get_embeddings

LABEL_NAMES = {-1: "negative", 0: "neutral", 1: "positive"}


def load_twitter(path):
    """Возвращает тексты и метки −1/0/1 без пустых строк."""
    frame = pd.read_csv(path)
    frame = frame.dropna(subset=["clean_text", "category"])
    text = frame["clean_text"].astype(str).str.strip()
    label = frame["category"].astype(int)
    keep = text.ne("") & label.isin(LABEL_NAMES)
    texts = text[keep].tolist()
    labels = label[keep].tolist()
    return texts, labels


def get_cls_embeddings(texts, tokenizer, model, batch_size: int = 32, progress_every: int = 2048):
    """CLS-эмбеддинги батчами. Печатает прогресс каждые progress_every текстов."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    step = max(batch_size, (progress_every // batch_size) * batch_size)
    chunks = []
    for start in range(0, len(texts), step):
        chunk = texts[start : start + step]
        chunks.append(get_embeddings(chunk, tokenizer, model, batch_size=batch_size))
        done = min(start + step, len(texts))
        print(f"{done}/{len(texts)}", flush=True)

    return np.vstack(chunks)
