"""CLS-эмбеддинги DistilBERT. День 2 использует tokenize_texts из дня 1."""

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel

from src.tokenization import MODEL_NAME, tokenize_texts


def load_model(model_name: str = MODEL_NAME):
    """Загружает модель и переводит её в режим оценки."""
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return model


def get_embeddings(texts, tokenizer, model, batch_size: int = 32):
    """Возвращает CLS-эмбеддинги текстов, форма (n_texts, hidden_size)."""
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        tokens = tokenize_texts(batch_texts, tokenizer)

        with torch.no_grad():
            outputs = model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def similarity(text1: str, text2: str, tokenizer, model) -> float:
    """Косинусное сходство CLS-эмбеддингов двух текстов."""
    emb = get_embeddings([text1, text2], tokenizer, model)
    return float(cosine_similarity(emb[0:1], emb[1:2])[0][0])
