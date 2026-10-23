"""Attention-матрицы DistilBERT и тепловые карты. День 3."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModel

from src.tokenization import MODEL_NAME

FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "attention"


def load_attention_model(model_name: str = MODEL_NAME):
    """Загружает модель так, чтобы forward возвращал attention всех слоёв."""
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()
    return model


def visualize_attention(
    tokens,
    attention,
    tokenizer,
    layer: int = 0,
    head: int = 0,
    save_dir: Path | None = None,
    filename: str | None = None,
) -> Path:
    """Рисует heatmap одной головы и сохраняет PNG."""
    attn = attention[layer][0, head]
    token_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn.detach().cpu().numpy(),
        xticklabels=token_list,
        yticklabels=token_list,
        cmap="viridis",
        cbar=True,
    )
    plt.title(f"Attention - Layer {layer}, Head {head}")
    plt.xlabel("Keys")
    plt.ylabel("Queries")
    plt.tight_layout()

    directory = Path(save_dir) if save_dir is not None else FIGURES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (filename or f"attention_layer{layer}_head{head}.png")
    fig = plt.gcf()
    fig.savefig(path)

    try:
        from IPython.display import display

        display(fig)
    except Exception:
        pass

    plt.close(fig)
    return path
