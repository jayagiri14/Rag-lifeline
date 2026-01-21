"""
PubMed-BERT embeddings for medical text.
Simple, deterministic wrapper suitable for Qdrant.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Model name
_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

# Lazy-loaded globals
_tokenizer = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model():
    """Load tokenizer and model once."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModel.from_pretrained(_MODEL_NAME)
        _model.to(_device)
        _model.eval()


def _mean_pooling(model_output, attention_mask):
    """Mean pooling with attention mask."""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    counts = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate PubMed-BERT embeddings for a list of texts.
    Returns L2-normalized vectors.
    """
    _load_model()

    with torch.no_grad():
        encoded = _tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        encoded = {k: v.to(_device) for k, v in encoded.items()}
        model_output = _model(**encoded)

        embeddings = _mean_pooling(model_output, encoded["attention_mask"])
        embeddings = embeddings.cpu().numpy()

    # L2 normalize (important for cosine similarity in Qdrant)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    return embeddings.tolist()


def get_embedding(text: str) -> list[float]:
    """Generate embedding for a single text."""
    return get_embeddings([text])[0]