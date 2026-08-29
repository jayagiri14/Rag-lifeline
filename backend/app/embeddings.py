"""
PubMedBERT-MSMARCO embeddings for medical text.
Simple, deterministic wrapper suitable for Qdrant.

Model choice: benchmarked against raw PubMedBERT and all-MiniLM-L6-v2 in
eval/retrieval_eval.py. Raw PubMedBERT (masked-LM only, no retrieval training)
scored worst (recall@1 0.60). This checkpoint adds MS MARCO retrieval
fine-tuning on top of PubMedBERT's biomedical vocabulary (recall@1 0.85),
chosen over the higher-scoring general-purpose MiniLM (recall@1 0.94) to keep
domain-specific (medical) representations for this use case. See README
"Retrieval Evaluation" section for full numbers and the tradeoff discussion.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_BATCH_SIZE

_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

# Lazy-loaded global
_model = None


def _load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def get_embeddings(texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> list[list[float]]:
    """
    Generate PubMedBERT-MSMARCO embeddings for a list of texts.
    Returns L2-normalized vectors.
    """
    if not texts:
        return []

    model = _load_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    # L2 normalize (important for cosine similarity in Qdrant)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    return embeddings.tolist()


def get_embedding(text: str) -> list[float]:
    """Generate embedding for a single text."""
    return get_embeddings([text])[0]
