import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VECTOR_SIZE = 768


def _fake_vector(text: str) -> list[float]:
    """Deterministic pseudo-embedding so tests never load the real (slow, network
    dependent) PubMedBERT model. Different text -> different vector, so similarity
    search still behaves sensibly."""
    rng = random.Random(hash(text) & 0xFFFFFFFF)
    vec = [rng.uniform(-1, 1) for _ in range(VECTOR_SIZE)]
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def fake_get_embeddings(texts):
    return [_fake_vector(t) for t in texts]


def fake_get_embedding(text):
    return _fake_vector(text)


class RaisingAsyncClient:
    """Stand-in for httpx.AsyncClient that always fails fast instead of making a
    real network call, so tests deterministically exercise the app's documented
    fallback behavior (see rag_chain._fallback_query_response)."""

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, *args, **kwargs):
        raise RuntimeError("network calls are disabled in tests")


class _RaisingHttpxModule:
    AsyncClient = RaisingAsyncClient


@pytest.fixture
def client(monkeypatch):
    import app.main as main_module
    import app.qdrant_store as qdrant_store_module
    import app.rag_chain as rag_chain_module

    # Fresh in-memory Qdrant per test for isolation.
    monkeypatch.setattr(qdrant_store_module, "_client", None)

    # Small, fast corpus + no real model / network calls.
    monkeypatch.setattr(main_module, "MAX_MEDICAL_DOCS", 25)
    monkeypatch.setattr(main_module, "get_embeddings", fake_get_embeddings)
    monkeypatch.setattr(rag_chain_module, "get_embeddings", fake_get_embeddings)
    monkeypatch.setattr(rag_chain_module, "get_embedding", fake_get_embedding)
    monkeypatch.setattr(rag_chain_module, "httpx", _RaisingHttpxModule)
    monkeypatch.setattr(rag_chain_module, "OPENROUTER_API_KEY", "")

    from fastapi.testclient import TestClient
    with TestClient(main_module.app) as test_client:
        yield test_client
