import pytest

from tests.conftest import fake_get_embeddings


@pytest.fixture(autouse=True)
def _fresh_client(monkeypatch):
    import app.qdrant_store as qdrant_store_module
    monkeypatch.setattr(qdrant_store_module, "_client", None)
    yield


def test_add_and_search_similar_returns_closest_match():
    from app.qdrant_store import add_documents, search_similar

    docs = [
        {"content": "flu symptoms include fever and cough", "metadata": {"condition": "flu"}},
        {"content": "diabetes is managed with insulin and diet", "metadata": {"condition": "diabetes"}},
    ]
    embeddings = fake_get_embeddings([d["content"] for d in docs])
    add_documents(docs, embeddings)

    query_embedding = fake_get_embeddings(["flu symptoms include fever and cough"])[0]
    results = search_similar(query_embedding, limit=1)

    assert len(results) == 1
    assert results[0]["metadata"]["condition"] == "flu"


def test_history_search_is_scoped_to_patient_id():
    from app.qdrant_store import add_history_documents, search_history

    docs = [
        {"content": "patient A note", "metadata": {"patient_id": "a", "is_chronic": False}},
        {"content": "patient B note", "metadata": {"patient_id": "b", "is_chronic": False}},
    ]
    embeddings = fake_get_embeddings([d["content"] for d in docs])
    add_history_documents(docs, embeddings)

    query_embedding = fake_get_embeddings(["some symptom query"])[0]
    results = search_history("a", query_embedding, limit=5)

    assert len(results) == 1
    assert results[0]["metadata"]["patient_id"] == "a"


def test_chronic_history_filters_by_flag():
    from app.qdrant_store import add_history_documents, get_chronic_history

    docs = [
        {"content": "ongoing condition", "metadata": {"patient_id": "a", "is_chronic": True}},
        {"content": "one-off symptom", "metadata": {"patient_id": "a", "is_chronic": False}},
    ]
    embeddings = fake_get_embeddings([d["content"] for d in docs])
    add_history_documents(docs, embeddings)

    results = get_chronic_history("a")

    assert len(results) == 1
    assert results[0]["metadata"]["is_chronic"] is True
