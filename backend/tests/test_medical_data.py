from app.medical_data import MEDICAL_KNOWLEDGE, get_medical_documents


def test_corpus_has_thousands_of_qa_pairs():
    """Regression test for the ingestion bug: MEDICAL_KNOWLEDGE is a nested
    list-of-lists, and a naive iteration over it previously produced a single
    malformed document instead of one document per Q&A pair."""
    docs = get_medical_documents()
    assert len(docs) > 15000


def test_documents_have_real_text_content_not_dict_repr():
    docs = get_medical_documents(limit=5)
    for doc in docs:
        assert isinstance(doc["content"], str)
        assert doc["content"].startswith("Q:")
        assert "{'Question'" not in doc["content"]
        assert "metadata" in doc and isinstance(doc["metadata"], dict)


def test_limit_caps_document_count():
    docs = get_medical_documents(limit=10)
    assert len(docs) == 10


def test_skips_entries_missing_question_or_response():
    docs = get_medical_documents(limit=3)
    for doc in docs:
        assert doc["metadata"]["question"]
        assert "A:" in doc["content"]
