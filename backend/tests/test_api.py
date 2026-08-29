def test_health_check_reports_loaded_documents(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["documents_loaded"] == 25  # MAX_MEDICAL_DOCS is patched to 25 in tests


def test_query_falls_back_gracefully_without_llm(client):
    """No OpenRouter key + network disabled -> app should still return retrieved
    context instead of a 500, per rag_chain._fallback_query_response."""
    resp = client.post("/query", json={"query": "I have a headache and nausea", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "local-fallback"
    assert len(body["sources"]) > 0
    assert body["response"]


def test_query_rejects_missing_query_field(client):
    resp = client.post("/query", json={})
    assert resp.status_code == 422


def test_reload_data_reembeds_corpus(client):
    resp = client.post("/reload-data")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["documents_added"] == 25


def test_history_insight_with_no_prior_history(client):
    resp = client.post(
        "/history/insight",
        json={"patient_id": "no-such-patient", "symptoms": "fatigue and thirst"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "No prior history" in body["insight"]
    assert body["history_used"] == []


def test_history_insight_uses_ingested_prescription(client, monkeypatch):
    import app.main as main_module

    async def fake_ingest_prescription_text(patient_id, raw_text):
        structured = {
            "diagnosis": ["hypertension"],
            "medicines": ["lisinopril"],
            "is_chronic": True,
            "date": None,
            "doctor_notes": "stable",
            "raw_text": raw_text,
        }
        from app.rag_chain import _build_history_payload
        from app.embeddings import get_embeddings
        from app.qdrant_store import add_history_documents

        payload = _build_history_payload(patient_id, structured, raw_text)
        embedding = main_module.get_embeddings([payload["content"]])[0]
        stored = add_history_documents([payload], [embedding])
        return structured, stored

    monkeypatch.setattr(main_module, "extract_text_from_image", lambda file_bytes: ("hypertension, lisinopril", "tesseract"))
    monkeypatch.setattr(main_module, "ingest_prescription_text", fake_ingest_prescription_text)

    upload = client.post(
        "/history/prescription",
        data={"patient_id": "patient-1"},
        files={"file": ("rx.png", b"fake-image-bytes", "image/png")},
    )
    assert upload.status_code == 200
    assert upload.json()["stored"] == 1

    insight = client.post(
        "/history/insight",
        json={"patient_id": "patient-1", "symptoms": "swelling and fatigue"},
    )
    assert insight.status_code == 200
    body = insight.json()
    assert len(body["history_used"]) == 1
    assert body["history_used"][0]["is_chronic"] is True


def test_prescription_upload_returns_400_on_ocr_error(client, monkeypatch):
    import app.main as main_module
    from app.ocr_utils import OCRError

    def fake_extract(file_bytes):
        raise OCRError("No text detected in image")

    monkeypatch.setattr(main_module, "extract_text_from_image", fake_extract)

    resp = client.post(
        "/history/prescription",
        data={"patient_id": "patient-2"},
        files={"file": ("rx.png", b"not-really-an-image", "image/png")},
    )
    assert resp.status_code == 400


def test_audio_upload_stores_transcript(client, monkeypatch):
    import app.main as main_module

    async def fake_extract_text_from_audio(audio_bytes):
        return "I have been coughing for three days"

    monkeypatch.setattr(main_module, "extract_text_from_audio", fake_extract_text_from_audio)

    resp = client.post(
        "/history/audio",
        data={"patient_id": "patient-3"},
        files={"file": ("audio.webm", b"fake-audio-bytes", "audio/webm")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["transcript"] == "I have been coughing for three days"
    assert body["stored"] == 1
