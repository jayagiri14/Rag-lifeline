import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
from app.audio_utils import extract_text_from_audio, AudioError
from app.config import MAX_MEDICAL_DOCS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("medical_rag")

from app.models import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    LoadDataResponse,
    PrescriptionUploadResponse,
    HistoryInsightRequest,
    HistoryInsightResponse,
)
from app.rag_chain import query_rag, ingest_prescription_text,ingest_audio_symptom, query_history_correlation
from app.qdrant_store import (
    get_qdrant_client,
    ensure_collection_exists,
    add_documents,
    get_collection_count,
)
from app.embeddings import get_embeddings
from app.medical_data import get_medical_documents
from app.ocr_utils import extract_text_from_image, OCRError


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup."""
    logger.info("Starting Medical RAG System...")

    # Initialize Qdrant (in-memory)
    get_qdrant_client()
    ensure_collection_exists()

    # Auto-load medical data if collection is empty
    if get_collection_count() == 0:
        logger.info("Loading medical knowledge base (max %d documents)...", MAX_MEDICAL_DOCS)
        documents = get_medical_documents(limit=MAX_MEDICAL_DOCS)
        texts = [doc["content"] for doc in documents]
        embeddings = get_embeddings(texts)
        add_documents(documents, embeddings)
        logger.info("Loaded %d medical documents", len(documents))

    yield
    logger.info("Shutting down Medical RAG System...")


app = FastAPI(
    title="Medical RAG API",
    description="A medical question-answering system using RAG with Qdrant and DeepSeek R1",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        documents_loaded=get_collection_count()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        documents_loaded=get_collection_count()
    )


@app.post("/query", response_model=QueryResponse)
async def query_medical(request: QueryRequest):
    """Query the medical RAG system."""
    try:
        result = await query_rag(request.query, request.top_k)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload-data", response_model=LoadDataResponse)
async def reload_medical_data():
    """Reload the medical knowledge base."""
    try:
        documents = get_medical_documents(limit=MAX_MEDICAL_DOCS)
        texts = [doc["content"] for doc in documents]
        embeddings = get_embeddings(texts)
        count = add_documents(documents, embeddings)
        return LoadDataResponse(
            status="success",
            documents_added=count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/history/prescription", response_model=PrescriptionUploadResponse)
async def upload_prescription(patient_id: str = Form(...), file: UploadFile = File(...)):
    """Upload a prescription image, run OCR + structuring, and store in history."""
    try:
        file_bytes = await file.read()
        text, engine = extract_text_from_image(file_bytes)
        structured, stored = await ingest_prescription_text(patient_id, text)
        return PrescriptionUploadResponse(
            status="stored",
            patient_id=patient_id,
            stored=stored,
            engine=engine,
            structured=structured,
        )
    except OCRError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/history/insight", response_model=HistoryInsightResponse)
async def history_insight(request: HistoryInsightRequest):
    """Generate history-based medical insight for a patient's symptoms."""
    try:
        result = await query_history_correlation(request.patient_id, request.symptoms, request.top_k)
        return HistoryInsightResponse(
            insight=result["response"],
            history_used=result.get("sources", []),
            model=result.get("model", "unknown"),
            usage=result.get("usage"),
            disclaimer="⚠️ This is a history-based insight, not a diagnosis. Consult a clinician.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/history/audio")
async def upload_audio_description(
    patient_id: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        audio_bytes = await file.read()

        transcript = await extract_text_from_audio(audio_bytes)
        logger.info("Audio transcript for patient %s: %s", patient_id, transcript)

        stored = await ingest_audio_symptom(patient_id, transcript)

        return {
            "status": "stored",
            "patient_id": patient_id,
            "stored": stored,
            "engine": "whisper",
            "transcript": transcript,
        }

    except Exception as e:
        logger.exception("Audio history ingestion failed for patient %s", patient_id)
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
