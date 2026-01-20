from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime

from app.models import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    LoadDataResponse,
    PrescriptionUploadResponse,
    HistoryInsightRequest,
    HistoryInsightResponse,
)
from app.rag_chain import query_rag, ingest_prescription_text, query_history_correlation
from app.qdrant_store import (
    get_qdrant_client,
    ensure_collection_exists,
    add_documents,
    get_collection_count,
)
from app.embeddings import get_embeddings
from app.medical_data import get_medical_documents
from app.ocr_utils import extract_text_from_image, OCRError


def _normalize_documents(raw_docs):
    """Convert raw medical documents into dicts with content/metadata.

    The source file is large and may contain items as strings, tuples, or dicts.
    This normalizer makes sure we always have {"content": str, "metadata": dict}.
    """
    normalized = []
    for doc in raw_docs:
        if isinstance(doc, dict):
            content = doc.get("content") or doc.get("text") or ""
            metadata = doc.get("metadata") or {}
        elif isinstance(doc, (list, tuple)):
            content = doc[0] if len(doc) > 0 else ""
            metadata = doc[1] if len(doc) > 1 and isinstance(doc[1], dict) else {}
        elif isinstance(doc, str):
            content = doc
            metadata = {}
        else:
            content = str(doc)
            metadata = {}

        if not isinstance(metadata, dict):
            metadata = {"info": str(metadata)}

        if content:
            normalized.append({"content": str(content), "metadata": metadata})
    return normalized


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup."""
    print("🚀 Starting Medical RAG System...")
    
    # Initialize Qdrant (in-memory)
    get_qdrant_client()
    ensure_collection_exists()
    
    # Auto-load medical data if collection is empty
    if get_collection_count() == 0:
        print("📚 Loading medical knowledge base...")
        documents = _normalize_documents(get_medical_documents())
        texts = [doc["content"] for doc in documents]
        embeddings = get_embeddings(texts)
        add_documents(documents, embeddings)
        print(f"✅ Loaded {len(documents)} medical documents")
    
    yield
    print("👋 Shutting down Medical RAG System...")


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
        documents = _normalize_documents(get_medical_documents())
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
