from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.models import QueryRequest, QueryResponse, HealthResponse, LoadDataResponse
from app.rag_chain import query_rag
from app.qdrant_store import get_qdrant_client, ensure_collection_exists, add_documents, get_collection_count
from app.embeddings import get_embeddings
from app.medical_data import get_medical_documents


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
        documents = get_medical_documents()
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
        documents = get_medical_documents()
        texts = [doc["content"] for doc in documents]
        embeddings = get_embeddings(texts)
        count = add_documents(documents, embeddings)
        return LoadDataResponse(
            status="success",
            documents_added=count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
