from qdrant_client import QdrantClient
from qdrant_client.http import models
from uuid import uuid4
from datetime import datetime
from app.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME,
    PATIENT_HISTORY_COLLECTION,
)

_client = None

# Vector size for embeddings
VECTOR_SIZE = 768

def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant client instance."""
    global _client
    if _client is None:
        if QDRANT_URL and QDRANT_API_KEY:
            # Use Qdrant Cloud
            _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            # Use local in-memory Qdrant (no server needed!)
            _client = QdrantClient(":memory:")
    return _client

def ensure_collection_exists():
    """Create the collection if it doesn't exist and ensure payload indexes."""
    client = get_qdrant_client()

    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    def _ensure(name: str):
        if name not in collection_names:
            client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Created collection: {name}")

    def _ensure_index(name: str, field: str, schema: models.PayloadSchemaType):
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=schema,
            )
        except Exception:
            # Index already exists or cannot be created; ignore so startup proceeds.
            pass

    _ensure(COLLECTION_NAME)
    _ensure(PATIENT_HISTORY_COLLECTION)

    # Needed for history filters to avoid Qdrant "index required" errors.
    _ensure_index(PATIENT_HISTORY_COLLECTION, "metadata.patient_id", models.PayloadSchemaType.KEYWORD)
    _ensure_index(PATIENT_HISTORY_COLLECTION, "metadata.is_chronic", models.PayloadSchemaType.BOOL)
    return True

def add_documents(documents: list[dict], embeddings: list[list[float]]):
    """Add documents to Qdrant collection."""
    client = get_qdrant_client()
    ensure_collection_exists()
    
    points = [
        models.PointStruct(
            id=str(uuid4()),
            vector=embedding,
            payload=doc
        )
        for doc, embedding in zip(documents, embeddings)
    ]
    
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)

def search_similar(query_embedding: list[float], limit: int = 5) -> list[dict]:
    """Search for similar documents."""
    client = get_qdrant_client()
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=limit,
    )
    
    return [
        {
            "content": hit.payload.get("content", ""),
            "metadata": hit.payload.get("metadata", {}),
            "score": hit.score
        }
        for hit in results.points
    ]


# ---------------- Patient History Helpers -----------------

def add_history_documents(documents: list[dict], embeddings: list[list[float]]):
    """Add patient history documents to history collection."""
    client = get_qdrant_client()
    ensure_collection_exists()

    points = [
        models.PointStruct(
            id=str(uuid4()),
            vector=embedding,
            payload=doc
        )
        for doc, embedding in zip(documents, embeddings)
    ]

    client.upsert(collection_name=PATIENT_HISTORY_COLLECTION, points=points)
    return len(points)


def search_history(patient_id: str, query_embedding: list[float], limit: int = 8) -> list[dict]:
    """Search patient history by similarity limited to the patient id."""
    client = get_qdrant_client()

    results = client.query_points(
        collection_name=PATIENT_HISTORY_COLLECTION,
        query=query_embedding,
        limit=limit,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="metadata.patient_id", match=models.MatchValue(value=patient_id))]
        ),
    )

    return [
        {
            "content": hit.payload.get("content", ""),
            "metadata": hit.payload.get("metadata", {}),
            "score": hit.score,
        }
        for hit in results.points
    ]


def get_chronic_history(patient_id: str, limit: int = 20) -> list[dict]:
    """Get chronic history entries for a patient (no similarity filter)."""
    client = get_qdrant_client()

    results = client.query_points(
        collection_name=PATIENT_HISTORY_COLLECTION,
        query=[0.0] * VECTOR_SIZE,
        limit=limit,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(key="metadata.patient_id", match=models.MatchValue(value=patient_id)),
                models.FieldCondition(key="metadata.is_chronic", match=models.MatchValue(value=True)),
            ]
        ),
    )

    return [
        {
            "content": hit.payload.get("content", ""),
            "metadata": hit.payload.get("metadata", {}),
            "score": hit.score,
        }
        for hit in results.points
    ]


def get_history_count() -> int:
    client = get_qdrant_client()
    try:
        info = client.get_collection(PATIENT_HISTORY_COLLECTION)
        return info.points_count
    except Exception:
        return 0

def get_collection_count() -> int:
    """Get the number of documents in the collection."""
    client = get_qdrant_client()
    try:
        info = client.get_collection(COLLECTION_NAME)
        return info.points_count
    except Exception:
        return 0
