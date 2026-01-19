from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import QDRANT_HOST, QDRANT_PORT, QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME

_client = None

# Vector size for bert-base-uncased model
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
    """Create the collection if it doesn't exist."""
    client = get_qdrant_client()
    
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )
        print(f"Created collection: {COLLECTION_NAME}")
    return True

def add_documents(documents: list[dict], embeddings: list[list[float]]):
    """Add documents to Qdrant collection."""
    client = get_qdrant_client()
    ensure_collection_exists()
    
    points = [
        models.PointStruct(
            id=i,
            vector=embedding,
            payload=doc
        )
        for i, (doc, embedding) in enumerate(zip(documents, embeddings))
    ]
    
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)

def search_similar(query_embedding: list[float], limit: int = 5) -> list[dict]:
    """Search for similar documents."""
    client = get_qdrant_client()
    
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=limit
    )
    
    return [
        {
            "content": hit.payload.get("content", ""),
            "metadata": hit.payload.get("metadata", {}),
            "score": hit.score
        }
        for hit in results
    ]

def get_collection_count() -> int:
    """Get the number of documents in the collection."""
    client = get_qdrant_client()
    try:
        info = client.get_collection(COLLECTION_NAME)
        return info.points_count
    except Exception:
        return 0
