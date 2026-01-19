from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3


class SourceInfo(BaseModel):
    content: str
    condition: str
    relevance_score: float


class QueryResponse(BaseModel):
    response: str
    sources: list[SourceInfo]
    model: str
    usage: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    documents_loaded: int


class LoadDataResponse(BaseModel):
    status: str
    documents_added: int
