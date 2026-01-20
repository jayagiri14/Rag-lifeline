from pydantic import BaseModel
from typing import Optional
from datetime import date


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3


class HistoryInsightRequest(BaseModel):
    patient_id: str
    symptoms: str
    top_k: Optional[int] = 6


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


class PrescriptionUploadResponse(BaseModel):
    status: str
    patient_id: str
    stored: int
    engine: str
    structured: dict


class HistoryInsightSource(BaseModel):
    summary: str
    date: Optional[str] = None
    is_chronic: bool = False
    type: Optional[str] = None
    score: float
    raw_text: Optional[str] = None


class HistoryInsightResponse(BaseModel):
    insight: str
    history_used: list[HistoryInsightSource]
    model: str
    usage: Optional[dict] = None
    disclaimer: str


class LoadDataResponse(BaseModel):
    status: str
    documents_added: int
