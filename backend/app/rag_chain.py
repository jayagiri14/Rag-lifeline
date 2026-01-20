import httpx
import json
from datetime import datetime, timezone
from typing import Tuple, Optional
from app.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    LLM_MODEL,
    HISTORY_RECENT_DAYS,
    HISTORY_TOP_K,
)
from app.embeddings import get_embedding, get_embeddings
from app.qdrant_store import (
    search_similar,
    search_history,
    get_chronic_history,
    add_history_documents,
)

SYSTEM_PROMPT = """You are a helpful medical assistant AI. Your role is to provide information about symptoms, conditions, and general health advice based on the medical knowledge provided to you.

Important guidelines:
1. Always base your answers on the provided context/knowledge.
2. Be empathetic and clear in your responses.
3. Always recommend consulting a healthcare professional for proper diagnosis and treatment.
4. Never provide definitive diagnoses - only suggest possibilities.
5. If you don't have enough information, say so clearly.
6. Mention when symptoms require urgent medical attention.

Remember: You are providing general health information, not medical advice. Always encourage users to seek professional medical care."""


async def generate_response(query: str, context: list[dict]) -> dict:
    """Generate a response using OpenRouter DeepSeek R1."""
    
    # Format context for the prompt
    context_text = "\n\n".join([
        f"--- Medical Information ---\n{doc['content']}"
        for doc in context
    ])
    
    user_message = f"""Based on the following medical knowledge, please help answer the user's question.

MEDICAL KNOWLEDGE:
{context_text}

USER'S QUESTION: {query}

Please provide a helpful, informative response based on the knowledge above. Remember to recommend consulting a healthcare professional."""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Medical RAG Assistant"
    }
    
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7,
        "max_tokens": 1500
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            error_detail = response.text
            print(f"OpenRouter API Error: {response.status_code} - {error_detail}")
            raise Exception(f"OpenRouter API error: {response.status_code} - {error_detail}")
        
        result = response.json()
        return {
            "response": result["choices"][0]["message"]["content"],
            "model": result.get("model", LLM_MODEL),
            "usage": result.get("usage", {})
        }


# ---------------- Prescription Structuring -----------------

STRUCTURE_SYSTEM_PROMPT = """You are a medical scribe. Extract a minimal JSON summary from a prescription note.
Fields:
- diagnosis: list of diagnoses (strings)
- medicines: list of medicine strings
- is_chronic: boolean (true if chronic/long-term condition is mentioned)
- date: ISO date string if present, else null
- doctor_notes: brief free-text notes if present
- raw_text: original text echoed back
Return ONLY JSON."""


async def structure_prescription_text(raw_text: str) -> dict:
    """Use the LLM to structure prescription text into a JSON record."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Medical RAG Assistant",
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": STRUCTURE_SYSTEM_PROMPT},
            {"role": "user", "content": raw_text},
        ],
        "temperature": 0.2,
        "max_tokens": 400,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )

    if response.status_code != 200:
        raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")

    try:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        structured = json.loads(content)
    except Exception as exc:  # pragma: no cover - defensive
        raise Exception(f"Failed to parse LLM structuring output: {exc}") from exc

    # Normalize
    structured.setdefault("diagnosis", [])
    structured.setdefault("medicines", [])
    structured.setdefault("is_chronic", False)
    structured.setdefault("raw_text", raw_text)
    return structured


def _build_history_payload(patient_id: str, structured: dict, raw_text: str) -> dict:
    date_str = structured.get("date") or datetime.now(timezone.utc).date().isoformat()
    try:
        dt = datetime.fromisoformat(date_str)
    except Exception:
        dt = datetime.now(timezone.utc)
    payload = {
        "content": " | ".join([
            "Diagnoses: " + ", ".join(structured.get("diagnosis", []) or ["unknown"]),
            "Medicines: " + ", ".join(structured.get("medicines", []) or ["unspecified"]),
            structured.get("doctor_notes") or "",
        ]),
        "metadata": {
            "patient_id": patient_id,
            "date": dt.date().isoformat(),
            "date_ts": dt.timestamp(),
            "is_chronic": bool(structured.get("is_chronic", False)),
            "type": "prescription",
            "diagnosis": structured.get("diagnosis", []),
            "medicines": structured.get("medicines", []),
            "raw_text": raw_text,
        },
    }
    return payload


def _score_history_entry(entry: dict) -> float:
    base = entry.get("score", 0.0) or 0.0
    meta = entry.get("metadata", {})
    if meta.get("is_chronic"):
        base += 1.0
    ts = meta.get("date_ts")
    if ts:
        days = (datetime.now(timezone.utc).timestamp() - ts) / 86400
        if days <= 30:
            base += 0.5
        elif days <= HISTORY_RECENT_DAYS:
            base += 0.2
        else:
            base -= 0.2
    return base


def _summarize_history_for_llm(history: list[dict]) -> Tuple[str, list[dict]]:
    sorted_items = sorted(history, key=_score_history_entry, reverse=True)
    top_items = sorted_items[:HISTORY_TOP_K]

    lines = []
    for item in top_items:
        meta = item.get("metadata", {})
        lines.append(
            f"Date: {meta.get('date', 'unknown')}; Chronic: {meta.get('is_chronic', False)}; "
            f"Diagnoses: {', '.join(meta.get('diagnosis', []))}; Medicines: {', '.join(meta.get('medicines', []))}; "
            f"Notes: {meta.get('raw_text', '')[:200]}"
        )
    return "\n".join(lines), top_items


def _fallback_history_message(symptoms: str, items: list[dict], reason: Optional[str] = None) -> str:
    reason_text = "LLM service unavailable."
    if reason:
        trimmed = reason.strip().replace("\n", " ")
        reason_text = f"LLM service unavailable ({trimmed[:180]})."

    if not items:
        history_lines = ["No structured history entries are stored for this patient yet."]
    else:
        history_lines = []
        for item in items:
            meta = item.get("metadata", {})
            diagnoses = ", ".join(meta.get("diagnosis", [])) or "diagnosis unspecified"
            medicines = ", ".join(meta.get("medicines", [])) or "medicines unspecified"
            history_lines.append(
                f"- {meta.get('date', 'unknown')}: chronic={meta.get('is_chronic', False)}; "
                f"diagnoses={diagnoses}; medicines={medicines}"
            )

    return "\n".join([
        reason_text,
        f"Symptoms reported: {symptoms or 'not specified'}.",
        "Key history signals:",
        *history_lines,
        "Consider validating with a clinician once the full LLM service is available."
    ])


HISTORY_INSIGHT_PROMPT = """You are a medical reasoning assistant.
Use the patient's past medical history and current symptoms to highlight possible correlations and risk factors.
- Do NOT provide definitive diagnoses.
- Emphasize how chronic diseases interact with current symptoms.
- If evidence is weak, say so clearly.
- Include a brief disclaimer.
Return concise paragraphs.
"""


async def generate_history_insight(symptoms: str, history: list[dict]) -> dict:
    history_text, used_items = _summarize_history_for_llm(history)
    user_message = f"""Current symptoms: {symptoms}

Relevant history:
{history_text}

Provide a history-based medical insight (not a diagnosis)."""

    if not OPENROUTER_API_KEY:
        return {
            "response": _fallback_history_message(symptoms, used_items, "missing OPENROUTER_API_KEY"),
            "model": "local-fallback",
            "usage": {"error": "openrouter_api_key_missing"},
            "history_used": used_items,
        }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Medical RAG Assistant",
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": HISTORY_INSIGHT_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,
        "max_tokens": 700,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
        result = response.json()
        return {
            "response": result["choices"][0]["message"]["content"],
            "model": result.get("model", LLM_MODEL),
            "usage": result.get("usage", {}),
            "history_used": used_items,
        }
    except Exception as exc:
        reason = str(exc)
        return {
            "response": _fallback_history_message(symptoms, used_items, reason),
            "model": "local-fallback",
            "usage": {"error": reason[:200]},
            "history_used": used_items,
        }


async def ingest_prescription_text(patient_id: str, raw_text: str) -> Tuple[dict, int]:
    structured = await structure_prescription_text(raw_text)
    payload = _build_history_payload(patient_id, structured, raw_text)
    embedding = get_embeddings([payload["content"]])[0]
    stored = add_history_documents([payload], [embedding])
    return structured, stored


async def query_rag(query: str, top_k: int = 3) -> dict:
    """Main RAG function: retrieve relevant docs and generate response."""
    
    # Step 1: Generate embedding for the query
    query_embedding = get_embedding(query)
    
    # Step 2: Search for relevant documents
    relevant_docs = search_similar(query_embedding, limit=top_k)
    
    if not relevant_docs:
        return {
            "response": "I don't have enough medical information to answer your question. Please consult a healthcare professional.",
            "sources": [],
            "model": LLM_MODEL
        }
    
    # Step 3: Generate response using LLM
    llm_result = await generate_response(query, relevant_docs)
    
    return {
        "response": llm_result["response"],
        "sources": [
            {
                "content": doc["content"][:200] + "...",
                "condition": doc["metadata"].get("condition", "Unknown"),
                "relevance_score": doc["score"]
            }
            for doc in relevant_docs
        ],
        "model": llm_result["model"],
        "usage": llm_result.get("usage", {})
    }


async def query_history_correlation(patient_id: str, symptoms: str, top_k: int = HISTORY_TOP_K) -> dict:
    """History-aware insight generation."""
    symptom_embedding = get_embedding(symptoms)

    similar = search_history(patient_id, symptom_embedding, limit=top_k + 4)
    chronic = get_chronic_history(patient_id)

    combined = similar + [item for item in chronic if item not in similar]
    if not combined:
        return {
            "response": "No prior history found for this patient to correlate with current symptoms.",
            "sources": [],
            "model": LLM_MODEL,
        }

    llm_result = await generate_history_insight(symptoms, combined)

    return {
        "response": llm_result["response"],
        "sources": [
            {
                "summary": item.get("content", "")[:200],
                "date": item.get("metadata", {}).get("date"),
                "is_chronic": item.get("metadata", {}).get("is_chronic", False),
                "type": item.get("metadata", {}).get("type"),
                "score": _score_history_entry(item),
                "raw_text": item.get("metadata", {}).get("raw_text"),
            }
            for item in llm_result["history_used"]
        ],
        "model": llm_result["model"],
        "usage": llm_result.get("usage", {}),
    }
