import httpx
from app.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL
from app.embeddings import get_embedding
from app.qdrant_store import search_similar

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
