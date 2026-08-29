import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_URL = os.getenv("QDRANT_URL", None)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

# Collection name for medical data
COLLECTION_NAME = "medical_knowledge"
PATIENT_HISTORY_COLLECTION = "patient_history"

# Embedding model (see app/embeddings.py docstring for why this was chosen
# over raw PubMedBERT and all-MiniLM-L6-v2 -- benchmarked in eval/retrieval_eval.py)
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"

# LLM Model (OpenAI OSS via OpenRouter)
LLM_MODEL="google/gemini-2.5-flash"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
# History retrieval settings
HISTORY_RECENT_DAYS = int(os.getenv("HISTORY_RECENT_DAYS", 180))
HISTORY_TOP_K = int(os.getenv("HISTORY_TOP_K", 6))

# Corpus loading: the full medical corpus is ~19.7k Q&A pairs, which is slow to
# embed on CPU at startup. Cap it for local dev; raise/remove for a full-corpus run.
MAX_MEDICAL_DOCS = int(os.getenv("MAX_MEDICAL_DOCS", 1500))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 32))
