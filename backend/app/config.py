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

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM Model (DeepSeek R1 via OpenRouter)
LLM_MODEL = "deepseek/deepseek-r1"

# History retrieval settings
HISTORY_RECENT_DAYS = int(os.getenv("HISTORY_RECENT_DAYS", 180))
HISTORY_TOP_K = int(os.getenv("HISTORY_TOP_K", 6))
