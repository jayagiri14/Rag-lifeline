# Medical RAG System

A simple medical RAG (Retrieval-Augmented Generation) system that answers questions about symptoms using:
- **Qdrant** - Vector database for storing medical knowledge embeddings
- **LangChain** - For building the RAG pipeline
- **OpenRouter DeepSeek R1** - LLM for generating responses
- **FastAPI** - Backend API
- **React + Vite + TailwindCSS** - Frontend

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- OpenRouter API key (get one at https://openrouter.ai/keys)

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API key
copy .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### 2. Configure API Key

Edit `backend/.env`:
```
OPENROUTER_API_KEY=your_actual_api_key_here
```

### 3. Start Backend

```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

The backend will:
- Start on http://localhost:8000
- Auto-load 15 medical conditions into in-memory Qdrant
- No separate Qdrant server needed!

### 4. Frontend Setup (new terminal)

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

Frontend runs on http://localhost:3000

## 📁 Project Structure

```
clinicalragp2/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app
│   │   ├── config.py        # Configuration
│   │   ├── embeddings.py    # Sentence transformer embeddings
│   │   ├── qdrant_store.py  # Qdrant vector store
│   │   ├── rag_chain.py     # RAG logic with OpenRouter
│   │   ├── medical_data.py  # Medical knowledge base
│   │   └── models.py        # Pydantic models
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # Main React component
│   │   ├── api.ts           # API client
│   │   └── main.tsx         # Entry point
│   ├── package.json
│   └── vite.config.ts
└── README.md
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Health check with doc count |
| POST | `/query` | Query the RAG system |
| POST | `/reload-data` | Reload medical knowledge |

### Query Example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "I have a headache and nausea", "top_k": 3}'
```

## 💡 How It Works

1. **User asks a question** about their symptoms
2. **Embedding Generation**: Query is converted to vector using SentenceTransformer
3. **Vector Search**: Qdrant finds similar medical documents
4. **Context Assembly**: Relevant medical info is gathered
5. **LLM Generation**: DeepSeek R1 generates a helpful response
6. **Response**: User receives medical information with sources

## 🏥 Medical Conditions Included

- Common Cold
- Influenza (Flu)
- Migraine
- Gastroenteritis
- Allergic Rhinitis
- Type 2 Diabetes
- Hypertension
- Anxiety Disorder
- UTI
- Asthma
- Lower Back Pain
- Depression
- GERD (Acid Reflux)
- Osteoarthritis
- Insomnia

## ⚠️ Disclaimer

This is an educational project. The medical information provided is for general informational purposes only and should not be considered medical advice. Always consult a qualified healthcare professional for medical concerns.

## 🔧 Customization

### Add More Medical Data

Edit `backend/app/medical_data.py` to add more conditions:

```python
{
    "content": "Your medical content here...",
    "metadata": {"condition": "Condition Name", "category": "Category"}
}
```

### Change LLM Model

Edit `backend/app/config.py`:
```python
LLM_MODEL = "deepseek/deepseek-r1"  # or other OpenRouter models
```

### Use Qdrant Cloud

Update `backend/.env`:
```
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
```
