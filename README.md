# Medical RAG Assistant

A Retrieval Augmented Generation (RAG) assistant that combines a FastAPI backend, in-memory Qdrant vector search, and a Vite + React frontend to surface medical reference information, store structured patient history, and accept speech or prescription uploads. Use it as an internal prototype, not as medical advice.

## Features
- Symptom QA: Ask natural-language questions and receive LLM answers grounded on curated medical snippets.
- Medical knowledge reload: Seed or refresh the vector store with the bundled dataset in `backend/app/medical_data.py`.
- Patient history ingestion: Upload prescription images, extract text via OCR, and store structured entries in Qdrant.
- Audio intake: Record speech in the browser, transcribe it through Groq Whisper, and save transcripts as history.
- History insights: Query prior prescriptions and notes to surface pattern-aware insights for a patient ID.

## Repository Layout
```
Rag-lifeline/
├─ backend/
│  ├─ app/
│  │  ├─ main.py            # FastAPI app and endpoints
│  │  ├─ rag_chain.py       # RAG orchestration + OpenRouter calls
│  │  ├─ embeddings.py      # PubMedBERT embeddings
│  │  ├─ qdrant_store.py    # In-memory Qdrant helper utilities
│  │  ├─ medical_data.py    # Large medical reference corpus
│  │  ├─ models.py          # Pydantic schemas
│  │  ├─ audio_utils.py     # Groq Whisper transcription
│  │  └─ ocr_utils.py       # pytesseract helpers
│  ├─ tests/                # pytest suite (mocked model + LLM calls)
│  ├─ eval/                 # retrieval quality benchmark (recall@k, MRR)
│  ├─ requirements.txt
│  └─ run.py                # uvicorn wrapper
├─ frontend/
│  ├─ src/App.tsx           # React UI
│  ├─ src/api.ts            # Axios client
│  ├─ src/main.tsx
│  └─ index.html
├─ start-backend.bat        # Windows helper script
├─ start-frontend.bat       # Windows helper script
└─ README.md
```

## Requirements
- Python 3.10+
- Node.js 18+
- FFmpeg (for Whisper audio conversions)
- Tesseract OCR (only if you plan to upload prescriptions)
- OpenRouter API key (LLM responses)
- Optional Groq API key (audio transcription) and Qdrant Cloud credentials

## Environment Variables
Create `backend/.env` with the values you need:

| Variable | Required | Description |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | Yes | Grants access to OpenRouter chat completions. |
| `LLM_MODEL` | No | Overrides the default `google/gemini-2.5-flash`. |
| `GROQ_API_KEY` | For audio | Enables Whisper transcription in `/history/audio`. |
| `QDRANT_URL`, `QDRANT_API_KEY` | Optional | Point to Qdrant Cloud instead of in-memory mode. |
| `QDRANT_HOST`, `QDRANT_PORT` | Optional | Host/port for self-managed Qdrant. Ignored when using `:memory:`. |
| `HISTORY_RECENT_DAYS`, `HISTORY_TOP_K` | Optional | Tune patient history recall parameters. |

Example `.env`:
```
OPENROUTER_API_KEY=sk-or-...
LLM_MODEL=google/gemini-2.5-flash
GROQ_API_KEY=sk-groq-...
# Uncomment if you run a remote Qdrant cluster
# QDRANT_URL=https://YOUR-CLUSTER.aws.cloud.qdrant.io
# QDRANT_API_KEY=xxxxxxxx
```

## Backend Setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate    # macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt

# Ensure backend/.env exists before running the server
uvicorn app.main:app --reload --port 8000
# or: python run.py
```

First launch automatically loads the medical corpus into an in-memory Qdrant collection. If you later switch to Qdrant Cloud, set the host variables above and restart.

### Extra native dependencies
- **Tesseract OCR**: Install the engine and ensure `pytesseract` can find `tesseract.exe` on Windows PATH.
- **FFmpeg**: Required for browser-recorded audio uploads so Whisper can decode `.webm` blobs.

## Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

The Vite dev server runs on http://localhost:3000 and, by default, axios calls the backend directly at http://localhost:8000. Update `API_BASE_URL` in `frontend/src/api.ts` if your backend lives elsewhere.

## Quick Start on Windows
- `start-backend.bat`: Creates/activates `backend/venv`, installs pip dependencies, and launches uvicorn on port 8000.
- `start-frontend.bat`: Installs npm packages (if needed) and runs `npm run dev` on port 3000.

## API Reference
| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/` | Health + document count. |
| GET | `/health` | Same payload as `/`. |
| POST | `/query` | Main RAG question endpoint (`query`, optional `top_k`). |
| POST | `/reload-data` | Re-embeds the static dataset. |
| POST | `/history/prescription` | Form data upload of an image plus `patient_id`. Runs OCR and stores structured history. |
| POST | `/history/insight` | JSON body with `patient_id`, `symptoms`, optional `top_k` to summarize history. |
| POST | `/history/audio` | Form data upload of audio + `patient_id`; transcribes via Whisper and stores transcript. |

### Sample calls
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"I have headache and nausea","top_k":3}'

curl -X POST http://localhost:8000/history/insight \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"demo-patient","symptoms":"fatigue and thirst"}'
```

## Typical Workflows
1. **Symptom triage**: Use the main text area on the frontend (`/query` under the hood). Sources are shown so you can verify context.
2. **Upload a prescription**: Submit an image file via the UI. `ocr_utils.py` runs pytesseract, `rag_chain.py` structures the text, and `qdrant_store.py` writes to the `patient_history` collection.
3. **History insight**: Provide `patient_id` and current symptoms. The backend fetches similarity-matched and chronic entries, crafts a cautious summary, and responds with supporting history snippets.
4. **Audio intake**: Record speech in the browser; the frontend posts `.webm` audio to `/history/audio`, and the resulting transcript can be appended to your question.

## Customization
- **Add more medical knowledge**: Append new dict entries to `backend/app/medical_data.py`, each containing `content` and `metadata`. Run `/reload-data` or restart the backend to embed them.
- **Switch embedding models**: Update `_MODEL_NAME` in `backend/app/embeddings.py` to load a different Hugging Face checkpoint (make sure `VECTOR_SIZE` in `qdrant_store.py` matches the new model's dimension). Benchmark first with `eval/retrieval_eval.py` — see the Retrieval Evaluation section below for why the current model was chosen.
- **Try other LLMs**: Set `LLM_MODEL` in `.env` to any OpenRouter-supported model. Adjust `rag_chain.py` prompts if needed.
- **Persist Qdrant**: Point to a remote Qdrant cluster when you no longer want in-memory storage.

## Testing
```bash
cd backend
pip install -r requirements.txt
pytest
```
Tests mock the embedding model and OpenRouter calls, so they run in seconds without a GPU, network access, or an API key. They cover the FastAPI endpoints (including the graceful LLM-failure fallback path), the Qdrant helper functions, and a regression test for the corpus-flattening fix described below.

## Retrieval Evaluation
`eval/retrieval_eval.py` benchmarks retrieval quality: for a sample of medical Q&A pairs, it embeds the question and checks whether the matching answer is recovered out of same-domain distractors, reporting Recall@1/3/5 and MRR.
```bash
cd backend
python -m eval.retrieval_eval --n 300
```
Results are written to `backend/eval/results.json` and printed as a markdown table. See the script's docstring for the benchmark's limitations (it's a closed-world eval over the corpus's own Q&A pairs, not independently authored paraphrases).

### Results (n=300, seed=42)

| model | recall@1 | recall@3 | recall@5 | mrr | embed time (300 texts, CPU) |
| --- | --- | --- | --- | --- | --- |
| PubMedBERT (raw, masked-LM only) | 0.597 | 0.717 | 0.770 | 0.674 | 69.0s |
| **PubMedBERT-MSMARCO (used in this app)** | **0.847** | **0.923** | **0.943** | **0.889** | 36.0s |
| all-MiniLM-L6-v2 (general-purpose) | 0.937 | 0.973 | 0.983 | 0.958 | 5.3s |

**Model choice: `pritamdeka/S-PubMedBert-MS-MARCO`.** Three candidates were benchmarked:
- Raw PubMedBERT (the original choice) scored worst despite being domain-specific — it was pretrained only with masked-language-modeling on PubMed abstracts, never with a similarity/retrieval objective, so mean-pooling its token embeddings produces a mediocre sentence-embedding space.
- All-MiniLM-L6-v2, a general-purpose sentence-similarity model, actually scored *highest* on this benchmark and is ~7x faster — because it was explicitly trained for retrieval, and this corpus's phrasing (exam-style Q&A) is closer to general English than dense biomedical-literature text.
- PubMedBERT-MSMARCO — PubMedBERT's biomedical vocabulary with MS MARCO retrieval fine-tuning layered on top — lands between the two: a clear improvement over raw PubMedBERT (+0.25 recall@1) and only ~0.09 recall@1 behind MiniLM. We chose it over the higher-scoring MiniLM to keep medically-tuned representations for this domain, since the app's queries will include real symptom/diagnosis terminology not well represented in this exam-style benchmark. This is a deliberate quality/domain-fit tradeoff, not a claim that it's the highest-scoring option on this specific benchmark.

## Known Limitations & Roadmap
- **In-memory vector store**: Qdrant runs with `:memory:` by default, so both the medical knowledge base and all patient history are wiped on every backend restart. Point `QDRANT_URL`/`QDRANT_API_KEY` at a real Qdrant instance for anything beyond a local demo.
- **Corpus size is capped for dev speed**: the full medical corpus is ~19.7k Q&A pairs; embedding all of them on CPU at startup is slow, so `MAX_MEDICAL_DOCS` (default 1500) caps what gets loaded. Raise it (or remove the cap) for a full-corpus run — expect startup to take significantly longer without a GPU.
- **No authentication**: all endpoints, including patient history write endpoints, are open, and CORS allows any origin. `patient_id` is caller-supplied and unverified. Do not point this at real patient data as-is.
- **No automated retrieval quality tracking in CI**: `eval/retrieval_eval.py` is a manual benchmark, not wired into a pipeline; there's no regression alarm if a future change degrades retrieval quality.
- **Benchmark is a closed-world, exam-style eval**: the retrieval numbers above measure recovery of the corpus's own Q&A pairs, not independently authored paraphrased symptom queries — real user queries may perform differently than this benchmark suggests.

## Troubleshooting
- `Missing OPENROUTER_API_KEY`: RAG responses will fall back to static summaries; add the key to `.env`.
- `pytesseract not installed`: The `/history/prescription` endpoint returns HTTP 400. Install Tesseract and the Python binding.
- `Groq Whisper error`: Ensure `GROQ_API_KEY` is set and FFmpeg can be found on PATH.
- `LLM call fails`: The backend degrades gracefully by returning the top retrieved snippets, but check OpenRouter status and rate limits.

## Disclaimer
This project is for educational use only. It is **not** a medical device, and its responses are not diagnoses or treatment advice. Always consult a licensed clinician for medical decisions.
