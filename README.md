# Chat With Your Documents (Local RAG)

Upload .txt/.pdf files, then ask questions answered only from your documents,
with citations (file, page, chunk) and streaming responses.

Stack: Python, FastAPI, ChromaDB, Gemini API (free tier).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your own Gemini API key
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs

## Note

Free-tier API content may be used by Google to improve its products.
Use non-sensitive documents.
