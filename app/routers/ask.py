import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import config
from app.services.citations import build_citations
from app.services.embeddings import EmbeddingError
from app.services.generation import GenerationError, generate_answer, generate_answer_stream
from app.services.retrieval import retrieve

router = APIRouter(tags=["questions"])


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=config.MAX_QUESTION_CHARS)
    top_k: int = Field(config.TOP_K, ge=1, le=10, description="How many chunks to retrieve")


def _retrieve_or_raise(body: AskRequest) -> tuple[str, list[dict]]:
    """Shared by both endpoints: validate, retrieve, and raise proper HTTP errors."""
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="The question is empty.")

    try:
        chunks = retrieve(question, body.top_k)
    except EmbeddingError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No documents have been uploaded yet. Use POST /upload first.",
        )
    return question, chunks


def _compact(chunks: list[dict]) -> list[dict]:
    return [
        {
            "source": c["source"],
            "page": c["page"],
            "chunk_index": c["chunk_index"],
            "distance": c["distance"],
        }
        for c in chunks
    ]


@router.post("/ask")
def ask_question(body: AskRequest):
    """Answer a question using only the uploaded documents, with citations."""
    question, chunks = _retrieve_or_raise(body)

    try:
        answer = generate_answer(question, chunks)
    except GenerationError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))

    return {
        "question": question,
        "answer": answer,
        "citations": build_citations(answer, chunks),
        "retrieved_chunks": _compact(chunks),
    }


def sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event. json.dumps escapes newlines, so each event stays intact."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/ask/stream")
def ask_question_stream(body: AskRequest):
    """Same as /ask, but the answer streams in as it is written (Server-Sent Events)."""
    # Errors here still return normal HTTP error codes, because streaming hasn't started.
    question, chunks = _retrieve_or_raise(body)

    def event_stream():
        yield sse("retrieved", {"chunks": _compact(chunks)})

        parts: list[str] = []
        try:
            for piece in generate_answer_stream(question, chunks):
                parts.append(piece)
                yield sse("token", {"text": piece})
        except GenerationError as e:
            yield sse("error", {"detail": str(e), "status_code": e.status_code})
            return
        except Exception as e:  # anything unexpected: tell the client instead of dropping silently
            yield sse("error", {"detail": f"Unexpected error: {e}", "status_code": 500})
            return

        answer = "".join(parts).strip()
        yield sse("done", {"answer": answer, "citations": build_citations(answer, chunks)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )