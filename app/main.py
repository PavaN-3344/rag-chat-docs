import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

from app import config
from app.routers import ask, upload

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag")

config.ensure_directories()

app = FastAPI(
    title="Chat With Your Documents",
    description="A local RAG app: upload documents, then ask questions about them.",
    version="1.0.0",
)

app.include_router(upload.router)
app.include_router(ask.router)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Turn FastAPI's long validation errors into one readable sentence."""
    problems = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err["loc"] if p != "body") or "request"
        problems.append(f"{field}: {err['msg']}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request. " + "; ".join(problems)},
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    """Catch anything we didn't plan for. Log the details, show the caller a safe message."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong on the server. Check the Terminal running uvicorn for details."},
    )


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["system"])
def health_check():
    """Quick check that the server is running and configured."""
    return {
        "status": "ok",
        "gemini_key_loaded": bool(config.GEMINI_API_KEY),
        "generation_model": config.GENERATION_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
    }