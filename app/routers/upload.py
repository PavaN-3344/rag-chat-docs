from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app import config
from app.services import vector_store
from app.services.chunking import chunk_pages
from app.services.embeddings import EmbeddingError, embed_documents
from app.services.extract import ExtractionError, extract_pages

router = APIRouter(tags=["documents"])

@router.post("/upload")
def upload_document(file: UploadFile = File(...)):
    """Upload a .txt or .pdf file and extract its text."""
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    if not filename or extension not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Only these file types are allowed: {sorted(config.ALLOWED_EXTENSIONS)}",
        )

    content = file.file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(content) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File is larger then {config.MAX_UPLOAD_MB} MB.",
        )

    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    save_path = config.UPLOAD_DIR / filename
    save_path.write_bytes(content)

    try:
        pages = extract_pages(save_path)
    except ExtractionError as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e))

    if not pages:
        save_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail="No readable text found. If this is a scanned pdf, it needs OCR, which this project doesn't do. ",
        )    

    chunks = chunk_pages(pages)

    if len(chunks) > config.MAX_CHUNKS_PER_UPLOAD:
        save_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=(
                f"This file makes {len(chunks)} chunks, above the limit of "
                f"{config.MAX_CHUNKS_PER_UPLOAD}. Upload a smaller file or raise "
                f"MAX_CHUNKS_PER_UPLOAD in config.py (watch your free quota)."
            ),
        )

    try:
        vectors = embed_documents([c.text for c in chunks])
    except EmbeddingError as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=e.status_code, detail=str(e))

    vector_store.replace_document(filename, chunks, vectors)    

    return {
        "filename": filename,
        "file_type": extension.lstrip("."),
        "size_bytes": len(content),
        "pages_with_text": len(pages),
        "chunks_stored": len(chunks),
        "embedding_dimensions": len(vectors[0]),
        "total_chunks_in_database":vector_store.count(),
    }

@router.get("/documents")
def list_documents():
    """List the documents stored in the vector database"""
    return{"documents": vector_store.list_documents(), "total_chunks":vector_store.count()}

@router.delete("/documents/{filename}")
def delete_document(filename: str):
    """Remove a document's chunks from the database and delete the saved file."""
    filename = Path(filename).name
    removed = vector_store.delete_document(filename)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"No stored document named '{filename}'.")
    (config.UPLOAD_DIR / filename).unlink(missing_ok=True)
    return {"deleted": filename, "chunks_removed": removed}

@router.delete("/documents")
def delete_all_documents(confirm: bool = False):
    """Delete ALL stored documents and uploaded files. Requires ?confirm=true."""
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="This deletes everything. Add ?confirm=true to proceed.",
        )

    chunks_removed = vector_store.clear_all()

    files_removed = 0
    for path in config.UPLOAD_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in config.ALLOWED_EXTENSIONS:
            path.unlink()
            files_removed += 1

    return {"chunks_removed": chunks_removed, "files_removed": files_removed}    