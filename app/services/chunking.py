from dataclasses import dataclass

from app import config
from app.services.extract import PageText

@dataclass
class Chunk:
    """One searchable piece of a document."""
    text: str
    page: int
    chunk_index: int

def split_text(text:str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into pieces of at most chunk_size charactters.

    Neighbouring pieces share about `overlap` characters. We try to cut at
    paragraph, line, sentence, or word boundaries instead of mid-word.
    """    

    if chunk_size <= 0:
        raise ValueError("Chunk size must be grater than 0")
    if overlap <0 or overlap >= chunk_size:
        raise ValueError("overlap must be 0 or more, and smaller than chunk_size")

    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    pieces: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            window_start = start + int(chunk_size * 0.6)
            for seperator in ("\n\n", "\n", ". ", " "):
                found = text.rfind(seperator, window_start, end)
                if found != -1:
                    end = found + len(seperator)
                    break

        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)

        if end >= len(text): 
            break                
    
        next_start = end - overlap

        if next_start > start:
            space = text.find(" ", next_start, end)
            if space != -1:
                next_start = space + 1

        if next_start <= start:
            next_start = end
        start = next_start

    return pieces

def chunk_pages(
        pages: list[PageText],
        chunk_size: int = config.CHUNK_SIZE,
        overlap: int = config.CHUNK_OVERLAP
) -> list[Chunk]:
    """Chunk every page, keeping page numbers and numbering chunks 0, 1, 2..."""
    chunks: list[Chunk] = []
    for page in pages:
        for piece in split_text(page.text, chunk_size, overlap):
            chunks.append(Chunk(text=piece, page=page.page, chunk_index=len(chunks)))
    return chunks        