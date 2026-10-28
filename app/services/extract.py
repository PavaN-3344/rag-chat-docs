from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
import re

@dataclass
class PageText:
    """The text of one page, plus its page number (starting at 1)."""
    page: int
    text: str

class ExtractionError(Exception):
    """Raised when a file can't be read."""    

def extract_txt(path: Path) -> list[PageText]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    text = text.strip()

    return [PageText(page=1, text=text)] if text else []

def extract_pdf(path: Path) -> list[PageText]:
    try:
        reader = PdfReader(path)
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise ExtractionError("This PDF is password-protected.")

        pages = []
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            # Some PDFs put a newline between every word. Collapse all runs of
            # whitespace (spaces, tabs, newlines) into a single space.
            text = re.sub(r"\s+", " ", text).strip()
            if text:  # skip blank pages, but keep real page numbers
                pages.append(PageText(page=number, text=text))
        return pages
    except ExtractionError:
        raise
    except Exception as e:
        raise ExtractionError(f"Could not read this PDF: {e}") from e

def extract_pages(path: Path) -> list[PageText]:
    """Pick the right extractor based on the file extension."""
    ext = path.suffix.lower()
    if ext == ".txt":
        return extract_txt(path)
    if ext == ".pdf":
        return extract_pdf(path)
    raise ExtractionError(f"Unsupported file type: {ext}")
                           