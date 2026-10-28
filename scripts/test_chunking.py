from pathlib import Path

from app.services.chunking import split_text, chunk_pages
from app.services.extract import PageText

SIZE, OVERLAP = 300,60

sample = Path("samples/zephyr_policies.txt")
if not sample.exists():
    raise SystemExit("samples/zephyr_policies.txt not found. See step 2e.")

text = sample.read_text(encoding="utf-8")
pieces = split_text(text, SIZE, OVERLAP)

def shared_length(a:str, b:str) -> int:
    """Hpw many characters at the end of a are repeated at the start of b."""
    for k in range(min(len(a), len(b)), 0, -1):
        if a.endswith(b[:k]):
            return k
    return 0

print(f"Text length: {len(text)} chars -> {len(pieces)} chunks "
      f"(size={SIZE}, overlap={OVERLAP})\n")

for i, piece in enumerate(pieces):
    print(f"---chunk {i} ({len(piece)} chars) ---")
    print(piece)
    if i > 0:
        print(f"[overlap with previous chunk: {shared_length(pieces[i - 1], piece)} chars]")
    print()

assert all(len(p) <= SIZE for p in pieces), "a chunk is too long"
assert all(shared_length(pieces[i-1], pieces[i]) > 0 for i in range(1, len(pieces))), "no overlap found"
assert split_text("", SIZE, OVERLAP) == [], "empty text should have no chunks"
assert split_text("short text", SIZE, OVERLAP) == ["short text"], "short text should stay whole"

fake_pages = [PageText(page=1, text=text), PageText(page=5, text=text)]
chunks = chunk_pages(fake_pages, SIZE, OVERLAP)
assert [c.chunk_index for c in chunks] == list(range(len(chunks))), "chunk_index must be 0,1,2..."
assert {c.page for c in chunks} == {1,5}, "page number lost"

print("All chunking checks passed!")