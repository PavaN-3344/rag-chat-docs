import re

# Matches [1], [2], [1, 2], and also [Excerpt 1] in case the model copies the old label.
CITATION_PATTERN = re.compile(
    r"\[(?:Excerpt\s*)?([1-9]\d*(?:\s*,\s*[1-9]\d*)*)\]", re.IGNORECASE
)


def extract_citation_numbers(answer: str) -> list[int]:
    """Return the excerpt numbers cited in the answer, in order of first appearance."""
    numbers: list[int] = []
    for match in CITATION_PATTERN.finditer(answer):
        for part in match.group(1).split(","):
            n = int(part.strip())
            if n >= 1 and n not in numbers:  # excerpts start at 1, so ignore [0]
                numbers.append(n)
    return numbers


def build_citations(answer: str, chunks: list[dict]) -> list[dict]:
    """Map the numbers in the answer to the real chunks we retrieved.

    Excerpt numbers start at 1 and match the order of `chunks`. Numbers that
    don't exist (the model made them up) are ignored.
    """
    citations = []
    for n in extract_citation_numbers(answer):
        if not 1 <= n <= len(chunks):
            continue
        c = chunks[n - 1]
        snippet = c["text"][:200].replace("\n", " ")
        citations.append(
            {
                "number": n,
                "source": c["source"],
                "page": c["page"],
                "chunk_index": c["chunk_index"],
                "label": f"{c['source']}, page {c['page']}, chunk {c['chunk_index']}",
                "distance": c["distance"],
                "snippet": snippet + ("..." if len(c["text"]) > 200 else ""),
            }
        )
    return citations