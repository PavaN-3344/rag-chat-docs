from app.services.citations import build_citations, extract_citation_numbers

chunks = [
    {"text": "Returns within 45 days.", "source": "a.txt", "page": 1, "chunk_index": 0, "distance": 0.2},
    {"text": "Express shipping is $12.", "source": "b.pdf", "page": 3, "chunk_index": 7, "distance": 0.3},
]

assert extract_citation_numbers("Yes [1]. Also [2][1].") == [1, 2]
assert extract_citation_numbers("Both apply [1, 2].") == [1, 2]
assert extract_citation_numbers("Per [Excerpt 2].") == [2]
assert extract_citation_numbers("No markers here.") == []
assert extract_citation_numbers("Array a[0] is unrelated.") == []

cites = build_citations("Returns take 45 days [1] and shipping costs $12 [2].", chunks)
assert [c["label"] for c in cites] == ["a.txt, page 1, chunk 0", "b.pdf, page 3, chunk 7"]

# A number the model invented (9) is ignored.
assert [c["number"] for c in build_citations("Made up [9].", chunks)] == []

# A refusal has no citations.
assert build_citations("I couldn't find that in the uploaded documents.", chunks) == []

print("All citation checks passed!")