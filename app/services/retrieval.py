from app.services import vector_store
from app.services.embeddings import embed_query

def retrieve(question:str, top_k: int) -> list[dict]:
    """Find the chunks closest in meaning to the question."""
    collection = vector_store.get_collection()
    total = collection.count()
    if total == 0:
        return []

    query_vector = embed_query(question)
    result = collection.query(
        query_embeddings = [query_vector],
        n_results = min(top_k, total),
        include = ["documents", "metadatas", "distances"],
    )

    chunks = []

    for text, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        chunks.append(
            {
                "text": text,
                "source": meta["source"],
                "page": meta["page"],
                "chunk_index": meta["chunk_index"],
                "distance": round(distance, 4),
            }
        )
    return chunks    