import math

from app.services.embeddings import embed_documents, embed_query


def cosine(a,b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


question = "How do I get my money back for something I bought?"
related = "Customers may return unopened coffee bags within 45 days fo a full refund."
unrelated = "Beans are roasted in small batches every Tuesday and Thursday"

q = embed_query(question)
r, u = embed_documents([related, unrelated])

print("Vector length:", len(q))
print(f"Similarity to the refund text:   {cosine(q, r):.3f}")
print(f"Similarity to the roasting text:   {cosine(q, u):.3f}")

assert cosine(q, r) > cosine(q, u), "Related text should score higher!"
print("\nThe refund text is closer to the question, even though they share almost no words.")
