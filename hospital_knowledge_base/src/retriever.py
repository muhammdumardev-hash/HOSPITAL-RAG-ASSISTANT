from src.embeddings import embed_query
from src.vector_store import load_index, load_metadata
from src.config import TOP_K_DEFAULT, SIMILARITY_THRESHOLD_DEFAULT


def retrieve(
    query,
    top_k=TOP_K_DEFAULT,
    similarity_threshold=SIMILARITY_THRESHOLD_DEFAULT,
):
    index = load_index()
    metadata = load_metadata()

    query_embedding = embed_query(query)

    # Search the complete FAISS index.
    # This is important because app.py may apply
    # category/document filters after retrieval.
    search_k = min(index.ntotal, max(top_k * 10, 100))

    scores, indices = index.search(
        query_embedding,
        search_k,
    )

    results = []

    for score, index_id in zip(scores[0], indices[0]):

        if index_id < 0:
            continue

        score = float(score)

        if score < similarity_threshold:
            continue

        chunk = metadata[int(index_id)].copy()

        chunk["similarity"] = score

        results.append(chunk)

    return results