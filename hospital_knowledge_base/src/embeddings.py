"""
embeddings.py
Wraps the sentence-transformers model used both at ingestion time (embedding
every chunk) and at query time (embedding the user's question).

Why sentence-transformers/all-MiniLM-L6-v2:
- Small (~80MB) and fast on CPU. Streamlit Community Cloud has no GPU and
  limited RAM, so a lightweight model matters for both load time and
  per-query latency.
- 384-dimensional embeddings keep the FAISS index small even with thousands
  of chunks from 20 PDFs.
- It's trained for semantic similarity / sentence-pair tasks, which is
  exactly the retrieval job here (short-to-medium passages of ~700-1000
  characters), and it's a well-established default for RAG projects.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME

_model = None  # loaded once per process, reused after that


def get_embedding_model():
    """Load the embedding model once (this is the expensive step) and cache it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts, batch_size=32, normalize=True):
    """
    Embed a list of chunk texts (ingestion time).
    normalize=True L2-normalizes each embedding so that FAISS inner-product
    search is mathematically equivalent to cosine similarity — this is the
    similarity convention used throughout the project (see vector_store.py).
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return embeddings.astype(np.float32)


def embed_query(text, normalize=True):
    """Embed a single user query (retrieval time). Returns a (1, 384) float32 array."""
    model = get_embedding_model()
    embedding = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return embedding.astype(np.float32)
