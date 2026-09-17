"""
vector_store.py
Builds, saves, and loads the FAISS index, and saves/loads chunk metadata as a
separate JSON file. The mapping from FAISS vector ID -> chunk is simply "list
position": chunk metadata is written in the exact order embeddings were added
to the index, so FAISS search result index `i` always corresponds to
chunks_metadata[i].

Index type: IndexFlatIP (exact inner-product search).
- Embeddings are L2-normalized (see embeddings.py), so inner product between
  two vectors equals their cosine similarity.
- "Flat" means exact (brute-force) search, not an approximate index (like
  IVF or HNSW). With a hospital knowledge base of 20 PDFs — a few thousand
  chunks — exact search is fast enough and removes an entire class of
  approximate-search tuning problems. This keeps retrieval scores directly
  interpretable as cosine similarity, which the similarity-threshold logic
  (added in a later step) depends on.
"""

import json
import faiss
import numpy as np
from src.config import FAISS_INDEX_DIR, FAISS_INDEX_PATH, METADATA_DIR, METADATA_PATH


def build_index(embeddings: np.ndarray):
    """Create a fresh FAISS IndexFlatIP index from an (n, dim) embedding matrix."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index):
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))


def load_index():
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. Run `python ingest.py` first."
        )
    return faiss.read_index(str(FAISS_INDEX_PATH))


def save_metadata(chunks):
    """
    Save chunk metadata as a JSON list. List position == FAISS vector ID
    (see module docstring). Every field needed for source citations lives here.
    """
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "chunk_id": c.chunk_id,
            "category": c.category,
            "filename": c.filename,
            "document_title": c.document_title,
            "page": c.page,
            "source": c.source,
            "text": c.text,
        }
        for c in chunks
    ]
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found at {METADATA_PATH}. Run `python ingest.py` first."
        )
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
