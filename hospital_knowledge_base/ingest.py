"""
ingest.py
Standalone ingestion script. Run this once locally (and again any time the
PDFs in hospital_knowledge_base/ change) to build the FAISS index and
metadata that app.py will later load. This is intentionally separate from
the Streamlit app — the app must never rebuild these at startup.

Usage (from the project root, with your venv activated):
    python ingest.py
"""

import sys
import time

from src.pdf_loader import load_all_documents
from src.text_splitter import chunk_page_records
from src.embeddings import embed_texts
from src.vector_store import build_index, save_index, save_metadata
from src.config import FAISS_INDEX_PATH, METADATA_PATH, TOTAL_EXPECTED_PDFS


def main():
    start_time = time.time()

    print("=" * 60)
    print("Loading hospital knowledge base...")
    print("=" * 60)

    try:
        page_records, missing, extra, load_stats = load_all_documents()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    if load_stats["documents_found"] != TOTAL_EXPECTED_PDFS:
        print(f"\n[WARNING] Expected {TOTAL_EXPECTED_PDFS} PDFs, found "
              f"{load_stats['documents_found']}.")

    if missing:
        print("\n[WARNING] The following expected PDFs were NOT found:")
        for m in missing:
            print(f"   - {m}")

    if extra:
        print("\n[INFO] The following PDFs were found but are not in the expected list:")
        for e_ in extra:
            print(f"   - {e_}")

    if load_stats["documents_failed"] > 0:
        print("\n[WARNING] The following documents produced no extractable text:")
        for f in load_stats["failed_documents"]:
            print(f"   - {f}")

    if not page_records:
        print("\n[ERROR] No text could be extracted from any document. Aborting.")
        sys.exit(1)

    print(f"\nExtracted text from {load_stats['pages_extracted']} pages "
          f"across {load_stats['documents_found']} documents.")

    print("\nCreating chunks...")
    chunks = chunk_page_records(page_records)
    print(f"Created {len(chunks)} chunks.")

    print("\nGenerating embeddings...")
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    print(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}.")

    print("\nBuilding FAISS index...")
    index = build_index(embeddings)
    save_index(index)

    print("Saving metadata...")
    save_metadata(chunks)

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("Ingestion completed successfully.")
    print("=" * 60)
    print(f"Documents processed : {load_stats['documents_found']} "
          f"(expected {TOTAL_EXPECTED_PDFS})")
    print(f"Documents failed     : {load_stats['documents_failed']}")
    print(f"Pages processed      : {load_stats['pages_extracted']}")
    print(f"Chunks created       : {len(chunks)}")
    print(f"Categories found     : {', '.join(load_stats['categories'])}")
    print(f"FAISS index          : {FAISS_INDEX_PATH}")
    print(f"Metadata             : {METADATA_PATH}")
    print(f"Time elapsed         : {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
