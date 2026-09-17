"""
pdf_loader.py
Recursively scans hospital_knowledge_base/, extracts text page-by-page from
every PDF, and reports on expected vs. found documents.

This module ONLY deals with reading PDFs into plain text + metadata. It does
NOT chunk or embed anything — that happens in text_splitter.py / embeddings.py.
"""

import fitz  # PyMuPDF
from src.config import BASE_DIR, KB_DIR, EXPECTED_PDFS


class PageRecord:
    """One page of extracted text plus everything needed to cite it later."""

    def __init__(self, category, filename, source, page_number, text):
        self.category = category
        self.filename = filename
        self.source = source          # relative path, e.g. hospital_knowledge_base/Emergency/emergency_triage.pdf
        self.page_number = page_number  # 1-indexed, human-readable
        self.text = text


def find_pdfs():
    """Recursively find all PDFs under KB_DIR, paired with their category (parent folder name)."""
    if not KB_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge base folder not found: {KB_DIR}\n"
            f"Make sure 'hospital_knowledge_base' exists in the project root "
            f"(same folder as ingest.py)."
        )

    pdfs = []
    for pdf_path in sorted(KB_DIR.rglob("*.pdf")):
        category = pdf_path.parent.name
        pdfs.append((category, pdf_path))
    return pdfs


def verify_expected_pdfs(found_pdfs):
    """
    Compare found PDFs against EXPECTED_PDFS.
    Returns (missing, extra) — both lists of "Category/filename.pdf" strings.
    """
    found_by_category = {}
    for category, path in found_pdfs:
        found_by_category.setdefault(category, set()).add(path.name)

    missing = []
    for category, expected_files in EXPECTED_PDFS.items():
        found_files = found_by_category.get(category, set())
        for f in expected_files:
            if f not in found_files:
                missing.append(f"{category}/{f}")

    extra = []
    for category, found_files in found_by_category.items():
        expected_files = set(EXPECTED_PDFS.get(category, []))
        for f in found_files:
            if f not in expected_files:
                extra.append(f"{category}/{f}")

    return missing, extra


def extract_pdf_pages(category, pdf_path):
    """
    Extract text page-by-page from a single PDF.
    Returns a list of PageRecord objects (possibly empty if the file is
    corrupted, has no pages, or has no extractable text — these are handled
    gracefully, never raised, so one bad PDF can't crash ingestion).
    """
    records = []

    try:
        source_str = str(pdf_path.relative_to(BASE_DIR)).replace("\\", "/")
    except ValueError:
        source_str = str(pdf_path)

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  [WARNING] Could not open {pdf_path.name}: {e}")
        return records

    if doc.page_count == 0:
        print(f"  [WARNING] {pdf_path.name} has no pages (possibly corrupted).")
        doc.close()
        return records

    for page_index in range(doc.page_count):
        try:
            page = doc.load_page(page_index)
            text = page.get_text("text").strip()
        except Exception as e:
            print(f"  [WARNING] Failed to read page {page_index + 1} of {pdf_path.name}: {e}")
            continue

        if not text:
            continue  # empty page — skip, don't fail the whole document

        records.append(
            PageRecord(
                category=category,
                filename=pdf_path.name,
                source=source_str,
                page_number=page_index + 1,
                text=text,
            )
        )

    doc.close()

    if not records:
        print(f"  [WARNING] No extractable text found in {pdf_path.name} "
              f"(it may be scanned/image-only, which this project does not OCR).")

    return records


def load_all_documents():
    """
    Full ingestion-time PDF loading step.
    Returns (all_page_records, missing, extra, stats).
    """
    pdfs = find_pdfs()
    missing, extra = verify_expected_pdfs(pdfs)

    print(f"Found {len(pdfs)} PDF documents.\n")

    all_records = []
    failed_documents = []

    for i, (category, pdf_path) in enumerate(pdfs, start=1):
        rel_display = f"{category}/{pdf_path.name}"
        print(f"[{i}/{len(pdfs)}] {rel_display}")
        records = extract_pdf_pages(category, pdf_path)
        if not records:
            failed_documents.append(rel_display)
        all_records.extend(records)

    stats = {
        "documents_found": len(pdfs),
        "documents_failed": len(failed_documents),
        "failed_documents": failed_documents,
        "pages_extracted": len(all_records),
        "categories": sorted(set(r.category for r in all_records)),
    }

    return all_records, missing, extra, stats
