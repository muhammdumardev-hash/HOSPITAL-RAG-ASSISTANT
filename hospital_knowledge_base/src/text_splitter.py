"""
text_splitter.py
Page-aware chunking: splits each page's text into overlapping chunks while
preserving the exact page number as metadata.

Design note: chunking happens PER PAGE, not across the whole document. This
means a chunk can never "forget" which page it came from — there is no
multi-page merging logic that could invent or blur a page number. This is a
deliberate trade-off: a rare sentence that crosses a page boundary might end
up split between two chunks, but every chunk's page citation is always
100% accurate, which the project explicitly requires.
"""

from dataclasses import dataclass
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    chunk_id: int
    category: str
    filename: str
    document_title: str
    page: int
    source: str
    text: str


def _split_text(text, chunk_size, overlap):
    """Sliding-window character splitter with overlap, biased to break on sentence/line ends."""
    length = len(text)
    if length <= chunk_size:
        return [text]

    pieces = []
    start = 0
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            # try to break at a sentence or line boundary rather than mid-word
            candidate = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if candidate > start + int(chunk_size * 0.5):
                end = candidate + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= length:
            break
        start = max(end - overlap, start + 1)  # always move forward

    return pieces


def make_title_from_filename(filename):
    """e.g. 'emergency_triage.pdf' -> 'Emergency Triage'"""
    name = filename[:-4] if filename.lower().endswith(".pdf") else filename
    return name.replace("_", " ").replace("-", " ").title()


def chunk_page_records(page_records, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Turn PageRecord objects (from pdf_loader) into Chunk objects.
    chunk_id is assigned sequentially and will later match the FAISS vector ID,
    since chunks are embedded and added to the index in this same order.
    """
    chunks = []
    chunk_id = 0

    for record in page_records:
        title = make_title_from_filename(record.filename)
        for piece in _split_text(record.text, chunk_size, overlap):
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    category=record.category,
                    filename=record.filename,
                    document_title=title,
                    page=record.page_number,
                    source=record.source,
                    text=piece,
                )
            )
            chunk_id += 1

    return chunks
