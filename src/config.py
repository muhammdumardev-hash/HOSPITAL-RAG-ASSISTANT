"""
config.py
Single source of truth for paths and constants used by ingest.py, the
retriever, and the Streamlit app. Everything here is a RELATIVE path built
from the project root, so the project works identically on Windows locally
and on Streamlit Community Cloud (which checks out the repo fresh each time).
"""

from pathlib import Path

# Project root = the folder that contains this "src" folder's parent.
BASE_DIR = Path(__file__).resolve().parent.parent

KB_DIR = BASE_DIR / "hospital_knowledge_base"

DATA_DIR = BASE_DIR / "data"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
FAISS_INDEX_PATH = FAISS_INDEX_DIR / "index.faiss"
METADATA_DIR = DATA_DIR / "metadata"
METADATA_PATH = METADATA_DIR / "chunks.json"

# The exact 5 categories — these are the folder names under hospital_knowledge_base/
CATEGORIES = ["Admissions", "Department", "Emergency", "Hospital", "Patient Safety"]

# The exact 20 expected PDFs, used by ingest.py to verify nothing is missing.
EXPECTED_PDFS = {
    "Admissions": [
        "patient_admission_guidelines.pdf",
        "patient_registration.pdf",
    ],
    "Department": [
        "billing_information.pdf",
        "nursing_guidelines.pdf",
        "pharmacy_guidelines.pdf",
    ],
    "Emergency": [
        "emergency_care.pdf",
        "emergency_procedures.pdf",
        "emergency_triage.pdf",
    ],
    "Hospital": [
        "hospital_information.pdf",
        "hospital_policies.pdf",
        "hospital_services.pdf",
        "hospital_departments.pdf",
        "hospital_facilities.pdf",
        "hospital_contact_information.pdf",
    ],
    "Patient Safety": [
        "patient_safety_guidelines.pdf",
        "infection_control.pdf",
        "medication_safety.pdf",
        "fall_prevention.pdf",
        "patient_identification.pdf",
        "patient_rights_and_safety.pdf",
    ],
}

TOTAL_EXPECTED_PDFS = sum(len(v) for v in EXPECTED_PDFS.values())  # 20

# Embedding model — small, fast on CPU, good for short/medium passages.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking targets (characters, not tokens — simple and predictable).
CHUNK_SIZE = 900       # within the 700-1000 target range
CHUNK_OVERLAP = 130    # within the 100-150 target range

# Retrieval defaults (used later by the retriever / Streamlit UI).
TOP_K_DEFAULT = 5
SIMILARITY_THRESHOLD_DEFAULT = 0.35  # cosine similarity, since we normalize embeddings
