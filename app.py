"""
app.py
Hospital Knowledge Base Assistant — Streamlit front end.

Loads the pre-built FAISS index and metadata, retrieves relevant
hospital knowledge-base chunks, and generates grounded answers
through the Groq LLM.
"""

from datetime import datetime

import streamlit as st
from groq import Groq

from src.config import (
    CATEGORIES,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    EMBEDDING_MODEL_NAME,
    TOP_K_DEFAULT,
    SIMILARITY_THRESHOLD_DEFAULT,
)
from src.vector_store import load_metadata
from src.retriever import retrieve
from src.prompts import SYSTEM_PROMPT, build_user_prompt


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hospital Knowledge Base Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CATEGORY INFORMATION
# ============================================================

CATEGORY_META = {
    "Admissions": {
        "icon": "📝",
        "key": "admissions",
    },
    "Department": {
        "icon": "💊",
        "key": "departments",
    },
    "Emergency": {
        "icon": "🚨",
        "key": "emergency",
    },
    "Hospital": {
        "icon": "🏥",
        "key": "hospital",
    },
    "Patient Safety": {
        "icon": "🛡️",
        "key": "patient_safety",
    },
}


def cat_icon(category):
    return CATEGORY_META.get(
        category,
        {"icon": "📄"},
    )["icon"]


def category_key(category):

    if category == "All Categories":
        return None

    return CATEGORY_META.get(
        category,
        {"key": category},
    )["key"]


# ============================================================
# CUSTOM CSS
# ============================================================

st.html(
    """
    <style>

    /* ========================================================
       HOSPITAL HEADER
       ======================================================== */

    .hospital-header {
        padding: 28px;
        border-radius: 18px;
        margin-bottom: 24px;

        background: linear-gradient(
            120deg,
            #0b5c4f,
            #14958a,
            #22c3ab
        );

        color: white;

        box-shadow:
            0 10px 28px
            rgba(15, 110, 95, 0.25);
    }

    .hospital-header-title {
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hospital-header-text {
        font-size: 16px;
        opacity: 0.94;
        line-height: 1.6;
    }


    /* ========================================================
       SOURCE BOX
       ======================================================== */

    .source-box {
        padding: 10px 14px;
        margin: 6px 0;
        border-left: 4px solid #14958a;
        border-radius: 8px;
        background: rgba(20, 149, 138, 0.08);
    }


    /* ========================================================
       CONFIDENCE
       ======================================================== */

    .confidence-box {
        padding: 7px 12px;
        border-radius: 999px;
        display: inline-block;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .high-confidence {
        background: #e3f6ea;
        color: #1a7d3c;
    }

    .medium-confidence {
        background: #fff3d6;
        color: #9a6a00;
    }

    .low-confidence {
        background: #fde8e8;
        color: #b3261e;
    }


    /* ========================================================
       FIXED CHAT INPUT
       ======================================================== */

    /*
       Keep Streamlit chat input fixed at the bottom
       while the page is being scrolled.
    */

    [data-testid="stChatInput"] {
        position: fixed !important;

        bottom: 20px !important;

        /*
           Keep the input inside the main content area,
           instead of placing it over the sidebar.
        */
        left: calc(50% + 160px) !important;

        transform: translateX(-50%) !important;

        width: min(
            900px,
            calc(100vw - 380px)
        ) !important;

        z-index: 9999 !important;
    }


    /*
       Make the actual chat input use the full width.
    */

    [data-testid="stChatInput"] > div {
        width: 100% !important;
    }


    /*
       Add extra bottom space to the page.
       This prevents the last answer from being hidden
       behind the fixed input.
    */

    [data-testid="stAppViewContainer"] .main .block-container {
        padding-bottom: 140px !important;
    }


    /*
       Make sure the input stays above other content.
    */

    [data-testid="stChatInput"] textarea {
        position: relative !important;
        z-index: 10000 !important;
    }


    /* ========================================================
       RESPONSIVE DESIGN
       ======================================================== */

    /*
       On smaller screens there is usually no visible
       sidebar width to account for.
    */

    @media (max-width: 900px) {

        [data-testid="stChatInput"] {

            left: 50% !important;

            width: calc(100vw - 40px) !important;

            transform: translateX(-50%) !important;
        }
    }


    /*
       Extra adjustment for very small screens.
    */

    @media (max-width: 600px) {

        [data-testid="stChatInput"] {

            bottom: 10px !important;

            width: calc(100vw - 20px) !important;
        }

        [data-testid="stAppViewContainer"] .main .block-container {

            padding-bottom: 120px !important;
        }
    }

    </style>
    """
)


# ============================================================
# HEADER
# ============================================================

st.html(
    """
    <div class="hospital-header">

        <div class="hospital-header-title">
            🏥 Hospital Knowledge Base Assistant
        </div>

        <div class="hospital-header-text">
            Ask about hospital policies, procedures, departments,
            emergency care, admissions, and patient safety.
            Answers are grounded strictly in the hospital's
            own knowledge-base documents.
        </div>

    </div>
    """
)


# ============================================================
# STARTUP CHECKS
# ============================================================

if not FAISS_INDEX_PATH.exists():

    st.error(
        "FAISS index not found. "
        "Please run `python ingest.py` first."
    )

    st.stop()


if not METADATA_PATH.exists():

    st.error(
        "Metadata file not found. "
        "Please run `python ingest.py` first."
    )

    st.stop()


# ============================================================
# LOAD METADATA
# ============================================================

@st.cache_data
def get_metadata():

    return load_metadata()


try:

    metadata = get_metadata()

except Exception as error:

    st.error(
        f"Could not load knowledge-base metadata: {error}"
    )

    st.stop()


# ============================================================
# KNOWLEDGE BASE STATISTICS
# ============================================================

total_chunks = len(metadata)

documents = sorted(
    set(
        item["filename"]
        for item in metadata
    )
)

categories_found = sorted(
    set(
        item["category"]
        for item in metadata
    )
)

total_documents = len(documents)

total_categories = len(categories_found)


def get_documents_for_category(category):

    if category == "All Categories":

        return sorted(
            set(
                item["filename"]
                for item in metadata
            )
        )

    raw_category = category_key(category)

    return sorted(
        set(
            item["filename"]
            for item in metadata
            if item["category"] == raw_category
        )
    )


# ============================================================
# GROQ CONFIGURATION
# ============================================================

try:

    groq_api_key = st.secrets.get(
        "GROQ_API_KEY",
        ""
    )

    groq_model = st.secrets.get(
        "GROQ_MODEL",
        ""
    )

    groq_ready = bool(
        groq_api_key
        and groq_model
    )

except Exception:

    groq_api_key = ""

    groq_model = ""

    groq_ready = False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 Knowledge Base")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Docs",
            total_documents,
        )

    with col2:

        st.metric(
            "Chunks",
            total_chunks,
        )

    with col3:

        st.metric(
            "Categories",
            total_categories,
        )


    if groq_ready:

        st.success(
            "Groq connected",
            icon="✅",
        )

    else:

        st.warning(
            "Groq not configured",
            icon="⚠️",
        )


    st.divider()


    # ========================================================
    # CATEGORY BREAKDOWN
    # ========================================================

    with st.expander(
        "📊 Category breakdown",
        expanded=False,
    ):

        for category in CATEGORIES:

            docs_count = len(
                get_documents_for_category(
                    category
                )
            )

            st.write(
                f"{cat_icon(category)} "
                f"**{category}** — "
                f"{docs_count} document(s)"
            )


    st.divider()


    # ========================================================
    # FILTERS
    # ========================================================

    st.subheader("🔎 Filters")

    category_options = [
        "All Categories"
    ] + CATEGORIES

    selected_category = st.selectbox(
        "Category",
        category_options,
    )


    document_options = [
        "All Documents"
    ] + get_documents_for_category(
        selected_category
    )

    selected_document = st.selectbox(
        "Document",
        document_options,
    )


    # ========================================================
    # RETRIEVAL SETTINGS
    # ========================================================

    with st.expander(
        "⚙️ Advanced retrieval settings"
    ):

        top_k = st.slider(
            "Number of retrieved chunks",
            min_value=2,
            max_value=10,
            value=TOP_K_DEFAULT,
            step=1,
        )


        similarity_threshold = st.slider(
            "Similarity threshold",
            min_value=0.0,
            max_value=0.9,
            value=SIMILARITY_THRESHOLD_DEFAULT,
            step=0.05,
            help="Higher values require stronger semantic matches.",
        )


        st.caption(
            "Embedding model:"
        )

        st.code(
            EMBEDDING_MODEL_NAME.split("/")[-1],
            language=None,
        )


        if groq_ready:

            st.caption(
                "LLM:"
            )

            st.code(
                groq_model,
                language=None,
            )


    st.divider()


    # ========================================================
    # CLEAR CHAT
    # ========================================================

    if st.button(
        "🗑️ Clear chat history",
        use_container_width=True,
    ):

        st.session_state.chat_history = []

        st.toast(
            "Chat history cleared."
        )

        st.rerun()


    st.caption(
        "🔒 Answers are grounded only in the hospital knowledge base."
    )


# ============================================================
# SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


if "pending_question" not in st.session_state:

    st.session_state.pending_question = None


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

EXAMPLE_QUESTIONS = [

    (
        "📝",
        "How does the patient admission process work?",
    ),

    (
        "📝",
        "What documents are required for patient registration?",
    ),

    (
        "🚨",
        "What are the emergency triage procedures?",
    ),

    (
        "💊",
        "What are the hospital pharmacy guidelines?",
    ),

    (
        "🛡️",
        "What are the patient identification requirements?",
    ),

    (
        "🛡️",
        "What are the fall prevention guidelines?",
    ),

    (
        "🏥",
        "How can I contact the hospital?",
    ),

]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_confidence(chunks):

    if not chunks:

        return None

    top_score = max(
        float(
            chunk["similarity"]
        )
        for chunk in chunks
    )

    if top_score >= 0.60:

        return (
            "🟢 High confidence",
            top_score,
        )

    if top_score >= 0.40:

        return (
            "🟡 Medium confidence",
            top_score,
        )

    return (
        "🔴 Low confidence",
        top_score,
    )


def render_confidence(chunks):

    confidence = get_confidence(
        chunks
    )

    if confidence is None:

        return

    label, score = confidence

    st.caption(
        f"{label} · similarity {score:.2f}"
    )


def render_sources(sources):

    if not sources:

        return

    st.markdown(
        "### 📚 Sources"
    )

    for source in sources:

        category = source.get(
            "category",
            "Unknown",
        )

        filename = source.get(
            "filename",
            "Unknown document",
        )

        page = source.get(
            "page",
            "Unknown",
        )

        st.info(
            f"{cat_icon(category)} "
            f"**{filename}**  \n"
            f"Page {page} · {category}"
        )


def render_context(chunks):

    if not chunks:

        return

    with st.expander(
        f"🔍 View retrieved context ({len(chunks)} chunks)"
    ):

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):

            score = float(
                chunk["similarity"]
            )

            st.markdown(
                f"**{index}. "
                f"{cat_icon(chunk['category'])} "
                f"{chunk['filename']} "
                f"— Page {chunk['page']}**"
            )


            st.progress(
                min(
                    max(score, 0.0),
                    1.0,
                )
            )


            st.write(
                chunk["text"]
            )

            st.divider()


def generate_grounded_answer(
    question,
    retrieved_chunks,
):

    context = build_user_prompt(
        question,
        retrieved_chunks,
    )

    client = Groq(
        api_key=groq_api_key
    )

    response = client.chat.completions.create(
        model=groq_model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": context,
            },
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content


def retrieve_with_filters(
    question,
    category,
    document,
    top_k,
    threshold,
):
    """
    Retrieve more candidates first, then apply category/document
    filters. This avoids losing relevant results because a filtered
    document happened to fall outside the first small top-k results.
    """

    candidate_k = max(
        top_k * 5,
        25,
    )

    retrieved = retrieve(
        question,
        top_k=candidate_k,
        similarity_threshold=threshold,
    )


    if category != "All Categories":

        raw_category = category_key(
            category
        )

        retrieved = [
            item
            for item in retrieved
            if item["category"] == raw_category
        ]


    if document != "All Documents":

        retrieved = [
            item
            for item in retrieved
            if item["filename"] == document
        ]


    retrieved = sorted(
        retrieved,
        key=lambda item: float(
            item["similarity"]
        ),
        reverse=True,
    )


    return retrieved[:top_k]


# ============================================================
# TABS
# ============================================================

tab_chat, tab_explore, tab_about = st.tabs(
    [
        "💬 Chat",
        "🗂️ Explore Knowledge Base",
        "ℹ️ About",
    ]
)


# ============================================================
# CHAT TAB
# ============================================================

with tab_chat:

    # --------------------------------------------------------
    # EXAMPLE QUESTIONS
    # --------------------------------------------------------

    if not st.session_state.chat_history:

        st.caption(
            "Try an example question:"
        )

        columns = st.columns(2)

        for index, (
            icon,
            question,
        ) in enumerate(
            EXAMPLE_QUESTIONS
        ):

            with columns[index % 2]:

                if st.button(
                    f"{icon} {question}",
                    use_container_width=True,
                    key=f"example_{index}",
                ):

                    st.session_state.pending_question = question

                    st.rerun()


    # --------------------------------------------------------
    # DISPLAY CHAT HISTORY
    # --------------------------------------------------------

    for message in st.session_state.chat_history:

        role = message["role"]

        with st.chat_message(role):

            st.markdown(
                message["content"]
            )

            if role == "assistant":

                if message.get("sources"):

                    render_confidence(
                        message["sources"]
                    )

                    render_sources(
                        message["sources"]
                    )

                    render_context(
                        message["sources"]
                    )


    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask a question about the hospital knowledge base..."
    )


    if st.session_state.pending_question:

        question = st.session_state.pending_question

        st.session_state.pending_question = None


    # --------------------------------------------------------
    # PROCESS QUESTION
    # --------------------------------------------------------

    if question:

        if not groq_ready:

            st.error(
                "Groq is not configured. "
                "Please add GROQ_API_KEY and GROQ_MODEL "
                "to Streamlit secrets."
            )

            st.stop()


        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )


        with st.chat_message("user"):

            st.markdown(
                question
            )


        with st.chat_message("assistant"):

            with st.spinner(
                "Searching the hospital knowledge base..."
            ):

                retrieved_chunks = retrieve_with_filters(
                    question,
                    selected_category,
                    selected_document,
                    top_k,
                    similarity_threshold,
                )


            if not retrieved_chunks:

                answer = (
                    "I could not find this information in the "
                    "hospital knowledge base."
                )


            else:

                try:

                    answer = generate_grounded_answer(
                        question,
                        retrieved_chunks,
                    )

                except Exception as error:

                    st.error(
                        f"Could not generate an answer: {error}"
                    )

                    answer = (
                        "I could not generate the answer because "
                        "the language model request failed."
                    )


            st.markdown(
                answer
            )


            if retrieved_chunks:

                render_confidence(
                    retrieved_chunks
                )

                render_sources(
                    retrieved_chunks
                )

                render_context(
                    retrieved_chunks
                )


        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": retrieved_chunks,
            }
        )


# ============================================================
# EXPLORE KNOWLEDGE BASE TAB
# ============================================================

with tab_explore:

    st.subheader(
        "🗂️ Explore Knowledge Base"
    )


    st.write(
        "Browse the documents currently indexed in the hospital knowledge base."
    )


    for category in CATEGORIES:

        category_documents = get_documents_for_category(
            category
        )


        with st.expander(
            f"{cat_icon(category)} {category} "
            f"({len(category_documents)} documents)",
            expanded=False,
        ):

            if not category_documents:

                st.caption(
                    "No documents found in this category."
                )


            else:

                for filename in category_documents:

                    st.markdown(
                        f"📄 **{filename}**"
                    )


# ============================================================
# ABOUT TAB
# ============================================================

with tab_about:

    st.subheader(
        "ℹ️ About This Assistant"
    )


    st.markdown(
        """
        This application uses Retrieval-Augmented Generation (RAG) to answer
        questions from a hospital knowledge base.

        **Pipeline**

        Hospital PDFs → Text Extraction → Chunking → Embeddings → FAISS Search
        → Retrieved Context → LLM → Grounded Answer

        The assistant is designed to answer only from the indexed hospital
        documents. It does not intentionally use outside information to fill
        gaps in the knowledge base.

        **Technology**

        - Streamlit
        - PyMuPDF
        - Sentence Transformers
        - FAISS
        - Groq API
        - `openai/gpt-oss-120b`
        """
    )


    st.divider()


    st.caption(
        f"Knowledge base loaded at "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
