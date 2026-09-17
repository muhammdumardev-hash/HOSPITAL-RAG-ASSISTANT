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

    .source-box {
        padding: 10px 14px;
        margin: 6px 0;
        border-left: 4px solid #14958a;
        border-radius: 8px;
        background: rgba(20, 149, 138, 0.08);
    }

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

    # --------------------------------------------------------
    # CATEGORY BREAKDOWN
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RETRIEVAL SETTINGS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------

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

    st.markdown("### 📚 Sources")

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

        raw_category = category_key(category)

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
                    f"{icon}  {question}",
                    key=f"example_{index}",
                    use_container_width=True,
                ):

                    st.session_state.pending_question = (
                        question
                    )

                    st.rerun()

    # --------------------------------------------------------
    # PREVIOUS CHAT HISTORY
    # --------------------------------------------------------

    for turn in st.session_state.chat_history:

        with st.chat_message(
            "user",
            avatar="🧑",
        ):

            st.write(
                turn["question"]
            )

            st.caption(
                turn["time"]
            )

        with st.chat_message(
            "assistant",
            avatar="🏥",
        ):

            render_confidence(
                turn["sources"]
            )

            st.write(
                turn["answer"]
            )

            render_sources(
                turn["sources"]
            )

            render_context(
                turn["sources"]
            )

    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    user_question = st.chat_input(
        "Ask about hospital policies, procedures, or services..."
    )

    # Example button question
    if st.session_state.pending_question:

        user_question = (
            st.session_state.pending_question
        )

        st.session_state.pending_question = None

    # --------------------------------------------------------
    # PROCESS QUESTION
    # --------------------------------------------------------

    if user_question:

        current_time = datetime.now().strftime(
            "%I:%M %p"
        )

        # User message

        with st.chat_message(
            "user",
            avatar="🧑",
        ):

            st.write(
                user_question
            )

            st.caption(
                current_time
            )

        # Assistant message

        with st.chat_message(
            "assistant",
            avatar="🏥",
        ):

            result = {
                "answer": "",
                "sources": [],
            }

            # ------------------------------------------------
            # GROQ CHECK
            # ------------------------------------------------

            if not groq_ready:

                result["answer"] = (
                    "Groq is not configured. "
                    "Please add GROQ_API_KEY and GROQ_MODEL "
                    "to Streamlit Secrets."
                )

                st.error(
                    result["answer"]
                )

            else:

                # --------------------------------------------
                # RETRIEVAL
                # --------------------------------------------

                with st.spinner(
                    "🔎 Searching the hospital knowledge base..."
                ):

                    try:

                        retrieved = retrieve_with_filters(
                            question=user_question,
                            category=selected_category,
                            document=selected_document,
                            top_k=top_k,
                            threshold=similarity_threshold,
                        )

                    except Exception as error:

                        retrieved = []

                        result["answer"] = (
                            "Something went wrong while "
                            f"searching the knowledge base: {error}"
                        )

                        st.error(
                            result["answer"]
                        )

                # --------------------------------------------
                # NO RESULTS
                # --------------------------------------------

                if not retrieved:

                    if not result["answer"]:

                        result["answer"] = (
                            "I could not find this information "
                            "in the selected hospital knowledge base."
                        )

                    st.info(
                        result["answer"]
                    )

                # --------------------------------------------
                # GENERATE GROUNDED ANSWER
                # --------------------------------------------

                else:

                    result["sources"] = retrieved

                    with st.spinner(
                        "🤖 Generating a grounded answer..."
                    ):

                        try:

                            result["answer"] = (
                                generate_grounded_answer(
                                    user_question,
                                    retrieved,
                                )
                            )

                        except Exception as error:

                            result["answer"] = (
                                "Something went wrong while "
                                f"generating the answer: {error}"
                            )

                            st.error(
                                result["answer"]
                            )

            # ------------------------------------------------
            # DISPLAY RESULT
            # ------------------------------------------------

            if result["sources"]:

                render_confidence(
                    result["sources"]
                )

            st.write(
                result["answer"]
            )

            render_sources(
                result["sources"]
            )

            render_context(
                result["sources"]
            )

        # ----------------------------------------------------
        # SAVE CHAT HISTORY
        # ----------------------------------------------------

        st.session_state.chat_history.append(
            {
                "question": user_question,
                "answer": result["answer"],
                "sources": result["sources"],
                "time": current_time,
            }
        )

        st.rerun()


# ============================================================
# EXPLORE KNOWLEDGE BASE TAB
# ============================================================

with tab_explore:

    st.subheader(
        "🗂️ Browse the Knowledge Base"
    )

    st.caption(
        "Read-only view of the indexed hospital documents."
    )

    for category in CATEGORIES:

        documents_in_category = (
            get_documents_for_category(
                category
            )
        )

        with st.expander(
            f"{cat_icon(category)} "
            f"{category} · "
            f"{len(documents_in_category)} document(s)"
        ):

            if not documents_in_category:

                st.info(
                    "No documents found."
                )

            else:

                for document in documents_in_category:

                    st.write(
                        f"📄 {document}"
                    )


# ============================================================
# ABOUT TAB
# ============================================================

with tab_about:

    st.subheader(
        "ℹ️ How this Assistant Works"
    )

    st.markdown(
        """
This assistant uses **Retrieval-Augmented Generation (RAG)**.

Hospital documents are processed during the ingestion stage,
converted into embeddings, and stored in a FAISS vector index.

### RAG Pipeline

**PDF → Text Extraction → Chunking → Embeddings → FAISS Search → Groq LLM → Answer**

### Grounding

- Answers are generated from retrieved hospital document chunks.
- The assistant is instructed not to invent hospital policies.
- Source filenames and page numbers come from stored metadata.
- If relevant information cannot be retrieved, the assistant reports that.
- The similarity threshold controls how strict semantic retrieval is.

### Knowledge Base Categories

- 📝 Admissions
- 💊 Department
- 🚨 Emergency
- 🏥 Hospital
- 🛡️ Patient Safety
        """
    )

    st.divider()

    st.write(
        f"**Embedding Model:** "
        f"`{EMBEDDING_MODEL_NAME.split('/')[-1]}`"
    )

    st.write(
        f"**Indexed Documents:** "
        f"`{total_documents}`"
    )

    st.write(
        f"**Indexed Chunks:** "
        f"`{total_chunks}`"
    )

    if groq_ready:

        st.write(
            f"**Groq Model:** "
            f"`{groq_model}`"
        )