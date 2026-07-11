"""
Multi-Document Research Assistant — Streamlit entry point.

Run with:
    streamlit run app.py
"""

import sys
import os
from pathlib import Path

# Ensure src/ is importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from src.config import settings
from src.logger import get_logger
from src.loaders import DocumentLoader
from src.chunker import TextChunker
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.llm import LLMClient
from src.rag_chain import RAGChain
from src.utils import format_sources, save_uploaded_file, clear_directory

logger = get_logger(__name__)

# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multi-Document Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ---- Global resets ---- */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: #0f1117;
        border-right: 1px solid #1e2130;
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* ---- Chat bubbles ---- */
    .chat-bubble {
        padding: 0.85rem 1.1rem;
        border-radius: 12px;
        margin-bottom: 0.6rem;
        max-width: 85%;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    .chat-bubble.user {
        background: #1e40af;
        color: #e0e7ff;
        margin-left: auto;
    }
    .chat-bubble.assistant {
        background: #1e2130;
        color: #e2e8f0;
        border: 1px solid #2d3450;
    }

    /* ---- Source pills ---- */
    .source-pill {
        display: inline-block;
        background: #0f2d5a;
        color: #93c5fd;
        border: 1px solid #1e4080;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        margin: 2px 3px;
    }

    /* ---- Upload area ---- */
    [data-testid="stFileUploader"] {
        border: 2px dashed #2d3450;
        border-radius: 10px;
        padding: 0.5rem;
    }

    /* ---- Status badge ---- */
    .status-ready  { color: #4ade80; font-weight: 600; }
    .status-empty  { color: #f59e0b; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session-state initialisation ─────────────────────────────────────────────


def init_session_state() -> None:
    """Initialise all Streamlit session-state keys on first load."""
    defaults: dict = {
        "messages": [],          # [{"role": str, "content": str, "sources": list}]
        "rag_chain": None,       # RAGChain instance (built after indexing)
        "indexed_files": [],     # Names of files already indexed
        "vector_store": None,    # Persistent VectorStore handle
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# ── Cached resource builders ──────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def build_embedding_model() -> EmbeddingModel:
    """Load the sentence-transformer embedding model once per session."""
    logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
    return EmbeddingModel(model_name=settings.EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def build_vector_store(_embedding_model: EmbeddingModel) -> VectorStore:
    """Initialise (or reopen) the ChromaDB vector store."""
    logger.info("Opening ChromaDB at %s", settings.CHROMA_DB_PATH)
    return VectorStore(
        embedding_model=_embedding_model,
        persist_directory=str(settings.CHROMA_DB_PATH),
        collection_name=settings.CHROMA_COLLECTION_NAME,
    )


def build_rag_chain(vector_store: VectorStore) -> RAGChain:
    """Assemble the full RAG pipeline (retriever + LLM)."""
    retriever = Retriever(
        vector_store=vector_store,
        k=settings.RETRIEVER_K,
    )
    llm_client = LLMClient(
    model_name=settings.LLM_MODEL,
    temperature=settings.LLM_TEMPERATURE,
    max_tokens=settings.LLM_MAX_TOKENS,
    api_key=settings.GROQ_API_KEY,
    )
    return RAGChain(
        retriever=retriever,
        llm_client=llm_client,
        max_history_turns=settings.MAX_HISTORY_TURNS,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────


def render_sidebar() -> None:
    """Render the document-upload and knowledge-base management sidebar."""
    with st.sidebar:
        st.markdown("## 📂 Document Library")

        # ---- Status badge ----
        if st.session_state.indexed_files:
            count = len(st.session_state.indexed_files)
            st.markdown(
                f"<span class='status-ready'>● {count} file(s) indexed</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span class='status-empty'>● No documents indexed</span>",
                unsafe_allow_html=True,
            )

        st.divider()

        # ---- File uploader ----
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            help="Supports PDF, DOCX, TXT, and Markdown files.",
        )

        if uploaded_files:
            new_files = [
                f for f in uploaded_files
                if f.name not in st.session_state.indexed_files
            ]
            if new_files:
                if st.button("⚡ Index Documents", use_container_width=True, type="primary"):
                    _index_files(new_files)
            else:
                st.info("All uploaded files are already indexed.")

        st.divider()

        # ---- Already indexed list ----
        if st.session_state.indexed_files:
            st.markdown("**Indexed files**")
            for fname in st.session_state.indexed_files:
                st.markdown(f"• `{fname}`")

        st.divider()

        # ---- Controls ----
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear KB", use_container_width=True):
                _clear_knowledge_base()
        with col2:
            if st.button("💬 New Chat", use_container_width=True):
                st.session_state.messages = []
                if st.session_state.rag_chain:
                    st.session_state.rag_chain.clear_history()
                st.rerun()

        st.divider()
        st.markdown(
            "<small style='color:#64748b'>Model: "
            f"<b>{settings.LLM_MODEL}</b><br>"
            f"Embeddings: <b>{settings.EMBEDDING_MODEL}</b><br>"
            f"Chunk size: <b>{settings.CHUNK_SIZE}</b> | "
            f"Overlap: <b>{settings.CHUNK_OVERLAP}</b></small>",
            unsafe_allow_html=True,
        )


def _index_files(files: list) -> None:
    """Load, chunk, embed, and store the supplied uploaded files."""
    embedding_model = build_embedding_model()
    vector_store = build_vector_store(embedding_model)
    loader = DocumentLoader()
    chunker = TextChunker(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    progress_bar = st.sidebar.progress(0, text="Starting indexing…")
    total = len(files)

    for idx, uploaded_file in enumerate(files, start=1):
        try:
            progress_bar.progress(
                (idx - 1) / total,
                text=f"Processing {uploaded_file.name} ({idx}/{total})…",
            )
            # Persist the upload to disk so loaders can open it
            save_path = save_uploaded_file(
                uploaded_file,
                dest_dir=str(settings.DATA_PATH),
            )
            # Load → chunk → index
            docs = loader.load(save_path)
            chunks = chunker.split(docs)
            vector_store.add_documents(chunks)
            st.session_state.indexed_files.append(uploaded_file.name)
            logger.info("Indexed %s (%d chunks)", uploaded_file.name, len(chunks))
        except Exception as exc:
            logger.error("Failed to index %s: %s", uploaded_file.name, exc, exc_info=True)
            st.sidebar.error(f"Error indexing **{uploaded_file.name}**: {exc}")

    progress_bar.progress(1.0, text="Indexing complete!")

    # Rebuild the RAG chain with the updated vector store
    st.session_state.vector_store = vector_store
    st.session_state.rag_chain = build_rag_chain(vector_store)
    st.rerun()


def _clear_knowledge_base() -> None:
    """Drop the ChromaDB collection and wipe the data directory."""
    try:
        if st.session_state.vector_store:
            st.session_state.vector_store.clear()
        clear_directory(str(settings.DATA_PATH))
        st.session_state.indexed_files = []
        st.session_state.rag_chain = None
        st.session_state.vector_store = None
        st.session_state.messages = []
        # Bust the cached vector-store so a fresh one is created next time
        build_vector_store.clear()
        st.rerun()
        logger.info("Knowledge base cleared by user.")
    except Exception as exc:
        logger.error("Error clearing knowledge base: %s", exc, exc_info=True)
        st.sidebar.error(f"Could not clear knowledge base: {exc}")


# ── Main chat interface ───────────────────────────────────────────────────────


def render_chat() -> None:
    """Render the main chat column."""
    st.markdown("# 🔍 Research Assistant")
    st.markdown(
        "<small style='color:#64748b'>Upload documents in the sidebar, then ask questions below.</small>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ---- Message history ----
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            _render_message(msg)

    # ---- Input ----
    user_input = st.chat_input(
        "Ask a question about your documents…",
        disabled=(st.session_state.rag_chain is None),
    )

    if user_input:
        _handle_user_message(user_input.strip())

    if st.session_state.rag_chain is None:
        st.info(
            "👈 Upload one or more documents and click **Index Documents** to get started.",
            icon="📄",
        )


def _render_message(msg: dict) -> None:
    """Render a single chat message with optional source pills."""
    role = msg["role"]
    content = msg["content"]
    sources = msg.get("sources", [])

    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant"):
            st.markdown(content)
            if sources:
                st.markdown("**Sources:**")
                pills_html = "".join(
                    f"<span class='source-pill'>{s}</span>" for s in sources
                )
                st.markdown(
                    f"<div style='margin-top:4px'>{pills_html}</div>",
                    unsafe_allow_html=True,
                )


def _handle_user_message(question: str) -> None:
    """Send a question through the RAG chain and stream the response."""
    # Append user message
    st.session_state.messages.append({"role": "user", "content": question})

    rag_chain: RAGChain = st.session_state.rag_chain

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Thinking…_")
        try:
            result = rag_chain.query(question)
            answer: str = result["answer"]
            sources: list[str] = format_sources(result.get("source_documents", []))

            placeholder.markdown(answer)
            if sources:
                st.markdown("**Sources:**")
                pills_html = "".join(
                    f"<span class='source-pill'>{s}</span>" for s in sources
                )
                st.markdown(
                    f"<div style='margin-top:4px'>{pills_html}</div>",
                    unsafe_allow_html=True,
                )

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
            logger.info("Q: %s | sources: %s", question[:80], sources)
        except Exception as exc:
            error_msg = f"⚠️ An error occurred: {exc}"
            placeholder.error(error_msg)
            logger.error("RAG query failed: %s", exc, exc_info=True)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg, "sources": []}
            )


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Application entry point."""
    # Lazily rebuild the RAG chain if files were already indexed in a previous
    # Streamlit run (e.g. after a hot-reload) and the chain is missing.
    if st.session_state.indexed_files and st.session_state.rag_chain is None:
        try:
            embedding_model = build_embedding_model()
            vector_store = build_vector_store(embedding_model)
            st.session_state.vector_store = vector_store
            st.session_state.rag_chain = build_rag_chain(vector_store)
        except Exception as exc:
            logger.warning("Could not restore RAG chain: %s", exc)

    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()