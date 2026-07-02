"""
app.py — Streamlit frontend for the Smart PDF Study Assistant.

Run with:
    streamlit run app.py

This single file wires together all the backend modules and renders the UI.
The design is intentionally simple so a beginner can read it top-to-bottom
and understand the full data flow.

Data flow on each user question:
    User types question
        → RefinementLoop.run(question, history)
            → Retriever.retrieve()          [Context Engineering]
            → PromptBuilder.build()         [Context Engineering]
            → LLMClient.complete()          [Harness Engineering]
            → _is_answer_vague() → loop?    [Loop Engineering]
        → display iteration log, chunks, final answer
"""

import os
import sys

import streamlit as st

# Make sure the project root is on the Python path so `from backend.X` works
# regardless of how streamlit is launched.
sys.path.insert(0, os.path.dirname(__file__))

from backend.config import OPENAI_API_KEY
from backend.embeddings import EmbeddingManager
from backend.llm import LLMClient
from backend.loop import RefinementLoop
from backend.pdf_loader import PDFLoader
from backend.prompt_builder import PromptBuilder
from backend.retriever import Retriever
from backend.utils import setup_logger

logger = setup_logger(__name__)

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart PDF Study Assistant",
    layout="wide",
)

# ── Session state initialisation ───────────────────────────────────────────────
# Streamlit reruns the whole script on every interaction.
# st.session_state persists values across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []          # conversation history

if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False     # has a PDF been processed?

if "pdf_filename" not in st.session_state:
    st.session_state.pdf_filename = ""

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# ── Helper: build backend objects ─────────────────────────────────────────────
# We cache these with @st.cache_resource so they are created once per session
# (not re-created on every Streamlit rerun).

@st.cache_resource
def get_embedding_manager() -> EmbeddingManager:
    """Create (and cache) the EmbeddingManager."""
    return EmbeddingManager(api_key=OPENAI_API_KEY)


@st.cache_resource
def get_llm_client() -> LLMClient:
    """Create (and cache) the LLMClient."""
    return LLMClient(api_key=OPENAI_API_KEY)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Study Assistant")
    st.markdown("---")

    # PDF Upload
    st.subheader("Document Upload")
    uploaded_file = st.file_uploader(
        "Select a PDF file",
        type=["pdf"],
        help="Upload lecture slides or notes. The application will extract and index the text.",
    )

    if uploaded_file is not None:
        if st.button("Process PDF", use_container_width=True):
            with st.spinner(f"Processing '{uploaded_file.name}'..."):
                try:
                    # Step 1: Extract text and create chunks
                    loader = PDFLoader()
                    chunks = loader.load_and_chunk(
                        file_bytes=uploaded_file.getvalue(),
                        filename=uploaded_file.name,
                    )

                    # Step 2: Embed chunks and store in ChromaDB
                    manager = get_embedding_manager()
                    manager.add_chunks(chunks)

                    # Update session state
                    st.session_state.pdf_loaded = True
                    st.session_state.pdf_filename = uploaded_file.name
                    st.session_state.chunk_count = manager.get_chunk_count()
                    st.session_state.messages = []  # fresh conversation per PDF

                    st.success(
                        f"Processed '{uploaded_file.name}'\n\n"
                        f"Created **{len(chunks)}** chunks  \n"
                        f"Total in database: **{st.session_state.chunk_count}**"
                    )
                    logger.info(
                        "PDF '%s' processed: %d chunks.", uploaded_file.name, len(chunks)
                    )

                except Exception as exc:
                    st.error(f"Failed to process PDF: {exc}")
                    logger.error("PDF processing error: %s", exc)

    st.markdown("---")

    # Status
    if st.session_state.pdf_loaded:
        st.success(f"Active document: **{st.session_state.pdf_filename}**")
        st.caption(f"{st.session_state.chunk_count} chunks in database")
    else:
        st.info("No document loaded yet.")

    st.markdown("---")

    # Clear Database
    st.subheader("Database")
    if st.button("Clear Database", use_container_width=True, type="secondary"):
        try:
            manager = get_embedding_manager()
            manager.clear_collection()
            st.session_state.pdf_loaded = False
            st.session_state.pdf_filename = ""
            st.session_state.chunk_count = 0
            st.session_state.messages = []
            # Clear the cached resource so a fresh one is built next time
            get_embedding_manager.clear()
            get_llm_client.clear()
            st.success("Database cleared.")
            logger.info("ChromaDB collection cleared by user.")
        except Exception as exc:
            st.error(f"Could not clear database: {exc}")

    st.markdown("---")
    st.caption(
        "Study Assistant  \n"
        "Context, Harness, and Loop Engineering"
    )


# ── Main page ──────────────────────────────────────────────────────────────────
st.title("Study Assistant")
st.markdown(
    "Upload lecture notes in the sidebar, then submit questions below. "
    "Responses are generated from the indexed document only."
)

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**Context Engineering**\nRetrieves only relevant document chunks for each query.")
with col2:
    st.info("**Harness Engineering**\nOrchestrates PDF ingestion, embeddings, retrieval, and generation.")
with col3:
    st.info("**Loop Engineering**\nRefines answers when the initial response is insufficient.")

st.markdown("---")

# ── Conversation history display ───────────────────────────────────────────────
for message in st.session_state.messages:
    role = message["role"]
    with st.chat_message(role):
        st.markdown(message["content"])

# ── Chat input ─────────────────────────────────────────────────────────────────
user_question = st.chat_input(
    "Enter a question about your lecture notes",
    disabled=not st.session_state.pdf_loaded,
)

if user_question:
    if not st.session_state.pdf_loaded:
        st.warning("Please upload and process a PDF before asking questions.")
        st.stop()

    # Display the user's message immediately
    with st.chat_message("user"):
        st.markdown(user_question)

    # Add to conversation history (for context engineering / history trimming)
    st.session_state.messages.append({"role": "user", "content": user_question})

    # ── Run the refinement loop ────────────────────────────────────────────────
    with st.chat_message("assistant"):
        # Show a spinner while the loop runs
        with st.spinner("Generating response"):
            try:
                # Build backend objects (cached)
                embedding_manager = get_embedding_manager()
                llm_client = get_llm_client()
                retriever = Retriever(
                    collection=embedding_manager.get_collection(),
                    api_key=OPENAI_API_KEY,
                )
                prompt_builder = PromptBuilder()
                loop = RefinementLoop(
                    retriever=retriever,
                    prompt_builder=prompt_builder,
                    llm_client=llm_client,
                )

                # Pass only the assistant-visible history (role + content dicts)
                result = loop.run(
                    question=user_question,
                    history=st.session_state.messages[:-1],  # exclude current user turn
                )

            except Exception as exc:
                st.error(f"An error occurred: {exc}")
                logger.error("Loop error: %s", exc, exc_info=True)
                st.stop()

        # ── Display loop status ────────────────────────────────────────────
        early_stop = " (early termination)" if result.stopped_early else ""
        with st.expander(
            f"Loop status: {result.iterations} iteration(s){early_stop}",
            expanded=False,
        ):
            for i, log_line in enumerate(result.iteration_log, start=1):
                st.markdown(f"**{log_line}**" if i == len(result.iteration_log) else log_line)

        # ── Display retrieved chunks ───────────────────────────────────────
        with st.expander(
            f"Retrieved chunks ({len(result.chunks_used)} used)",
            expanded=False,
        ):
            for i, chunk in enumerate(result.chunks_used, start=1):
                st.markdown(
                    f"**[{i}]** `{chunk.get('source', 'unknown')}` "
                    f"- chunk {chunk.get('chunk_index', '?')} "
                    f"(distance: {chunk.get('distance', 'n/a')})"
                )
                st.text(chunk["text"][:500] + ("..." if len(chunk["text"]) > 500 else ""))
                if i < len(result.chunks_used):
                    st.markdown("---")

        # ── Display the final answer ───────────────────────────────────────
        st.markdown(result.answer)

    # Save assistant reply to conversation history
    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer}
    )
