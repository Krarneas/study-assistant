"""
embeddings.py — Generate embeddings and persist chunks to ChromaDB.

HARNESS ENGINEERING:
    This module owns the vector database.  No other file calls ChromaDB
    directly.  If we ever switch from ChromaDB to Pinecone or pgvector,
    only this file changes.

    Responsibilities:
      - Create / open the persistent ChromaDB collection
      - Call the OpenAI Embeddings API (with retry)
      - Upsert chunks so re-uploading the same PDF is idempotent
      - Expose clear_collection() for the "Clear Database" UI button

CONTEXT ENGINEERING (why embeddings matter):
    An embedding is a list of numbers (a vector) that captures the *meaning*
    of a piece of text.  Two chunks that discuss the same concept will have
    vectors that are close together in high-dimensional space.

    When the user asks a question, we embed the question too, then find the
    chunks whose vectors are closest — those are the most relevant passages.
    Only *those* chunks go into the LLM prompt, not the whole PDF.
"""

from typing import Any

import chromadb
import openai

from backend.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)
from backend.utils import retry, setup_logger

logger = setup_logger(__name__)


class EmbeddingManager:
    """
    Manages OpenAI embedding generation and ChromaDB persistence.

    Typical usage:
        manager = EmbeddingManager(api_key="sk-...")
        manager.add_chunks(chunks)          # after PDF load
        count = manager.get_chunk_count()
    """

    def __init__(self, api_key: str = "") -> None:
        """
        Args:
            api_key: OpenAI API key.  Falls back to config if not provided.
        """
        resolved_key = api_key or OPENAI_API_KEY
        if not resolved_key:
            raise ValueError(
                "OpenAI API key is missing.  Set OPENAI_API_KEY in your .env "
                "file or pass it through the Streamlit sidebar."
            )

        self._client = openai.OpenAI(api_key=resolved_key)

        # ChromaDB PersistentClient stores data on disk so it survives restarts.
        # HARNESS ENGINEERING: we point it at our dedicated database/ directory.
        self._chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._collection = self._get_or_create_collection()

        logger.info(
            "EmbeddingManager ready. Collection '%s' has %d chunks.",
            CHROMA_COLLECTION_NAME,
            self.get_chunk_count(),
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """
        Embed a list of text chunks and upsert them into ChromaDB.

        We use upsert (not add) so that uploading the same PDF twice does not
        create duplicate entries — the existing record is simply overwritten.

        Args:
            chunks: List of dicts from PDFLoader.load_and_chunk().
                    Each dict must have 'text', 'chunk_index', 'source'.
        """
        if not chunks:
            logger.warning("add_chunks called with empty list — nothing to do.")
            return

        texts = [c["text"] for c in chunks]
        logger.info("Embedding %d chunks via OpenAI (%s)…", len(texts), EMBEDDING_MODEL)

        vectors = self.embed_texts(texts)

        # Build the parallel lists that ChromaDB expects
        ids = [f"{c['source']}__chunk_{c['chunk_index']}" for c in chunks]
        metadatas = [
            {
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "word_start": c.get("word_start", 0),
                "word_end": c.get("word_end", 0),
            }
            for c in chunks
        ]

        # Upsert in one call — ChromaDB handles batching internally
        self._collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("Upserted %d chunks into ChromaDB.", len(chunks))

    @retry()  # HARNESS ENGINEERING: retry on transient API errors
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Call the OpenAI Embeddings API and return a vector for each text.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (each is a list of floats).
        """
        response = self._client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        # The API returns results in the same order as the input
        vectors = [item.embedding for item in response.data]
        logger.debug("Received %d embedding vectors (dim=%d).", len(vectors), len(vectors[0]))
        return vectors

    def clear_collection(self) -> None:
        """
        Delete all stored chunks and recreate the collection from scratch.
        Called when the user clicks "Clear Database" in the UI.
        """
        try:
            self._chroma.delete_collection(CHROMA_COLLECTION_NAME)
            logger.info("Deleted ChromaDB collection '%s'.", CHROMA_COLLECTION_NAME)
        except Exception:
            pass  # collection may not exist yet — that's fine
        self._collection = self._get_or_create_collection()
        logger.info("Recreated empty collection '%s'.", CHROMA_COLLECTION_NAME)

    def get_chunk_count(self) -> int:
        """Return the number of chunks currently stored in ChromaDB."""
        return self._collection.count()

    def get_collection(self) -> chromadb.Collection:
        """Expose the raw collection so Retriever can share it."""
        return self._collection

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get the ChromaDB collection, creating it if it doesn't exist."""
        # cosine distance is standard for semantic similarity with OpenAI vectors
        return self._chroma.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
