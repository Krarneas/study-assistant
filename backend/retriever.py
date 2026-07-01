"""
retriever.py — Vector similarity search over stored PDF chunks.

CONTEXT ENGINEERING:
    The whole point of a retriever is to make context selection intelligent.
    Instead of dumping the entire PDF into the LLM prompt (which would be
    slow, expensive, and often exceed the context window), we:

      1. Embed the user's question into the same vector space as the chunks.
      2. Ask ChromaDB for the N nearest neighbours — those are the chunks
         most likely to contain the answer.
      3. Pass only those N chunks to the prompt builder.

    This means the LLM sees *focused* context, which typically produces
    better answers than drowning it in irrelevant text.

LOOP ENGINEERING:
    retrieve_excluding() is used by the refinement loop.  When the first
    answer is deemed too vague, the loop calls this to pull *additional*
    chunks that were not part of the first attempt, widening the context
    without repeating what was already tried.
"""

from typing import Any

import chromadb

from backend.config import EMBEDDING_MODEL, OPENAI_API_KEY, TOP_K_INITIAL
from backend.utils import retry, setup_logger

logger = setup_logger(__name__)


class Retriever:
    """
    Performs semantic search over the ChromaDB chunk collection.

    Typical usage:
        retriever = Retriever(collection, api_key="sk-...")
        chunks = retriever.retrieve("What is binary search?", top_k=3)
    """

    def __init__(self, collection: chromadb.Collection, api_key: str = "") -> None:
        """
        Args:
            collection: ChromaDB collection shared from EmbeddingManager.
            api_key:    OpenAI API key for embedding the query.
        """
        import openai
        resolved_key = api_key or OPENAI_API_KEY
        self._client = openai.OpenAI(api_key=resolved_key)
        self._collection = collection

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K_INITIAL) -> list[dict[str, Any]]:
        """
        Find the top_k chunks most similar to the query.

        CONTEXT ENGINEERING:
            We embed the query and let ChromaDB do cosine-distance search.
            Only top_k results are returned — typically 3–6 chunks, not 200+.

        Args:
            query:  The user's question in plain English.
            top_k:  How many chunks to return.

        Returns:
            List of dicts sorted by relevance (most relevant first):
                text        — chunk text
                chunk_index — position in original PDF
                source      — filename
                distance    — cosine distance (lower = more similar)
        """
        if self._collection.count() == 0:
            logger.warning("ChromaDB collection is empty — no PDF has been uploaded yet.")
            return []

        # Clamp top_k to available chunks
        available = self._collection.count()
        top_k = min(top_k, available)

        query_vector = self._embed_query(query)

        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        return self._format_results(results)

    def retrieve_excluding(
        self,
        query: str,
        top_k: int,
        exclude_indices: list[int],
    ) -> list[dict[str, Any]]:
        """
        Retrieve top_k chunks, skipping any whose chunk_index is in exclude_indices.

        LOOP ENGINEERING:
            The refinement loop calls this after a first attempt that produced
            a vague answer.  By excluding already-seen chunk indices we force
            the retriever to find *new* passages that might contain the missing
            detail.

        Args:
            query:           User's question.
            top_k:           How many new chunks to return.
            exclude_indices: chunk_index values to skip.

        Returns:
            List of chunk dicts (same format as retrieve()).
        """
        # Fetch more than we need, then filter
        fetch_k = top_k + len(exclude_indices) + 5
        candidates = self.retrieve(query, top_k=min(fetch_k, self._collection.count()))

        filtered = [
            c for c in candidates
            if c["chunk_index"] not in exclude_indices
        ]

        logger.debug(
            "retrieve_excluding: fetched %d, excluded %d, returning %d",
            len(candidates), len(exclude_indices), len(filtered[:top_k]),
        )
        return filtered[:top_k]

    # ── Private helpers ────────────────────────────────────────────────────────

    @retry()  # HARNESS ENGINEERING: retry on transient OpenAI errors
    def _embed_query(self, query: str) -> list[float]:
        """Embed a single query string using the OpenAI Embeddings API."""
        response = self._client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query],
        )
        return response.data[0].embedding

    @staticmethod
    def _format_results(results: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Convert ChromaDB's raw query response into a clean list of dicts.

        ChromaDB returns nested lists (one sub-list per query, since you can
        batch queries).  We always send one query at a time, so we take [0].
        """
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        formatted: list[dict[str, Any]] = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            formatted.append({
                "text": doc,
                "chunk_index": meta.get("chunk_index", -1),
                "source": meta.get("source", "unknown"),
                "word_start": meta.get("word_start", 0),
                "word_end": meta.get("word_end", 0),
                "distance": round(dist, 4),
            })

        return formatted
