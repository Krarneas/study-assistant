"""
pdf_loader.py — PDF ingestion: save, extract text, split into chunks.

HARNESS ENGINEERING:
    All file-system and PDF-parsing concerns are contained here.
    No other module reads files or knows about PyMuPDF.  If we ever
    switch to pdfplumber or a remote storage bucket, only this file
    changes.

CONTEXT ENGINEERING (chunking):
    We deliberately break the PDF into small, overlapping pieces rather
    than sending the whole document to the LLM.  Reasons:
      - LLMs have a limited context window (e.g. 128 k tokens for GPT-4o).
      - Sending 200 pages would be expensive and slow.
      - Relevant passages can be retrieved precisely via vector search.
    The overlap (CHUNK_OVERLAP words shared between adjacent chunks) prevents
    a sentence that straddles a chunk boundary from being cut in half.
"""

from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from backend.config import CHUNK_OVERLAP, CHUNK_SIZE, UPLOAD_DIR
from backend.utils import setup_logger

logger = setup_logger(__name__)


class PDFLoader:
    """Handles saving, text extraction, and chunking of PDF files."""

    def __init__(self) -> None:
        self.upload_dir: Path = UPLOAD_DIR

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_and_chunk(self, file_bytes: bytes, filename: str) -> list[dict[str, Any]]:
        """
        Full pipeline: save → extract → chunk.

        Args:
            file_bytes: Raw bytes of the uploaded PDF.
            filename:   Original filename (used for saving and metadata).

        Returns:
            List of chunk dicts, each with keys:
                text        — the chunk's text content
                chunk_index — position in the sequence (0-based)
                source      — original filename
                page_range  — approximate page numbers covered
        """
        saved_path = self._save_file(file_bytes, filename)
        text, page_texts = self._extract_text(saved_path)
        chunks = self._chunk_text(text, filename)
        logger.info("Loaded '%s': %d chars → %d chunks", filename, len(text), len(chunks))
        return chunks

    def load(self, file_bytes: bytes, filename: str) -> str:
        """Save and extract raw text only (no chunking)."""
        saved_path = self._save_file(file_bytes, filename)
        text, _ = self._extract_text(saved_path)
        return text

    # ── Private helpers ────────────────────────────────────────────────────────

    def _save_file(self, file_bytes: bytes, filename: str) -> Path:
        """Persist uploaded bytes to the uploads/ directory."""
        # Sanitise filename: replace spaces with underscores
        safe_name = filename.replace(" ", "_")
        dest = self.upload_dir / safe_name
        dest.write_bytes(file_bytes)
        logger.info("Saved uploaded file to %s (%d bytes)", dest, len(file_bytes))
        return dest

    def _extract_text(self, pdf_path: Path) -> tuple[str, list[str]]:
        """
        Extract text from every page of a PDF using PyMuPDF.

        Returns:
            full_text  — all pages joined with newlines
            page_texts — list of per-page text strings
        """
        page_texts: list[str] = []

        # fitz.open() can accept a file path or bytes
        with fitz.open(str(pdf_path)) as doc:
            for page_num, page in enumerate(doc):
                page_text = page.get_text()  # plain text extraction
                if page_text.strip():          # skip blank pages
                    page_texts.append(page_text)
                    logger.debug("Page %d: %d chars", page_num + 1, len(page_text))

        if not page_texts:
            raise ValueError(
                f"No extractable text found in '{pdf_path.name}'. "
                "The PDF may be scanned (image-only) and would need OCR."
            )

        full_text = "\n".join(page_texts)
        logger.info("Extracted %d chars from %d pages", len(full_text), len(page_texts))
        return full_text, page_texts

    def _chunk_text(self, text: str, source: str) -> list[dict[str, Any]]:
        """
        Split text into overlapping word-based chunks.

        CONTEXT ENGINEERING:
            Word-based splitting (not character-based) keeps chunks at a
            predictable token count.  The overlap means the retriever can
            find context even when a key sentence sits near a boundary.

        Args:
            text:   Full extracted text of the PDF.
            source: Filename, stored as metadata on every chunk.

        Returns:
            List of chunk dicts.
        """
        words = text.split()  # split on whitespace

        if not words:
            return []

        chunks: list[dict[str, Any]] = []
        chunk_index = 0
        start = 0

        while start < len(words):
            end = start + CHUNK_SIZE
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "text": chunk_text,
                "chunk_index": chunk_index,
                "source": source,
                # Approximate character positions for debugging
                "word_start": start,
                "word_end": min(end, len(words)),
            })

            chunk_index += 1
            # Advance by (CHUNK_SIZE - CHUNK_OVERLAP) so the next chunk
            # starts CHUNK_OVERLAP words before the current chunk ended.
            start += CHUNK_SIZE - CHUNK_OVERLAP

        logger.debug(
            "Chunked into %d pieces (size=%d, overlap=%d)",
            len(chunks), CHUNK_SIZE, CHUNK_OVERLAP,
        )
        return chunks
