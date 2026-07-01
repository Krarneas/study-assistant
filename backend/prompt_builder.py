"""
prompt_builder.py — Construct the LLM prompt from its constituent parts.

CONTEXT ENGINEERING — THIS IS THE CORE OF IT:
    A language model can only reason about what is in its context window.
    Context Engineering is the discipline of deciding *what goes in* and
    *what stays out*.

    Our prompt has four layers, in order:

    ┌─────────────────────────────────────────────────────────────┐
    │ 1. SYSTEM PROMPT                                            │
    │    Defines the assistant's role and behaviour.              │
    │    Sent once per conversation, not repeated per message.    │
    ├─────────────────────────────────────────────────────────────┤
    │ 2. CONVERSATION HISTORY  (last MAX_HISTORY_EXCHANGES pairs) │
    │    Lets the model remember what was said recently.          │
    │    We deliberately truncate old turns to save tokens.       │
    ├─────────────────────────────────────────────────────────────┤
    │ 3. RETRIEVED CHUNKS (numbered citations)                    │
    │    Only the passages retrieved for THIS question.           │
    │    Never the whole PDF.  This is what makes it efficient.   │
    ├─────────────────────────────────────────────────────────────┤
    │ 4. USER QUESTION                                            │
    │    The actual thing the student wants to know.              │
    └─────────────────────────────────────────────────────────────┘

    What is deliberately *excluded*:
      - The rest of the PDF (not retrieved → not in prompt)
      - Old conversation turns beyond MAX_HISTORY_EXCHANGES
      - Irrelevant metadata (page numbers, word counts, distances)
"""

from typing import Any

from backend.config import MAX_HISTORY_EXCHANGES
from backend.utils import setup_logger

logger = setup_logger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────
# Written once here, reused on every call.  A good system prompt is
# concise — it sets behaviour without eating too many tokens.
SYSTEM_PROMPT = """You are a helpful study assistant for university students.

You have been given numbered excerpts from the student's uploaded lecture notes.
Use ONLY those excerpts to answer the question.  If the answer is not in the
provided excerpts, say so clearly rather than guessing.

Rules:
- Ground every claim in the provided excerpts.
- Cite the excerpt number when you use it, e.g. "[1]" or "[2]".
- Be concise but complete — a student should be able to understand your answer
  without reading the full PDF.
- If the excerpts are insufficient, explicitly say which part is missing and
  suggest what the student might search for.
"""


class PromptBuilder:
    """
    Assembles the list of messages (system + history + context + question)
    that is sent to the OpenAI Chat Completions API.

    The OpenAI API expects messages in this format:
        [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
            {"role": "user",      "content": "<chunks>\n\nQuestion: ..."},
        ]
    """

    def build(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        history: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """
        Build the full message list for a single LLM call.

        CONTEXT ENGINEERING:
            - System prompt   → always first (sets the rules)
            - Trimmed history → only the last N exchanges (saves tokens)
            - Chunks block    → only the retrieved passages (not the full PDF)
            - Question        → the user's current question

        Args:
            question: The current user question.
            chunks:   Retrieved chunks from Retriever.retrieve().
            history:  Full conversation history as list of
                      {"role": "user"|"assistant", "content": "..."} dicts.

        Returns:
            List of message dicts ready for openai.chat.completions.create().
        """
        messages: list[dict[str, str]] = []

        # Layer 1 — System prompt
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

        # Layer 2 — Trimmed conversation history
        # CONTEXT ENGINEERING: we only include the last MAX_HISTORY_EXCHANGES
        # question-answer pairs.  Older turns are dropped to keep the prompt
        # short and focused.
        trimmed_history = self._trim_history(history)
        messages.extend(trimmed_history)
        logger.debug(
            "History: %d total turns → %d kept (max %d exchanges = %d turns)",
            len(history), len(trimmed_history),
            MAX_HISTORY_EXCHANGES, MAX_HISTORY_EXCHANGES * 2,
        )

        # Layer 3 + 4 — Retrieved chunks + question (combined into one user turn)
        # We combine them so the model sees the evidence immediately before
        # the question it needs to answer — this improves answer grounding.
        user_content = self._build_user_turn(question, chunks)
        messages.append({"role": "user", "content": user_content})

        # Log a summary (not the full content) for debugging
        total_chars = sum(len(m["content"]) for m in messages)
        logger.debug(
            "Built prompt: %d messages, ~%d chars (~%d tokens estimated)",
            len(messages), total_chars, total_chars // 4,
        )

        return messages

    def format_chunks(self, chunks: list[dict[str, Any]]) -> str:
        """
        Render retrieved chunks as a numbered citation block.

        Example output:
            [1] Binary search is an algorithm that finds a target value...
                (Source: lecture3.pdf, chunk 7)

            [2] The time complexity of binary search is O(log n)...
                (Source: lecture3.pdf, chunk 8)
        """
        if not chunks:
            return "(No relevant excerpts found in the uploaded document.)"

        lines: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            source_info = f"Source: {chunk.get('source', 'unknown')}, chunk {chunk.get('chunk_index', '?')}"
            lines.append(f"[{i}] {chunk['text'].strip()}")
            lines.append(f"    ({source_info})")
            lines.append("")  # blank line between citations

        return "\n".join(lines).strip()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _trim_history(self, history: list[dict[str, str]]) -> list[dict[str, str]]:
        """
        Keep only the last MAX_HISTORY_EXCHANGES user/assistant pairs.

        CONTEXT ENGINEERING:
            Each exchange is one user turn + one assistant turn = 2 messages.
            We keep the most recent ones because they are most relevant to
            the current question.  Earlier messages are silently dropped.
        """
        max_turns = MAX_HISTORY_EXCHANGES * 2  # each exchange = 2 messages
        if len(history) <= max_turns:
            return list(history)
        return history[-max_turns:]

    def _build_user_turn(
        self,
        question: str,
        chunks: list[dict[str, Any]],
    ) -> str:
        """
        Combine the retrieved chunks and the user's question into one message.

        Layout:
            RELEVANT EXCERPTS FROM YOUR LECTURE NOTES:
            [1] ...
            [2] ...

            Question: <user question>
        """
        chunks_text = self.format_chunks(chunks)

        return (
            "RELEVANT EXCERPTS FROM YOUR LECTURE NOTES:\n"
            f"{chunks_text}\n\n"
            f"Question: {question}"
        )
