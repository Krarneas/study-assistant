"""
loop.py — Iterative answer refinement loop.

LOOP ENGINEERING:
    A naive RAG system asks once and returns whatever it gets.
    Loop Engineering adds a quality-check step: after each answer we
    evaluate whether it is well-grounded in the retrieved material.
    If not, we widen the retrieval and ask again.

    The loop here is *bounded* (at most MAX_LOOP_ITERATIONS) and
    *transparent* (the UI shows what happened in each iteration).
    It is NOT an autonomous agent — it cannot browse the web, run code,
    or take actions outside this application.

    Iteration flow:
    ┌──────────────────────────────────────────────────────────────┐
    │  Iteration N                                                 │
    │    1. Retrieve top-k chunks (excluding previously seen ones) │
    │    2. Build prompt  (system + history + chunks + question)   │
    │    3. Call LLM → get answer                                  │
    │    4. Evaluate answer quality                                │
    │       • Good → return immediately                            │
    │       • Vague → increment N and loop                         │
    │  After MAX_LOOP_ITERATIONS → return best answer found        │
    └──────────────────────────────────────────────────────────────┘

    Why is this useful?
    Sometimes the first 3 chunks retrieved do not contain the exact
    detail needed.  A second pass with 3 more chunks often fills the gap
    without the cost of always sending 6+ chunks on every query.
"""

from dataclasses import dataclass, field
from typing import Any

from backend.config import MAX_LOOP_ITERATIONS, TOP_K_EXTRA, TOP_K_INITIAL
from backend.llm import LLMClient
from backend.prompt_builder import PromptBuilder
from backend.retriever import Retriever
from backend.utils import setup_logger

logger = setup_logger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class LoopResult:
    """
    Everything the UI needs to display after the loop completes.

    Attributes:
        answer:        The final answer from the LLM.
        iterations:    How many loop iterations were run (1–MAX_LOOP_ITERATIONS).
        chunks_used:   All chunks that were fed into the last prompt.
        iteration_log: Human-readable log entry for each iteration.
        stopped_early: True if the loop exited before MAX_LOOP_ITERATIONS
                       because the answer was deemed good enough.
    """
    answer: str
    iterations: int
    chunks_used: list[dict[str, Any]]
    iteration_log: list[str] = field(default_factory=list)
    stopped_early: bool = False


# ── Vagueness heuristics ───────────────────────────────────────────────────────

# Phrases that suggest the LLM could not find the answer in the chunks.
# If the answer contains any of these, we consider it vague.
_VAGUE_PHRASES = [
    "i don't know",
    "i do not know",
    "not mentioned",
    "not provided",
    "not found",
    "no information",
    "cannot find",
    "not in the",
    "not covered",
    "not discussed",
    "not available",
]

# Minimum character length for an answer to be considered non-trivial.
_MIN_ANSWER_LENGTH = 80


class RefinementLoop:
    """
    Orchestrates the iterative retrieve → generate → evaluate cycle.

    Typical usage:
        loop = RefinementLoop(retriever, prompt_builder, llm_client)
        result = loop.run("What is binary search?", history=[])
        print(result.answer)
        for log_line in result.iteration_log:
            print(log_line)
    """

    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        llm_client: LLMClient,
    ) -> None:
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._llm = llm_client

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        question: str,
        history: list[dict[str, str]],
    ) -> LoopResult:
        """
        Run the refinement loop for a single user question.

        LOOP ENGINEERING:
            - Starts with TOP_K_INITIAL chunks.
            - On each subsequent iteration adds TOP_K_EXTRA new chunks
              (ones not already seen in this loop run).
            - Stops early if the answer passes the quality check.
            - Always stops after MAX_LOOP_ITERATIONS regardless.

        Args:
            question: The user's question in plain English.
            history:  Conversation history (list of role/content dicts).

        Returns:
            LoopResult with the final answer and iteration metadata.
        """
        iteration_log: list[str] = []
        seen_indices: list[int] = []    # chunk_indices used so far in this run
        all_chunks: list[dict[str, Any]] = []
        best_answer = ""
        best_chunks: list[dict[str, Any]] = []

        for iteration in range(1, MAX_LOOP_ITERATIONS + 1):
            logger.info("Loop iteration %d / %d", iteration, MAX_LOOP_ITERATIONS)

            # ── Step 1: Retrieve chunks ────────────────────────────────────
            if iteration == 1:
                new_chunks = self._retriever.retrieve(question, top_k=TOP_K_INITIAL)
            else:
                # Fetch additional chunks that were NOT used in prior iterations
                new_chunks = self._retriever.retrieve_excluding(
                    question,
                    top_k=TOP_K_EXTRA,
                    exclude_indices=seen_indices,
                )

            if not new_chunks and not all_chunks:
                log_entry = (
                    f"Iteration {iteration}: No chunks found in the database. "
                    "Please upload a PDF first."
                )
                iteration_log.append(log_entry)
                logger.warning(log_entry)
                break

            # Accumulate chunks across iterations
            all_chunks.extend(new_chunks)
            seen_indices.extend(c["chunk_index"] for c in new_chunks)

            chunk_summary = ", ".join(
                f"chunk {c['chunk_index']}" for c in new_chunks
            ) or "none"

            # ── Step 2: Build prompt ───────────────────────────────────────
            messages = self._prompt_builder.build(
                question=question,
                chunks=all_chunks,
                history=history,
            )

            # ── Step 3: Call LLM ───────────────────────────────────────────
            answer = self._llm.complete(messages)
            best_answer = answer
            best_chunks = list(all_chunks)

            # ── Step 4: Evaluate ───────────────────────────────────────────
            is_vague = self._is_answer_vague(answer, all_chunks)
            evaluation = self._evaluate_answer(answer, all_chunks, is_vague)

            log_entry = (
                f"Iteration {iteration}: Retrieved {len(new_chunks)} new chunk(s) "
                f"({chunk_summary}). "
                f"Total context: {len(all_chunks)} chunk(s). "
                f"Evaluation: {evaluation}"
            )
            iteration_log.append(log_entry)
            logger.info(log_entry)

            if not is_vague:
                # Answer is good enough — exit early
                logger.info("Answer passed quality check at iteration %d.", iteration)
                return LoopResult(
                    answer=answer,
                    iterations=iteration,
                    chunks_used=best_chunks,
                    iteration_log=iteration_log,
                    stopped_early=(iteration < MAX_LOOP_ITERATIONS),
                )

            # Answer was vague — log and try again (if iterations remain)
            if iteration < MAX_LOOP_ITERATIONS:
                logger.info(
                    "Answer vague at iteration %d — fetching more context.", iteration
                )

        # Exhausted all iterations — return the best answer we found
        iteration_log.append(
            f"Reached maximum iterations ({MAX_LOOP_ITERATIONS}). "
            "Returning best answer found."
        )
        return LoopResult(
            answer=best_answer,
            iterations=MAX_LOOP_ITERATIONS,
            chunks_used=best_chunks,
            iteration_log=iteration_log,
            stopped_early=False,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _is_answer_vague(
        self,
        answer: str,
        chunks: list[dict[str, Any]],
    ) -> bool:
        """
        Heuristic check: is the answer too vague to be useful?

        LOOP ENGINEERING:
            We use cheap, local heuristics rather than a second LLM call
            (which would double the cost).  The heuristics catch the most
            common failure modes:
              1. Answer is very short (the model likely said "I don't know").
              2. Answer contains a vague-phrase like "not mentioned".
              3. Answer shares no key terms with the retrieved chunks
                 (the model may be hallucinating or ignoring the context).

        Args:
            answer: The LLM's reply.
            chunks: The chunks that were in the prompt.

        Returns:
            True if the answer is considered vague (loop should continue).
        """
        if not answer:
            return True

        answer_lower = answer.lower()

        # Heuristic 1: too short
        if len(answer.strip()) < _MIN_ANSWER_LENGTH:
            logger.debug("Vague: answer too short (%d chars).", len(answer.strip()))
            return True

        # Heuristic 2: contains a known vague phrase
        for phrase in _VAGUE_PHRASES:
            if phrase in answer_lower:
                logger.debug("Vague: answer contains '%s'.", phrase)
                return True

        # Heuristic 3: answer shares very few words with the retrieved chunks
        # Build a set of "significant" words from the chunks (length > 4)
        if chunks:
            chunk_words: set[str] = set()
            for chunk in chunks:
                for word in chunk["text"].lower().split():
                    if len(word) > 4:
                        chunk_words.add(word.strip(".,;:()[]\"'"))

            answer_words = {
                w.strip(".,;:()[]\"'")
                for w in answer_lower.split()
                if len(w) > 4
            }

            overlap = answer_words & chunk_words
            # If fewer than 3 significant words overlap, the answer may be
            # ignoring the retrieved material entirely.
            if len(overlap) < 3:
                logger.debug(
                    "Vague: low word overlap with chunks (%d shared words).",
                    len(overlap),
                )
                return True

        return False  # answer passes all checks

    @staticmethod
    def _evaluate_answer(
        answer: str,
        chunks: list[dict[str, Any]],
        is_vague: bool,
    ) -> str:
        """Return a short human-readable quality verdict for the UI."""
        if is_vague:
            if len(answer.strip()) < _MIN_ANSWER_LENGTH:
                return "Too short — needs more context."
            return "Answer appears vague or not grounded — fetching more chunks."
        return f"Good answer ({len(answer)} chars, well-grounded in retrieved material)."
