"""Natural-language interface around the frozen extract_compare recommender.

Added 17-18 Nov 2026 after supervisor feedback ("user requirement in natural
language -> LLM -> vector database -> comparison engine"). Flow:

    applicant message
      -> QueryParser   (LLM reads the message; Python converts units)  src/domain/query_parsing.py
      -> missing field? ask a follow-up question instead of guessing
      -> CardRanker    (vector database: the existing FAISS index of rule chunks,
                        ranks cards by what the applicant asked for)
      -> RAGRecommender in extract_compare mode (UNCHANGED: retrieval, extraction,
                        deterministic comparison, fail-closed guard)
      -> AnswerWriter  (LLM phrases the decisions; src/domain/answer_guard.py rejects
                        a sentence with a wrong verdict or an unsupported number and
                        falls back to the engine's own reason)

Nothing in the decision path changes, so the frozen round 5.5 results still
describe the decisions this interface returns. Only the two ends are new, and
they are evaluated separately by evaluation/run_nl_eval.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.domain.answer_guard import check_sentence
from src.domain.entities import Decision, UserProfile
from src.domain.query_parsing import ParsedQuery
from src.infrastructure.llm_client import LLMClient
from src.model import RAGRecommender, _same_card, extract_json_items
from src.prompts import (
    ANSWER_WRITER_SYSTEM_PROMPT,
    ANSWER_WRITER_TASK_MARKER,
    QUERY_PARSE_TASK_MARKER,
    QUERY_PARSER_SYSTEM_PROMPT,
)

MESSAGE_HEADING = "### APPLICANT MESSAGE"
DECISIONS_HEADING = "### DECISIONS"


class QueryParser:
    """LLM #1: applicant message -> ParsedQuery."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    @staticmethod
    def build_prompt(message: str) -> str:
        return (
            f"SYSTEM:\n{QUERY_PARSER_SYSTEM_PROMPT}\n\n"
            f"{MESSAGE_HEADING}\n{message.strip()}\n\n"
            f"{QUERY_PARSE_TASK_MARKER}\n"
            "Respond with ONLY the JSON object."
        )

    def parse(self, message: str) -> ParsedQuery:
        raw = self.llm_client.generate(self.build_prompt(message))
        items = extract_json_items(raw)
        return ParsedQuery.from_llm_dict(items[0] if items else {})


class CardRanker:
    """Ranks cards by similarity of the applicant's stated preferences to each
    card's rule chunks in the existing vector database (FAISS + chunks.parquet).

    Ranking only ORDERS the answer; every card of the requested banks is still
    decided, so a weak ranking can never hide a card the applicant qualifies for.
    """

    def __init__(self, embedder: Any, vector_store: Any) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def rank(self, preferences: str, banks: list[str]) -> dict[tuple[str, str], float]:
        chunks = self.vector_store.chunks
        wanted = {b.casefold() for b in banks}
        cards = [
            (str(b), str(c))
            for b, c in chunks[["bank", "card"]].drop_duplicates().itertuples(index=False, name=None)
            if str(b).casefold() in wanted
        ]
        if not preferences.strip():
            return {card: 0.0 for card in cards}
        query = self.embedder.encode([f"A credit card for {preferences.strip()}"])
        hits = self.vector_store.search(query, top_k=len(chunks))
        scores: dict[tuple[str, str], float] = {card: -1.0 for card in cards}
        for hit in hits:
            key = (str(hit["bank"]), str(hit["card"]))
            if key in scores:
                scores[key] = max(scores[key], float(hit["score"]))
        return scores


@dataclass(frozen=True)
class CardAnswer:
    bank: str
    card: str
    recommended: bool
    needs_review: bool
    sentence: str
    source: str  # "llm" | "engine_reason"
    guard: str
    cited_chunk_id: str | None
    engine_reason: str


class AnswerWriter:
    """LLM #2: phrases the engine's decisions; never changes them."""

    def __init__(self, llm_client: LLMClient | None) -> None:
        self.llm_client = llm_client

    @staticmethod
    def build_prompt(decisions: list[Decision]) -> str:
        facts = [
            {
                "card": d.card,
                "eligible": d.recommended and not d.needs_review,
                "needs_review": d.needs_review,
                "engine_reason": d.reason,
            }
            for d in decisions
        ]
        return (
            f"SYSTEM:\n{ANSWER_WRITER_SYSTEM_PROMPT}\n\n"
            f"{DECISIONS_HEADING}\n{json.dumps(facts, ensure_ascii=False, indent=1)}\n\n"
            f"{ANSWER_WRITER_TASK_MARKER}\n"
            f"Return one sentence for each of the {len(decisions)} card(s). Respond with ONLY the JSON object."
        )

    def write(self, decisions: list[Decision], profile: UserProfile) -> list[CardAnswer]:
        sentences: dict[str, Any] = {}
        if self.llm_client is not None and decisions:
            items = extract_json_items(self.llm_client.generate(self.build_prompt(decisions)))
            for item in items:
                sentences.update({str(k): v for k, v in item.items()})
        answers = []
        for d in decisions:
            sentence = sentences.get(d.card)
            if sentence is None:
                sentence = next((v for k, v in sentences.items() if _same_card(d.bank, d.card, d.bank, k)), None)
            ok, why = check_sentence(sentence, d, profile) if self.llm_client is not None else (False, "writer off")
            answers.append(
                CardAnswer(
                    bank=d.bank,
                    card=d.card,
                    recommended=d.recommended,
                    needs_review=d.needs_review,
                    sentence=str(sentence) if ok else d.reason,
                    source="llm" if ok else "engine_reason",
                    guard=why,
                    cited_chunk_id=d.cited_chunk_id,
                    engine_reason=d.reason,
                )
            )
        return answers


@dataclass
class NLResult:
    status: str  # "ok" | "need_more_info"
    understood: ParsedQuery
    questions: list[str] = field(default_factory=list)
    answers: list[CardAnswer] = field(default_factory=list)
    retrieved_k: int = 0

    @property
    def text(self) -> str:
        if self.status != "ok":
            return "To check your eligibility I need a little more information:\n- " + "\n- ".join(self.questions)
        eligible = [a for a in self.answers if a.recommended and not a.needs_review]
        lines = [f"You qualify for {len(eligible)} of the {len(self.answers)} cards checked.", ""]
        for a in self.answers:
            mark = "?" if a.needs_review else ("✔" if a.recommended else "✘")
            cite = f"  [rule: {a.cited_chunk_id}]" if a.cited_chunk_id and not a.recommended else ""
            lines.append(f"{mark} {a.card}: {a.sentence}{cite}")
        return "\n".join(lines)


class NLRecommender:
    """Natural-language front end for an extract_compare RAGRecommender."""

    def __init__(
        self,
        parser: QueryParser,
        recommender: RAGRecommender,
        ranker: CardRanker | None = None,
        writer: AnswerWriter | None = None,
    ) -> None:
        if recommender.decision_mode != "extract_compare":
            raise ValueError("the natural-language interface uses the frozen extract_compare design")
        self.parser = parser
        self.recommender = recommender
        self.ranker = ranker
        self.writer = writer or AnswerWriter(None)

    def ask(self, message: str, previous: ParsedQuery | None = None) -> NLResult:
        parsed = self.parser.parse(message)
        if previous is not None:
            parsed = previous.merged_with(parsed)
        if parsed.missing:
            return NLResult(status="need_more_info", understood=parsed, questions=parsed.follow_up_questions())
        return self.decide(parsed)

    def decide(self, parsed: ParsedQuery) -> NLResult:
        profile = parsed.to_profile()
        decisions, retrieved_k = self.recommender.recommend(profile)
        scores = self.ranker.rank(parsed.preferences, profile.banks) if self.ranker else {}

        def order(d: Decision) -> tuple[int, float]:
            group = 1 if d.needs_review else (0 if d.recommended else 2)
            return group, -scores.get((d.bank, d.card), 0.0)

        decisions = sorted(decisions, key=order)
        return NLResult(
            status="ok", understood=parsed, answers=self.writer.write(decisions, profile), retrieved_k=retrieved_k
        )
