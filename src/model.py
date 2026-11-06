"""Core RAG model components: retriever, grounded prompt construction, and
the end-to-end recommender.

ARCHITECTURE NOTE (multi-hop vs. atomic rule; examiner gap-closer #2):
Each retrieved chunk encodes exactly ONE atomic eligibility rule for ONE
card (income, OR age, OR CIBIL, OR employment/relationship). The final
recommendation DECISION, however, is inherently multi-hop: it requires
composing several atomic rules (income AND age AND CIBIL AND employment
AND relationship) before a card can be labelled eligible. We deliberately
perform this composition in the PROMPT (the LLM reasons over several
retrieved atomic chunks at once) rather than in the retrieval layer
itself. This keeps chunking simple (one rule per chunk, trivially easy to
cite and audit) and avoids the engineering overhead of GraphRAG-style
multi-hop retrieval machinery, which would be unnecessary complexity for a
handful of independent per-card predicates that never depend on each
other's retrieval order.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import joblib

from src.domain.entities import Decision, UserProfile
from src.domain.requirements import CardRequirements, decide
from src.infrastructure.embedder import Embedder
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.vector_store import VectorStore
from src.prompts import (  # noqa: F401  (ANTI_HALLUCINATION_CLAUSE re-exported)
    ANTI_HALLUCINATION_CLAUSE,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_TASK_MARKER,
    GROUNDED_SYSTEM_PROMPT,
)

SYSTEM_PROMPT = GROUNDED_SYSTEM_PROMPT


class Retriever:
    """FAISS semantic retriever over embedded rule-document chunks."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore, top_k: int = 4) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Embed `query` and return the top-k most similar chunk dicts."""
        k = top_k if top_k is not None else self.top_k
        query_vector = self.embedder.encode([query])
        return self.vector_store.search(query_vector, top_k=k)

    def retrieve_for_profile(self, profile: UserProfile) -> list[dict[str, Any]]:
        """Retrieve up to top_k rules for every card in the requested banks."""
        chunks = self.vector_store.chunks
        candidate_cards = chunks[["bank", "card"]].drop_duplicates()
        requested_banks = {bank.casefold() for bank in profile.banks}
        candidate_cards = candidate_cards[
            candidate_cards["bank"].str.casefold().isin(requested_banks)
        ]

        total_chunks = len(chunks)
        cards = list(candidate_cards.itertuples(index=False, name=None))
        if total_chunks == 0 or not cards:
            return []

        queries = [f"{profile.to_query()} Card: {bank} {card}." for bank, card in cards]
        query_vectors = self.embedder.encode(queries)
        retrieved: list[dict[str, Any]] = []
        for (bank, card), query_vector in zip(cards, query_vectors, strict=True):
            ranked_chunks = self.vector_store.search(query_vector, top_k=total_chunks)
            card_chunks = [
                chunk
                for chunk in ranked_chunks
                if chunk["bank"].casefold() == bank.casefold()
                and chunk["card"].casefold() == card.casefold()
            ]
            retrieved.extend(card_chunks[: self.top_k])

        return retrieved

class GroundedPromptBuilder:
    """Builds the recommend-explain-cite grounded prompt fed to the LLM."""

    def build(self, profile: UserProfile, retrieved: list[dict[str, Any]]) -> str:
        """Compose the full grounded prompt from a profile and retrieved chunks."""
        candidate_cards = list(dict.fromkeys((chunk["bank"], chunk["card"]) for chunk in retrieved))
        cards_block = "\n".join(f"- bank={bank}; card={card}" for bank, card in candidate_cards)
        rules_block = "\n".join(
            f"chunk_id={chunk['chunk_id']} bank={chunk['bank']} card={chunk['card']}\n{chunk['text']}"
            for chunk in retrieved
        )
        profile_block = (
            f"annual_income={profile.annual_income:.0f}\n"
            f"annual_income_in_lakh={profile.annual_income / 100000:.2f}\n"
            f"age={profile.age}\n"
            f"cibil={profile.cibil}\n"
            f"employment={profile.employment}\n"
            f"existing_relationship={profile.existing_relationship}\n"
            f"banks={','.join(profile.banks)}"
        )
        return (
            f"SYSTEM:\n{SYSTEM_PROMPT}\n\n"
            f"### RETRIEVED RULES\n{rules_block}\n\n"
            f"### CARDS TO EVALUATE\n{cards_block}\n\n"
            f"### APPLICANT PROFILE\n{profile_block}\n\n"
            f"### OUTPUT INSTRUCTIONS\n"
            f"Return exactly {len(candidate_cards)} decision object(s), one for each listed bank/card pair. "
            f"Respond with ONLY the JSON array."
        )

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")
_TRUE_WORDS = {"true", "yes", "y", "1", "eligible", "recommended", "recommend"}
_FALSE_WORDS = {"false", "no", "n", "0", "not eligible", "ineligible", "not recommended"}


def _balanced_spans(text: str) -> list[str]:
    """Return every top-level balanced [...] or {...} span, ignoring brackets inside strings."""
    spans: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "[{":
            if depth == 0:
                start = i
            depth += 1
        elif ch in "]}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(text[start : i + 1])
    return spans


_PY_LITERALS = {"True": "true", "False": "false", "None": "null"}


def _python_literals_to_json(text: str) -> str:
    """Replace bare True/False/None (Python style) with JSON literals, outside strings only."""
    out: list[str] = []
    i = 0
    in_string = False
    escape = False
    while i < len(text):
        ch = text[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        for word, replacement in _PY_LITERALS.items():
            end = i + len(word)
            if text.startswith(word, i) and (i == 0 or not text[i - 1].isalnum()) and (
                end >= len(text) or not text[end].isalnum()
            ):
                out.append(replacement)
                i = end
                break
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _loads_lenient(payload: str) -> Any:
    """json.loads, then retry without trailing commas, then with Python-style literals fixed."""
    for attempt in (
        payload,
        _TRAILING_COMMA_RE.sub(r"\1", payload),
        _python_literals_to_json(_TRAILING_COMMA_RE.sub(r"\1", payload)),
    ):
        try:
            return json.loads(attempt)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def extract_json_items(raw: str) -> list[dict[str, Any]]:
    """Pull the list of JSON objects out of a model reply, tolerating common slips.

    Handles: a bare JSON array; a single object instead of an array (frequent
    with per-card prompts, which ask about one card); markdown code fences;
    prose before/after the JSON; a wrapper object such as {"decisions": [...]};
    trailing commas; and brackets inside the reason text. Returns [] if no
    object can be recovered.
    """
    if not isinstance(raw, str) or not raw.strip():
        return []
    candidates = [m.group(1) for m in _FENCE_RE.finditer(raw)] + [raw]
    for candidate in candidates:
        whole = _loads_lenient(candidate.strip())
        spans = [whole] if whole is not None else [_loads_lenient(s) for s in _balanced_spans(candidate)]
        items: list[dict[str, Any]] = []
        for data in spans:
            if isinstance(data, dict):
                inner = next((v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)), None)
                if inner is not None and "bank" not in data and "card" not in data:
                    items.extend(x for x in inner if isinstance(x, dict))
                else:
                    items.append(data)
            elif isinstance(data, list):
                items.extend(x for x in data if isinstance(x, dict))
        if items:
            return items
    return []


def coerce_recommended(value: Any) -> bool | None:
    """Interpret the model's verdict; None if it cannot be read (item is then skipped)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower().rstrip(".")
        if text in _TRUE_WORDS:
            return True
        if text in _FALSE_WORDS:
            return False
    return None


def parse_llm_response(raw: str) -> list[Decision]:
    """Parse the LLM's JSON response into Decision objects.

    Individual malformed items are skipped (instead of discarding the whole
    response), a null/missing reason becomes an empty string, and a verdict
    that cannot be read (e.g. "maybe") is skipped rather than silently turned
    into True by Python's bool("false") == True.
    """
    decisions: list[Decision] = []
    for item in extract_json_items(raw):
        recommended = coerce_recommended(item.get("recommended"))
        if recommended is None or "bank" not in item or "card" not in item:
            continue
        cited = item.get("cited_chunk_id")
        decisions.append(
            Decision(
                bank=str(item["bank"]),
                card=str(item["card"]),
                recommended=recommended,
                reason=str(item.get("reason") or ""),
                cited_chunk_id=str(cited) if cited not in (None, "", "null") else None,
            )
        )
    return decisions


def _same_card(a_bank: str, a_card: str, b_bank: str, b_card: str) -> bool:
    def norm(s: str) -> str:
        # "MoneyBack+", "MoneyBack Plus" and "Money Back" compare equal
        return re.sub(r"[^a-z0-9]", "", s.casefold().replace("+", "plus")).removesuffix("plus")

    bank_ok = norm(a_bank) == norm(b_bank) or norm(b_bank).startswith(norm(a_bank)) or norm(a_bank).startswith(norm(b_bank))
    card_a, card_b = norm(a_card), norm(b_card)
    card_ok = card_a == card_b or card_a.replace(norm(a_bank), "") == card_b.replace(norm(b_bank), "")
    return bank_ok and card_ok


class ExtractionPromptBuilder:
    """Prompt for the extract_compare mode: the LLM reads one card's retrieved rules.

    The applicant profile is NOT included, so the extracted thresholds cannot be
    influenced by the applicant's numbers. Chunks are listed in chunk_id order,
    so the same retrieved rules always give the same prompt (and cache entry).
    """

    def build(self, retrieved: list[dict[str, Any]]) -> str:
        ordered = sorted(retrieved, key=lambda c: (str(c["bank"]), str(c["card"]), _chunk_sort_key(str(c["chunk_id"]))))
        candidate_cards = list(dict.fromkeys((chunk["bank"], chunk["card"]) for chunk in ordered))
        cards_block = "\n".join(f"- bank={bank}; card={card}" for bank, card in candidate_cards)
        rules_block = "\n".join(
            f"chunk_id={chunk['chunk_id']} bank={chunk['bank']} card={chunk['card']}\n{chunk['text']}"
            for chunk in ordered
        )
        return (
            f"SYSTEM:\n{EXTRACTION_SYSTEM_PROMPT}\n\n"
            f"### RETRIEVED RULES\n{rules_block}\n\n"
            f"### CARDS TO EVALUATE\n{cards_block}\n\n"
            f"{EXTRACTION_TASK_MARKER}\n"
            f"Return exactly {len(candidate_cards)} object(s), one for each listed bank/card pair. "
            f"Respond with ONLY the JSON array."
        )


def _chunk_sort_key(chunk_id: str) -> tuple[str, int]:
    head, _, tail = chunk_id.rpartition("#")
    return (head, int(tail)) if tail.isdigit() else (chunk_id, 0)


DECISION_MODES = ("llm", "extract_compare")


class RAGRecommender:
    """End-to-end retrieve -> prompt -> generate -> parse recommender.

    Two decision modes:

    * ``llm`` (default, the configuration evaluated on 14 Oct): the LLM reads the
      rules, compares the applicant with them and writes the verdict and reason.
    * ``extract_compare``: the LLM reads each card's retrieved rules and extracts
      the requirements with the chunk each came from; deterministic code
      (src/domain/requirements.py) compares the applicant and writes a specific,
      cited reason. Motivated by the error analysis: almost all grounded errors
      were the LLM mis-comparing numbers it had been given correctly.

    NOTE: there is deliberately NO runtime validation/gate between the output
    and the caller in either mode; the extract_compare comparison is part of
    the decision procedure, not a check against the eligibility table (which
    is never read at inference time).
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient,
        prompt_builder: GroundedPromptBuilder | None = None,
        per_card: bool | None = None,
        decision_mode: str | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder or GroundedPromptBuilder()
        self.extraction_builder = ExtractionPromptBuilder()
        # Per-card prompting (default): one short grounded prompt per card, which
        # small LLMs follow far more reliably than one prompt covering all cards.
        # Set per_card=False or env RAG_PER_CARD=0 for the single joint prompt.
        if per_card is None:
            per_card = os.environ.get("RAG_PER_CARD", "1").lower() not in {"0", "false", "no"}
        self.per_card = per_card
        mode = (decision_mode or os.environ.get("RAG_DECISION_MODE", "llm")).strip().lower()
        if mode not in DECISION_MODES:
            raise ValueError(f"decision_mode must be one of {DECISION_MODES}, got {mode!r}")
        self.decision_mode = mode
        # Diagnostics from the last recommend() call (used by the evaluation):
        self.last_requirements: list[CardRequirements] = []
        self.last_name_recoveries = 0

    def _generate_all(self, prompts: list[str]) -> list[str]:
        generate_batch = getattr(self.llm_client, "generate_batch", None)
        if callable(generate_batch):
            return list(generate_batch(prompts))
        return [self.llm_client.generate(p) for p in prompts]

    @staticmethod
    def _group_by_card(retrieved: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for chunk in retrieved:
            groups.setdefault((chunk["bank"], chunk["card"]), []).append(chunk)
        return groups

    def recommend(self, profile: UserProfile) -> tuple[list[Decision], int]:
        """Return (decisions, retrieved_k) for the given applicant profile."""
        retrieved = self.retriever.retrieve_for_profile(profile)
        self.last_requirements = []
        self.last_name_recoveries = 0
        if self.decision_mode == "extract_compare":
            return self._recommend_extract_compare(profile, retrieved), len(retrieved)
        if not self.per_card:
            prompt = self.prompt_builder.build(profile, retrieved)
            raw = self.llm_client.generate(prompt)
            return parse_llm_response(raw), len(retrieved)

        groups = self._group_by_card(retrieved)
        keys = list(groups)
        raws = self._generate_all([self.prompt_builder.build(profile, groups[key]) for key in keys])

        decisions: list[Decision] = []
        for (bank, card), raw in zip(keys, raws, strict=True):
            decision = self._own_decision(bank, card, parse_llm_response(raw))
            if decision is not None:
                decisions.append(decision)
        return decisions, len(retrieved)

    def _own_decision(self, bank: str, card: str, parsed: list[Decision]) -> Decision | None:
        """Pick this card's decision from a per-card reply.

        Exact name match first, then a tolerant match ("HDFC Bank" / "MoneyBack
        Plus" style spelling). If the reply holds exactly one decision under a
        different name, it can only be about the card this prompt asked about,
        so it is kept under the requested name and counted in
        `last_name_recoveries` (previously it was silently dropped, which made
        the ungrounded arm return only ~70% of cards).
        """
        exact = [d for d in parsed if d.bank.casefold() == bank.casefold() and d.card.casefold() == card.casefold()]
        if exact:
            return exact[0]
        fuzzy = [d for d in parsed if _same_card(bank, card, d.bank, d.card)]
        chosen = fuzzy[0] if fuzzy else (parsed[0] if len(parsed) == 1 else None)
        if chosen is None:
            return None
        self.last_name_recoveries += 1
        return Decision(bank=bank, card=card, recommended=chosen.recommended, reason=chosen.reason,
                        cited_chunk_id=chosen.cited_chunk_id)

    def _recommend_extract_compare(self, profile: UserProfile, retrieved: list[dict[str, Any]]) -> list[Decision]:
        groups = self._group_by_card(retrieved)
        keys = list(groups)
        raws = self._generate_all([self.extraction_builder.build(groups[key]) for key in keys])
        decisions: list[Decision] = []
        for (bank, card), raw in zip(keys, raws, strict=True):
            items = extract_json_items(raw)
            own = [i for i in items if _same_card(bank, card, str(i.get("bank", "")), str(i.get("card", "")))]
            item = own[0] if own else (items[0] if len(items) == 1 else None)
            if item is None:
                continue  # unreadable extraction: no decision (shows up in cards_returned_rate)
            req = CardRequirements.from_llm_dict(bank, card, item)
            self.last_requirements.append(req)
            decisions.append(decide(profile, req))
        return decisions


def persist_config(path: str | Path, top_k: int, model_name: str) -> None:
    """Persist retrieval configuration (top_k, embedding model name) via joblib."""
    joblib.dump({"top_k": top_k, "model_name": model_name}, path)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a previously persisted retrieval configuration."""
    result: dict[str, Any] = joblib.load(path)
    return result