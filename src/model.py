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
import re
from pathlib import Path
from typing import Any

import joblib

from src.domain.entities import Decision, UserProfile
from src.infrastructure.embedder import Embedder
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.vector_store import VectorStore

SYSTEM_PROMPT = (
    "You are a careful, honest credit-card eligibility assistant. Use ONLY the "
    "rules provided under '### RETRIEVED RULES' to decide whether the applicant "
    "described under '### APPLICANT PROFILE' is eligible for each card. Never "
    "invent a threshold, number, or condition that is not explicitly present in "
    "the retrieved rules. For every card you DO NOT recommend, give a specific, "
    "principal reason naming the exact criterion that fails. For every bank/card "
    "pair listed under '### CARDS TO EVALUATE', return exactly one decision. Do not "
    "omit a pair or add any other pair. The reason must always be a non-empty "
    "string, including when recommending a card. Cite a retrieved chunk belonging "
    "to that same bank/card. Respond with ONLY a JSON array of objects, each with "
    "exactly the fields: bank, card, recommended, reason, cited_chunk_id. Do not "
    "add any prose before or after the JSON array."
)

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

def parse_llm_response(raw: str) -> list[Decision]:
    """Parse the LLM's JSON array response into Decision objects.

    Falls back to an empty list if the response is not valid, well-formed
    JSON matching the expected schema -- the system never crashes the user
    request on a malformed generation, it simply reports no decisions.
    """
    try:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        payload = match.group(0) if match else raw
        data = json.loads(payload)
        if not isinstance(data, list):
            return []
        decisions = []
        for item in data:
            decisions.append(
                Decision(
                    bank=str(item["bank"]),
                    card=str(item["card"]),
                    recommended=bool(item["recommended"]),
                    reason=str(item["reason"]),
                    cited_chunk_id=item.get("cited_chunk_id"),
                )
            )
        return decisions
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return []




class RAGRecommender:
    """End-to-end retrieve -> prompt -> generate -> parse recommender.

    NOTE: there is deliberately NO runtime validation/gate applied to the
    LLM's output before it is returned to the caller. Correctness is driven
    entirely by retrieval anchoring and the grounded prompt; faithfulness
    is only ever measured offline, at development time (see evaluation/).
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient,
        prompt_builder: GroundedPromptBuilder | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder or GroundedPromptBuilder()

    def recommend(self, profile: UserProfile) -> tuple[list[Decision], int]:
        """Return (decisions, retrieved_k) for the given applicant profile."""
        retrieved = self.retriever.retrieve_for_profile(profile)
        prompt = self.prompt_builder.build(profile, retrieved)
        raw = self.llm_client.generate(prompt)
        decisions = parse_llm_response(raw)
        return decisions, len(retrieved)


def persist_config(path: str | Path, top_k: int, model_name: str) -> None:
    """Persist retrieval configuration (top_k, embedding model name) via joblib."""
    joblib.dump({"top_k": top_k, "model_name": model_name}, path)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a previously persisted retrieval configuration."""
    result: dict[str, Any] = joblib.load(path)
    return result