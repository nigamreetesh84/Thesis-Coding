"""
Ablation harness: A1 grounded-vs-ungrounded (with McNemar significance),
plus the full A1-A6 ablation matrix.

RQ mapping printed alongside each metric:
  RQ1 faithfulness               -> RAGAS + gold-set accuracy (classification_metrics here;
                                    see also evaluation/ragas_faithfulness.py)
  RQ2 context precision/recall   -> evaluation/metrics.py::context_precision_recall
  RQ3 hallucination reduction    -> McNemar test on grounded vs ungrounded correctness (this module)
  RQ4 explanation correctness    -> consistent-with-expected AND traceable-to-cited-chunk (this module)

HONESTY NOTE: all figures below are computed on a SYNTHETIC gold set against
CURATED, representative (not live) rule documents for exactly two banks
(SBI, HDFC). They are an OPTIMISTIC UPPER BOUND on real-world performance,
not a claim of generalisation to other banks, live documents, or production
traffic. There is also NO runtime validation gate in the live system: these
offline numbers are the only place faithfulness is ever checked.

Run as a script: `python -m evaluation.run_ablation`
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from statsmodels.stats.contingency_tables import mcnemar

from config.settings import settings
from evaluation.metrics import (
    EXTRACTION_FIELDS,
    classification_metrics,
    context_precision_recall,
    extraction_field_matches,
    wilson_interval,
)
from src.domain.eligibility import EligibilityOracle
from src.domain.entities import Decision, UserProfile
from src.infrastructure.embedder import Embedder
from src.infrastructure.llm_client import (
    LLMClient,
    LocalLLMClient,
    StubLLM,
    UngroundedLLMClient,
    UngroundedStubLLM,
)
from src.infrastructure.vector_store import VectorStore
from src.model import (
    ANTI_HALLUCINATION_CLAUSE,
    DECISION_MODES,
    GroundedPromptBuilder,
    RAGRecommender,
    Retriever,
)

STRETCH_TARGET_ACCURACY = 0.90
ALPHA = 0.05


@dataclass(frozen=True)
class GoldRecord:
    """One deserialized row of data/gold_test_set.jsonl."""

    profile: UserProfile
    expected: list[Decision]
    variant: str
    difficulty: str
    profile_text: str = ""


def load_gold_set(path: str | Path = settings.gold_test_set_path) -> list[GoldRecord]:
    """Load the gold evaluation set produced by evaluation/build_gold_set.py."""
    records: list[GoldRecord] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            profile = UserProfile(**row["profile"])
            expected = [Decision(**d) for d in row["expected"]]
            records.append(
                GoldRecord(
                    profile=profile,
                    expected=expected,
                    variant=row["variant"],
                    difficulty=row["difficulty"],
                    profile_text=row.get("profile_text", ""),
                )
            )
    return records


def _decisions_to_map(decisions: list[Decision]) -> dict[tuple[str, str], bool]:
    return {(d.bank, d.card): d.recommended for d in decisions}


def profile_correct(actual: list[Decision], expected: list[Decision]) -> bool:
    """A profile is 'correct' iff every expected card decision is matched."""
    actual_map = _decisions_to_map(actual)
    expected_map = _decisions_to_map(expected)
    if set(actual_map.keys()) != set(expected_map.keys()):
        return False
    return all(actual_map[k] == expected_map[k] for k in expected_map)


def citation_is_valid(decision: Decision, retrieved: list[dict[str, Any]]) -> bool:
    """Check that a cited chunk was retrieved and belongs to this decision's card."""
    return decision.cited_chunk_id is not None and any(
        chunk["chunk_id"] == decision.cited_chunk_id
        and chunk["bank"] == decision.bank
        and chunk["card"] == decision.card
        for chunk in retrieved
    )


def build_retriever(top_k: int) -> Retriever:
    """Construct a Retriever wired to the default embedder and vector store."""
    embedder = Embedder(settings.model_name)
    vector_store = VectorStore(settings.faiss_index_path, settings.chunks_parquet_path).load()
    return Retriever(embedder, vector_store, top_k=top_k)


def _progress(items: list[GoldRecord]) -> Any:
    """Wrap an iterable in a tqdm progress bar when tqdm is installed."""
    try:
        from tqdm import tqdm

        return tqdm(items, leave=False)
    except ImportError:
        return items


def run_arm(
    recommender: RAGRecommender,
    records: list[GoldRecord],
    stats: dict[str, int] | None = None,
    collect: list[dict[str, Any]] | None = None,
) -> list[bool]:
    """Run one ablation arm over the gold set, returning per-profile correctness.

    `stats` (optional) accumulates how many card decisions were expected vs
    actually returned, so a parse failure or refusal shows up as low coverage
    instead of silently counting as "wrong". `collect` (optional) receives one
    dict per profile with expected vs actual decisions, for error analysis.
    """
    results: list[bool] = []
    for record in _progress(records):
        decisions = recommender.recommend(record.profile)[0]
        correct = profile_correct(decisions, record.expected)
        results.append(correct)
        expected_cards = {(d.bank, d.card) for d in record.expected}
        returned = {(d.bank, d.card) for d in decisions} & expected_cards
        if stats is not None:
            expected_map = _decisions_to_map(record.expected)
            actual_map = _decisions_to_map(decisions)
            stats["profiles"] = stats.get("profiles", 0) + 1
            stats["profiles_correct"] = stats.get("profiles_correct", 0) + int(correct)
            stats["cards_expected"] = stats.get("cards_expected", 0) + len(expected_cards)
            stats["cards_returned"] = stats.get("cards_returned", 0) + len(returned)
            stats["cards_correct_of_returned"] = stats.get("cards_correct_of_returned", 0) + sum(
                actual_map[k] == expected_map[k] for k in returned
            )
            stats["name_recoveries"] = stats.get("name_recoveries", 0) + getattr(recommender, "last_name_recoveries", 0)
        if collect is not None:
            collect.append(
                {
                    "variant": record.variant,
                    "difficulty": record.difficulty,
                    "profile": asdict(record.profile),
                    "correct": correct,
                    "expected": {f"{d.bank}|{d.card}": d.recommended for d in record.expected},
                    "actual": {f"{d.bank}|{d.card}": d.recommended for d in decisions},
                    "reasons": {f"{d.bank}|{d.card}": d.reason for d in decisions},
                    "cited": {f"{d.bank}|{d.card}": d.cited_chunk_id for d in decisions},
                    "requirements": {
                        f"{r.bank}|{r.card}": r.as_dict() for r in getattr(recommender, "last_requirements", [])
                    },
                }
            )
    return results


def mcnemar_report(grounded_correct: list[bool], ungrounded_correct: list[bool]) -> dict[str, Any]:
    """Paired McNemar test comparing grounded vs ungrounded correctness."""
    b = sum(1 for g, u in zip(grounded_correct, ungrounded_correct, strict=True) if g and not u)  # grounded fixes
    c = sum(1 for g, u in zip(grounded_correct, ungrounded_correct, strict=True) if not g and u)  # grounded breaks
    both_correct = sum(1 for g, u in zip(grounded_correct, ungrounded_correct, strict=True) if g and u)
    both_wrong = sum(1 for g, u in zip(grounded_correct, ungrounded_correct, strict=True) if not g and not u)

    table = [[both_correct, b], [c, both_wrong]]
    discordant = b + c
    result = mcnemar(table, exact=discordant < 25)

    grounded_acc = sum(grounded_correct) / len(grounded_correct)
    ungrounded_acc = sum(ungrounded_correct) / len(ungrounded_correct)
    grounded_error = 1 - grounded_acc
    ungrounded_error = 1 - ungrounded_acc
    relative_error_reduction = (
        (ungrounded_error - grounded_error) / ungrounded_error if ungrounded_error > 0 else 0.0
    )

    verdict = (
        "HYPOTHESIS SUPPORTED"
        if (result.pvalue < ALPHA and grounded_acc > ungrounded_acc)
        else "HYPOTHESIS NOT SUPPORTED"
    )

    n = len(grounded_correct)
    g_lo, g_hi = wilson_interval(sum(grounded_correct), n)
    u_lo, u_hi = wilson_interval(sum(ungrounded_correct), n)
    return {
        "grounded_accuracy": grounded_acc,
        "grounded_accuracy_ci95": [round(g_lo, 3), round(g_hi, 3)],
        "ungrounded_accuracy": ungrounded_acc,
        "ungrounded_accuracy_ci95": [round(u_lo, 3), round(u_hi, 3)],
        "b_grounded_fixes": b,
        "c_grounded_breaks": c,
        "discordant_pairs": discordant,
        "mcnemar_exact_binomial": discordant < 25,
        "mcnemar_statistic": float(result.statistic),
        "mcnemar_pvalue": float(result.pvalue),
        "accuracy_difference": grounded_acc - ungrounded_acc,
        "relative_error_reduction_vs_ungrounded": relative_error_reduction,
        "relative_error_reduction_note": (
            "equals grounded accuracy because ungrounded accuracy is 0; report the accuracy "
            "difference and McNemar b/c instead" if ungrounded_acc == 0 else ""
        ),
        "meets_stretch_target": grounded_acc >= STRETCH_TARGET_ACCURACY,
        "verdict": verdict,
    }


def make_llm_clients(use_stub: bool) -> tuple[LLMClient, LLMClient]:
    """Return (grounded_client, ungrounded_client), stub or real depending on `use_stub`."""
    if use_stub:
        return StubLLM(), UngroundedStubLLM()
    return LocalLLMClient(settings.llm_model_name), UngroundedLLMClient(settings.llm_model_name)


def a1_grounded_vs_ungrounded(
    records: list[GoldRecord],
    top_k: int,
    use_stub: bool,
    predictions: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """A1: grounded (RAG) vs ungrounded (B1, same cards and profile, no rules)."""
    retriever = build_retriever(top_k)
    grounded_llm, ungrounded_llm = make_llm_clients(use_stub)
    prompt_builder = GroundedPromptBuilder()

    grounded_recommender = RAGRecommender(retriever, grounded_llm, prompt_builder)
    ungrounded_recommender = RAGRecommender(retriever, ungrounded_llm, prompt_builder)

    g_stats: dict[str, int] = {}
    u_stats: dict[str, int] = {}
    g_rows: list[dict[str, Any]] = []
    u_rows: list[dict[str, Any]] = []
    grounded_correct = run_arm(grounded_recommender, records, g_stats, g_rows)
    ungrounded_correct = run_arm(ungrounded_recommender, records, u_stats, u_rows)
    if predictions is not None:
        predictions["A1_grounded"] = g_rows
        predictions["A1_ungrounded"] = u_rows

    report = mcnemar_report(grounded_correct, ungrounded_correct)
    report["n_profiles"] = len(records)
    report["decision_mode"] = grounded_recommender.decision_mode
    for name, st in (("grounded", g_stats), ("ungrounded", u_stats)):
        report[f"{name}_cards_returned_rate"] = st["cards_returned"] / st["cards_expected"] if st.get("cards_expected") else 0.0
        # Card-level accuracy on the cards the arm actually answered: separates
        # "wrong knowledge" from "no parseable answer" (format failure).
        report[f"{name}_card_accuracy_on_returned"] = (
            st["cards_correct_of_returned"] / st["cards_returned"] if st.get("cards_returned") else 0.0
        )
        report[f"{name}_name_recoveries"] = st.get("name_recoveries", 0)
    return report


def a2_top_k_sweep(records: list[GoldRecord], use_stub: bool, ks: list[int] | None = None) -> dict[int, float]:
    """A2 (STRONGEST NOVEL CONTRIBUTION): top-k sweep over rules kept per card.

    Each card has only 3-4 rule chunks, so every k >= 4 retrieves exactly the
    same context; the informative range is k = 1..4 (k=4 is the default).
    Reports grounded accuracy at each k, characterising the retrieval-depth
    / accuracy trade-off.
    """
    ks = ks or [1, 2, 3, 4]
    grounded_llm, _ = make_llm_clients(use_stub)
    prompt_builder = GroundedPromptBuilder()

    results: dict[int, float] = {}
    for k in ks:
        retriever = build_retriever(k)
        recommender = RAGRecommender(retriever, grounded_llm, prompt_builder)
        correct = run_arm(recommender, records)
        results[k] = sum(correct) / len(correct)
    return results


def a3_prompt_pattern_on_off(records: list[GoldRecord], top_k: int, use_stub: bool) -> dict[str, float]:
    """A3: recommend-explain-cite prompt pattern on vs a bare minimal prompt."""
    retriever = build_retriever(top_k)
    grounded_llm, _ = make_llm_clients(use_stub)

    on_recommender = RAGRecommender(retriever, grounded_llm, GroundedPromptBuilder())
    on_correct = run_arm(on_recommender, records)

    class MinimalPromptBuilder(GroundedPromptBuilder):
        """Prompt variant with the anti-hallucination instructions removed."""

        def build(self, profile: UserProfile, retrieved: list[dict[str, Any]]) -> str:
            base = super().build(profile, retrieved)
            stripped = base.replace(ANTI_HALLUCINATION_CLAUSE, "")
            assert stripped != base, "A3: anti-hallucination clause not found in the prompt"
            return stripped

    off_recommender = RAGRecommender(retriever, grounded_llm, MinimalPromptBuilder())
    off_correct = run_arm(off_recommender, records)

    return {
        "prompt_pattern_on_accuracy": sum(on_correct) / len(on_correct),
        "prompt_pattern_off_accuracy": sum(off_correct) / len(off_correct),
    }


DEFAULT_A4_MODELS = ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]


def a4_model_selection(
    records: list[GoldRecord], top_k: int, use_stub: bool, models: list[str] | None = None
) -> dict[str, Any]:
    """A4: model selection across LLM sizes/families (set with --a4-models).

    Requires a GPU-backed LocalLLMClient per model; under the GPU-free StubLLM
    every arm would collapse onto the same stub, so it is reported as not run
    instead of fabricating a comparison. The default compares two ungated
    Qwen2.5 sizes; pass e.g. Mistral/Llama names (gated, need HF_TOKEN) to
    reproduce the original comparison.
    """
    if use_stub:
        return {"note": "A4 skipped under StubLLM: no distinct local model weights are loaded without a GPU."}

    retriever = build_retriever(top_k)
    prompt_builder = GroundedPromptBuilder()
    results: dict[str, float] = {}
    for model_name in models or DEFAULT_A4_MODELS:
        llm = LocalLLMClient(model_name)
        recommender = RAGRecommender(retriever, llm, prompt_builder)
        correct = run_arm(recommender, records)
        results[model_name] = sum(correct) / len(correct)
    return results


def _build_coarse_index(index_path: Path, chunks_path: Path) -> None:
    """Build a coarse-grained (one merged chunk per card) index, for A5."""
    import faiss
    import numpy as np
    import pandas as pd

    from src.pipeline import RuleChunker

    chunker = RuleChunker()
    fine_chunks = chunker.chunk_directory(settings.rule_docs_dir)

    merged: dict[tuple[str, str], list[str]] = {}
    for c in fine_chunks:
        merged.setdefault((c.bank, c.card), []).append(c.text)

    rows = [
        {"chunk_id": f"coarse#{i}", "bank": bank, "card": card, "text": "\n\n".join(texts)}
        for i, ((bank, card), texts) in enumerate(merged.items())
    ]
    df = pd.DataFrame(rows)

    embedder = Embedder(settings.model_name)
    embeddings = embedder.encode(df["text"].tolist())

    base_index = faiss.IndexFlatIP(embeddings.shape[1])
    index = faiss.IndexIDMap(base_index)
    index.add_with_ids(embeddings, np.arange(len(df), dtype=np.int64))

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    df.to_parquet(chunks_path, index=False)


def a5_chunking_granularity(records: list[GoldRecord], use_stub: bool, top_k: int) -> dict[str, float]:
    """A5 (STRONGEST NOVEL CONTRIBUTION): 1-rule-per-chunk vs multi-rule chunks.

    Compares answer accuracy for the default one-atomic-rule-per-chunk
    granularity against coarser chunks that bundle all of a card's rules
    together, quantifying whether finer-grained chunking helps.
    """
    grounded_llm, _ = make_llm_clients(use_stub)
    prompt_builder = GroundedPromptBuilder()

    fine_retriever = build_retriever(top_k)
    fine_recommender = RAGRecommender(fine_retriever, grounded_llm, prompt_builder)
    fine_correct = run_arm(fine_recommender, records)

    coarse_chunks_path = settings.artifacts_dir / "chunks_coarse.parquet"
    coarse_index_path = settings.artifacts_dir / "faiss_coarse.index"
    _build_coarse_index(coarse_index_path, coarse_chunks_path)
    embedder = Embedder(settings.model_name)
    coarse_store = VectorStore(coarse_index_path, coarse_chunks_path).load()
    coarse_retriever = Retriever(embedder, coarse_store, top_k=max(1, top_k // 2))
    coarse_recommender = RAGRecommender(coarse_retriever, grounded_llm, prompt_builder)
    coarse_correct = run_arm(coarse_recommender, records)

    return {
        "fine_grained_1_rule_per_chunk_accuracy": sum(fine_correct) / len(fine_correct),
        "coarse_multi_rule_per_chunk_accuracy": sum(coarse_correct) / len(coarse_correct),
    }


class _FreeTextPromptBuilder(GroundedPromptBuilder):
    """Grounded prompt whose APPLICANT PROFILE block is the applicant's own wording.

    The model must extract income / age / CIBIL / employment / relationship from
    free text (e.g. '28.0 lakh rupees', 'working a regular job'), which is what
    makes the normal-vs-reworded comparison meaningful. `text` is set per record.
    """

    def __init__(self) -> None:
        self.text = ""

    def build(self, profile: UserProfile, retrieved: list[dict[str, Any]]) -> str:
        base = super().build(profile, retrieved)
        marker = "### APPLICANT PROFILE\n"
        start = base.index(marker) + len(marker)
        end = base.index("\n\n### OUTPUT INSTRUCTIONS")
        return base[:start] + (self.text or base[start:end]) + base[end:]


def a6_normal_vs_reworded(records: list[GoldRecord], top_k: int, use_stub: bool) -> dict[str, float]:
    """A6: normal-phrasing vs reworded-phrasing profile accuracy.

    With a real LLM the applicant is described to the model in free text
    (`profile_text` from the gold set), so phrasing really changes the input.
    The GPU-free StubLLM can only parse the structured block, so under
    `--use-stub-llm` both variants use the structured profile and the result
    only checks plumbing.
    """
    retriever = build_retriever(top_k)
    grounded_llm, _ = make_llm_clients(use_stub)
    builder = GroundedPromptBuilder() if use_stub else _FreeTextPromptBuilder()
    recommender = RAGRecommender(retriever, grounded_llm, builder)

    def run_variant(variant: str) -> list[bool]:
        correct: list[bool] = []
        for record in _progress([r for r in records if r.variant == variant]):
            if isinstance(builder, _FreeTextPromptBuilder):
                builder.text = record.profile_text
            decisions, _ = recommender.recommend(record.profile)
            correct.append(profile_correct(decisions, record.expected))
        return correct

    normal_correct = run_variant("normal")
    reworded_correct = run_variant("reworded")
    return {
        "normal_accuracy": sum(normal_correct) / len(normal_correct) if normal_correct else float("nan"),
        "reworded_accuracy": sum(reworded_correct) / len(reworded_correct) if reworded_correct else float("nan"),
        "normal_n": len(normal_correct),
        "reworded_n": len(reworded_correct),
    }


def a6_balanced_sample(all_records: list[GoldRecord], per_variant: int) -> list[GoldRecord]:
    """Evenly spaced, equally sized samples of 'normal' and 'reworded' profiles.

    The gold set has only 22 reworded profiles out of 120, so a plain evenly
    spaced subsample contains almost none of them and A6 would be meaningless.
    `per_variant <= 0` means "as many as there are reworded profiles".
    """
    reworded = [r for r in all_records if r.variant == "reworded"]
    normal = [r for r in all_records if r.variant == "normal"]
    n = len(reworded) if per_variant <= 0 else min(per_variant, len(reworded))

    def pick(items: list[GoldRecord]) -> list[GoldRecord]:
        if n <= 0 or n >= len(items):
            return items
        step = len(items) / n
        return [items[int(k * step)] for k in range(n)]

    return pick(normal) + pick(reworded)


def _truth_rows() -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (r.bank, r.card): {
            "min_income": r.min_income,
            "min_age": r.min_age,
            "min_cibil": r.min_cibil,
            "employment": r.employment,
            "relationship_required": r.relationship_required,
        }
        for r in EligibilityOracle(settings.eligibility_table_path).rules
    }


def extraction_accuracy(recommender: RAGRecommender, records: list[GoldRecord]) -> dict[str, Any]:
    """Score the requirements the LLM extracted against the eligibility table.

    Only meaningful in extract_compare mode. The table is used here for SCORING
    only; at inference the decisions use whatever the LLM extracted. Because the
    extraction prompt holds no profile, each card is normally read once and the
    result reused (cache), so the unit of analysis is (card, field): 8 x 5 = 40.
    """
    truth = _truth_rows()
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        recommender.recommend(record.profile)
        for req in recommender.last_requirements:
            seen.setdefault((req.bank, req.card), req.as_dict())
    per_field = {f: 0 for f in EXTRACTION_FIELDS}
    per_card: dict[str, dict[str, bool]] = {}
    for key, extracted in seen.items():
        if key not in truth:
            continue
        matches = extraction_field_matches(extracted, truth[key])
        per_card[key[1]] = matches
        for f, ok in matches.items():
            per_field[f] += int(ok)
    n_cards = len(per_card)
    total = sum(per_field.values())
    return {
        "cards_extracted": n_cards,
        "cards_expected": len({k for r in records for k in ((d.bank, d.card) for d in r.expected)}),
        "field_accuracy": total / (n_cards * len(EXTRACTION_FIELDS)) if n_cards else 0.0,
        "per_field_accuracy": {f: v / n_cards if n_cards else 0.0 for f, v in per_field.items()},
        "per_card": per_card,
        "extracted": {f"{b}|{c}": v for (b, c), v in seen.items()},
    }


def a7_decision_modes(
    records: list[GoldRecord],
    top_k: int,
    use_stub: bool,
    predictions: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """A7: LLM-decides (llm) vs LLM-extracts + code-compares (extract_compare).

    Same retriever, same LLM weights, same profiles. Also runs the ungrounded
    control in extract_compare mode, whose extraction accuracy is a direct
    measure of how often the model invents thresholds without the documents.
    """
    retriever = build_retriever(top_k)
    grounded_llm, ungrounded_llm = make_llm_clients(use_stub)
    llm_mode = RAGRecommender(retriever, grounded_llm, GroundedPromptBuilder(), decision_mode="llm")
    ec_mode = RAGRecommender(retriever, grounded_llm, GroundedPromptBuilder(), decision_mode="extract_compare")
    ec_ungrounded = RAGRecommender(retriever, ungrounded_llm, GroundedPromptBuilder(), decision_mode="extract_compare")

    s_llm: dict[str, int] = {}
    s_ec: dict[str, int] = {}
    s_ecu: dict[str, int] = {}
    rows_ec: list[dict[str, Any]] = []
    rows_ecu: list[dict[str, Any]] = []
    llm_correct = run_arm(llm_mode, records, s_llm)
    ec_correct = run_arm(ec_mode, records, s_ec, rows_ec)
    ecu_correct = run_arm(ec_ungrounded, records, s_ecu, rows_ecu)
    if predictions is not None:
        predictions["A7_extract_compare_grounded"] = rows_ec
        predictions["A7_extract_compare_ungrounded"] = rows_ecu

    def arm(correct: list[bool], st: dict[str, int]) -> dict[str, Any]:
        lo, hi = wilson_interval(sum(correct), len(correct))
        return {
            "profile_accuracy": sum(correct) / len(correct) if correct else 0.0,
            "profile_accuracy_ci95": [round(lo, 3), round(hi, 3)],
            "cards_returned_rate": st["cards_returned"] / st["cards_expected"] if st.get("cards_expected") else 0.0,
            "card_accuracy_on_returned": st["cards_correct_of_returned"] / st["cards_returned"] if st.get("cards_returned") else 0.0,
        }

    return {
        "llm_decides": arm(llm_correct, s_llm),
        "extract_compare": arm(ec_correct, s_ec),
        "extract_compare_ungrounded": arm(ecu_correct, s_ecu),
        "llm_vs_extract_compare_mcnemar": mcnemar_report(ec_correct, llm_correct),
        "extract_compare_grounded_vs_ungrounded_mcnemar": mcnemar_report(ec_correct, ecu_correct),
        "extraction_grounded": extraction_accuracy(ec_mode, records),
        "extraction_ungrounded": extraction_accuracy(ec_ungrounded, records),
        "n_profiles": len(records),
    }


def rq1_classification_metrics(records: list[GoldRecord], top_k: int, use_stub: bool) -> dict[str, Any]:
    """RQ1: per-card precision/recall/F1/accuracy for the grounded arm vs the oracle.

    A card with no returned decision counts as "not recommended" here, and is
    also reported separately as `cards_returned_rate` so parse failures are visible.
    """
    retriever = build_retriever(top_k)
    grounded_llm, _ = make_llm_clients(use_stub)
    recommender = RAGRecommender(retriever, grounded_llm, GroundedPromptBuilder())

    y_true: list[bool] = []
    y_pred: list[bool] = []
    per_card: dict[str, dict[str, int]] = {}
    returned = 0
    for record in records:
        decisions, _ = recommender.recommend(record.profile)
        actual_map = _decisions_to_map(decisions)
        for expected in record.expected:
            key = (expected.bank, expected.card)
            returned += key in actual_map
            pred = actual_map.get(key, False)
            y_true.append(expected.recommended)
            y_pred.append(pred)
            cell = per_card.setdefault(expected.card, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
            cell["tp" if expected.recommended and pred else "fn" if expected.recommended else "fp" if pred else "tn"] += 1
    metrics: dict[str, Any] = dict(classification_metrics(y_true, y_pred))
    metrics["cards_returned_rate"] = returned / len(y_true) if y_true else 0.0
    metrics["positive_rate_gold"] = sum(y_true) / len(y_true) if y_true else 0.0
    metrics["positive_rate_predicted"] = sum(y_pred) / len(y_pred) if y_pred else 0.0
    metrics["majority_class_accuracy"] = 1 - metrics["positive_rate_gold"]
    metrics["per_card"] = per_card
    return metrics


def rq2_context_precision_recall(records: list[GoldRecord], top_k: int) -> dict[str, float]:
    """RQ2: mean context precision/recall vs a proxy relevance set.

    The proxy "relevant" set for a profile is every chunk describing a
    card offered by one of the requested banks -- i.e. whether retrieval
    successfully narrowed down to the banks the applicant actually asked
    about.
    """
    retriever = build_retriever(top_k)
    chunks_df = retriever.vector_store.chunks
    precisions: list[float] = []
    recalls: list[float] = []
    for record in records:
        banks_norm = {b.strip().lower() for b in record.profile.banks}
        relevant_ids = chunks_df[chunks_df["bank"].str.lower().isin(banks_norm)]["chunk_id"].tolist()
        retrieved = retriever.retrieve_for_profile(record.profile)
        retrieved_ids = [r["chunk_id"] for r in retrieved]
        metrics = context_precision_recall(retrieved_ids, relevant_ids)
        precisions.append(metrics["context_precision"])
        recalls.append(metrics["context_recall"])
    result: dict[str, float] = {
        "mean_context_precision": sum(precisions) / len(precisions),
        "mean_context_recall": sum(recalls) / len(recalls),
        "note_proxy": "saturated by construction: retrieve_for_profile filters chunks by card metadata",
    }
    result.update(rq2_dense_retrieval(records, retriever))
    return result


def rq2_dense_retrieval(records: list[GoldRecord], retriever: Retriever) -> dict[str, Any]:
    """RQ2 without the metadata filter: does Sentence-BERT + FAISS alone find a card's rules?

    For every (profile, card) the query is the same one the system uses
    (profile text + card name). Over the WHOLE index (all cards, both banks),
    the top-k is compared with that card's own rule chunks:
    precision@k = share of the top-k that belong to the card;
    recall@k = share of the card's chunks that appear in the top-k.
    This is the honest measure of embedding quality; the production retriever
    then guarantees completeness by filtering on card metadata.
    """
    chunks_df = retriever.vector_store.chunks
    total = len(chunks_df)
    out: dict[str, Any] = {}
    for k in sorted({1, 2, 3, 4, retriever.top_k}):
        precisions: list[float] = []
        recalls: list[float] = []
        for record in records:
            banks = {b.casefold() for b in record.profile.banks}
            cards = chunks_df[chunks_df["bank"].str.casefold().isin(banks)][["bank", "card"]].drop_duplicates()
            queries = [f"{record.profile.to_query()} Card: {b} {c}." for b, c in cards.itertuples(index=False, name=None)]
            vectors = retriever.embedder.encode(queries)
            for (bank, card), vector in zip(cards.itertuples(index=False, name=None), vectors, strict=True):
                own = set(chunks_df[(chunks_df["bank"] == bank) & (chunks_df["card"] == card)]["chunk_id"])
                top = [c["chunk_id"] for c in retriever.vector_store.search(vector, top_k=min(k, total))]
                hits = len(own & set(top))
                precisions.append(hits / len(top) if top else 0.0)
                recalls.append(hits / len(own) if own else 0.0)
        out[f"dense_precision_at_{k}"] = sum(precisions) / len(precisions) if precisions else 0.0
        out[f"dense_recall_at_{k}"] = sum(recalls) / len(recalls) if recalls else 0.0
    return out


def rq4_traceability_metrics(records: list[GoldRecord], top_k: int, use_stub: bool) -> dict[str, float]:
    """Measure profile-level label consistency and per-decision citation validity."""
    if not records:
        return {"label_consistency_rate": 0.0, "citation_validity_rate": 0.0}

    retriever = build_retriever(top_k)
    grounded_llm, _ = make_llm_clients(use_stub)
    recommender = RAGRecommender(retriever, grounded_llm, GroundedPromptBuilder())
    consistent_profiles = 0
    valid_citations = 0
    decision_count = 0
    declines = 0
    declines_governed = 0
    correct_declines = 0
    correct_declines_governed = 0
    chunk_text = dict(zip(retriever.vector_store.chunks["chunk_id"], retriever.vector_store.chunks["text"], strict=True))
    rules = {(r.bank, r.card): r for r in EligibilityOracle(settings.eligibility_table_path).rules}

    for record in records:
        decisions, _ = recommender.recommend(record.profile)
        retrieved = retriever.retrieve_for_profile(record.profile)
        consistent_profiles += profile_correct(decisions, record.expected)
        decision_count += len(decisions)
        valid_citations += sum(citation_is_valid(decision, retrieved) for decision in decisions)
        expected_map = _decisions_to_map(record.expected)
        for d in decisions:
            if d.recommended or (d.bank, d.card) not in rules:
                continue
            declines += 1
            failing = failing_criteria(record.profile, rules[(d.bank, d.card)])
            governed = citation_is_valid(d, retrieved) and bool(
                chunk_criteria(chunk_text.get(d.cited_chunk_id or "", "")) & failing
            )
            declines_governed += governed
            if expected_map.get((d.bank, d.card)) is False:
                correct_declines += 1
                correct_declines_governed += governed

    return {
        "label_consistency_rate": consistent_profiles / len(records),
        "citation_validity_rate": valid_citations / decision_count if decision_count else 0.0,
        # Reasoning-level check (first-marker point F5): for a declined card, does the
        # cited chunk state a criterion the applicant ACTUALLY fails?
        "decline_citation_governs_rate": declines_governed / declines if declines else 0.0,
        "correct_decline_citation_governs_rate": (
            correct_declines_governed / correct_declines if correct_declines else 0.0
        ),
        "n_declines": declines,
        "n_correct_declines": correct_declines,
    }


_CRITERION_HEADING_WORDS: dict[str, tuple[str, ...]] = {
    "income": ("income",),
    "age": ("age",),
    "cibil": ("credit score", "credit history", "cibil", "bureau score"),
    "employment": ("employment",),
    "relationship": ("relationship",),
}


def chunk_criteria(text: str) -> set[str]:
    """Criteria a rule chunk states, read from its heading (e.g. 'Age and Credit Score')."""
    heading = text.strip().splitlines()[0].lower() if text.strip() else ""
    return {c for c, words in _CRITERION_HEADING_WORDS.items() if any(w in heading for w in words)}


def failing_criteria(profile: UserProfile, rule: Any) -> set[str]:
    """Criteria the applicant fails according to the eligibility table (scoring only)."""
    failing: set[str] = set()
    if profile.annual_income < rule.min_income:
        failing.add("income")
    if profile.age < rule.min_age:
        failing.add("age")
    if profile.cibil < rule.min_cibil:
        failing.add("cibil")
    if "any" not in rule.employment and profile.employment.lower() not in rule.employment:
        failing.add("employment")
    if rule.relationship_required and not profile.existing_relationship:
        failing.add("relationship")
    return failing


def print_rq_summary(
    a1: dict[str, Any], rq1: dict[str, Any], rq2: dict[str, float], rq4: dict[str, float]
) -> None:
    """Print each research question label next to its mapped metric."""
    print("\n=== Research Question summary ===")
    print(
        f"RQ1 (gold-set classification vs the eligibility oracle; RAGAS faithfulness is NOT computed here): "
        f"profile accuracy = {a1['grounded_accuracy']:.3f}; per-card precision={rq1['precision']:.3f} "
        f"recall={rq1['recall']:.3f} f1={rq1['f1']:.3f} accuracy={rq1['accuracy']:.3f} "
        f"(majority-class baseline accuracy={rq1['majority_class_accuracy']:.3f}; "
        f"cards returned={rq1['cards_returned_rate']:.3f})"
    )
    print(
        f"RQ2 (context precision/recall, proxy = all cards of the requested banks): "
        f"mean_context_precision={rq2['mean_context_precision']:.3f} "
        f"mean_context_recall={rq2['mean_context_recall']:.3f}"
    )
    print(
        f"RQ3 (hallucination reduction via McNemar): p={a1['mcnemar_pvalue']:.4f}; "
        f"relative error reduction={a1['relative_error_reduction_vs_ungrounded']:.1%}; verdict={a1['verdict']}"
    )
    print(
        f"RQ4 (explanation correctness): label_consistency_rate={rq4['label_consistency_rate']:.3f}; "
        f"citation_validity_rate={rq4['citation_validity_rate']:.3f} "
        "(cited chunk retrieved for the query and matching the decision's bank/card); "
        f"decline_citation_governs_rate={rq4.get('decline_citation_governs_rate', 0.0):.3f} "
        f"(correct declines only: {rq4.get('correct_decline_citation_governs_rate', 0.0):.3f}; "
        "cited chunk states a criterion the applicant actually fails)"
    )
    if "dense_recall_at_4" in rq2:
        print(
            "RQ2 (dense retrieval WITHOUT the card filter, over the whole index): "
            + " ".join(f"{k}={v:.3f}" for k, v in rq2.items() if k.startswith("dense_"))
        )


def select_records(all_records: list[GoldRecord], split: str, limit: int) -> list[GoldRecord]:
    """Pick the evaluation profiles.

    split 'test' = every 3rd profile (40 of 120); 'dev' = the other 80, so any
    prompt/design tuning can be done on dev and reported on test; 'all' = 120.
    `limit` then takes an evenly spaced subsample of the chosen pool.
    """
    if split == "test":
        pool = [r for i, r in enumerate(all_records) if i % 3 == 0]
    elif split == "dev":
        pool = [r for i, r in enumerate(all_records) if i % 3 != 0]
    elif split in {"dev40", "holdout"}:
        # The 14 Oct dev confirmation ran `--split dev --limit 40`, i.e. the EVEN
        # positions of the 80-profile dev pool ("dev40"). The ODD positions were
        # never evaluated in A1/RQ1-RQ4 ("holdout", 40 profiles). 12 of them did
        # appear in the A6 balanced sample (disclose this in the dissertation).
        dev_pool = [r for i, r in enumerate(all_records) if i % 3 != 0]
        pool = dev_pool[0::2] if split == "dev40" else dev_pool[1::2]
    else:
        pool = list(all_records)
    if limit and 0 < limit < len(pool):
        step = len(pool) / limit
        pool = [pool[int(i * step)] for i in range(limit)]
    return pool


class ResultsStore:
    """Accumulates every number into <out_dir>/results.json (merged across runs)."""

    def __init__(self, out_dir: Path) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "results.json"
        self.data: dict[str, Any] = {"runs": [], "arms": {}}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

    def put(self, arm: str, value: Any, meta: dict[str, Any]) -> None:
        self.data.setdefault("arms", {})[arm] = {"meta": meta, "result": value}
        self.path.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")

    def write_predictions(self, name: str, rows: list[dict[str, Any]]) -> None:
        with (self.dir / f"predictions_{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")


def main() -> None:
    """CLI entry point: `python -m evaluation.run_ablation`."""
    parser = argparse.ArgumentParser(description="Run the A1-A6 ablation matrix.")
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--use-stub-llm", action="store_true", default=settings.use_stub_llm)
    parser.add_argument("--gold-set", type=Path, default=settings.gold_test_set_path)
    parser.add_argument(
        "--joint-prompt",
        action="store_true",
        help="Use ONE prompt for all cards (original behaviour) instead of one prompt per card",
    )
    parser.add_argument(
        "--split",
        choices=["all", "test", "dev", "dev40", "holdout"],
        default="all",
        help="Which gold profiles to use (holdout = 40 dev profiles never evaluated in A1/RQ1-RQ4)",
    )
    parser.add_argument(
        "--decision-mode",
        choices=list(DECISION_MODES),
        default=None,
        help="llm = LLM decides (14 Oct configuration); extract_compare = LLM extracts requirements, code compares",
    )
    parser.add_argument("--limit", type=int, default=0, help="Evenly spaced subsample of N profiles (0 = all)")
    parser.add_argument("--out-dir", type=Path, default=Path("results"), help="Where results.json/predictions go")
    parser.add_argument("--a4-models", nargs="*", default=None, help="Model names for A4 (needs a GPU)")
    parser.add_argument(
        "--no-core",
        action="store_true",
        help="Skip A1/RQ1/RQ2/RQ4 (use when running extra arms on a different subset, so the main results are not overwritten)",
    )
    parser.add_argument(
        "--arms", nargs="*", default=["A1", "A2", "A3", "A5", "A6"], help="Which arms to run (A4 needs a GPU)"
    )
    args = parser.parse_args()

    if args.joint_prompt:
        os.environ["RAG_PER_CARD"] = "0"
    if args.decision_mode:
        os.environ["RAG_DECISION_MODE"] = args.decision_mode
    all_records = load_gold_set(args.gold_set)
    records = select_records(all_records, args.split, args.limit)
    store = ResultsStore(args.out_dir)

    # Integrity guard: a silent fallback to the hashing embedder would invalidate retrieval.
    probe = build_retriever(args.top_k)
    probe.embedder.encode(["probe"])
    if probe.embedder.using_fallback and not args.use_stub_llm:
        raise RuntimeError("Sentence-BERT weights could not be loaded; refusing to evaluate with the hashing fallback.")

    meta = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "llm": "StubLLM" if args.use_stub_llm else settings.llm_model_name,
        "embedding_model": settings.model_name,
        "per_card_prompting": not args.joint_prompt,
        "decision_mode": os.environ.get("RAG_DECISION_MODE", "llm"),
        "top_k": args.top_k,
        "split": args.split,
        "limit": args.limit,
        "n_profiles": len(records),
        "gold_set": str(args.gold_set),
    }
    print(f"Loaded {len(records)} gold profiles ({args.split} split) from {args.gold_set}")
    print(
        "CAVEAT: figures below are an OPTIMISTIC UPPER BOUND computed on a synthetic "
        "gold set against curated, representative (not live) documents for exactly "
        "two banks (SBI, HDFC); there is no runtime validation gate in the live "
        "system -- these are the only faithfulness checks that ever happen."
    )
    if args.use_stub_llm:
        print("NOTE: StubLLM reads the eligibility oracle, so stub numbers only test plumbing, not model quality.")

    def timed(name: str, fn: Any) -> Any:
        start = time.perf_counter()
        value = fn()
        meta_arm = {**meta, "seconds": round(time.perf_counter() - start, 1)}
        store.put(name, value, meta_arm)
        return value

    if not args.no_core:
        predictions: dict[str, list[dict[str, Any]]] = {}
        a1 = timed("A1", lambda: a1_grounded_vs_ungrounded(records, args.top_k, args.use_stub_llm, predictions))
        for name, rows in predictions.items():
            store.write_predictions(name, rows)
        print("\n=== A1: grounded vs ungrounded (B1) ===")
        for k, v in a1.items():
            print(f"  {k}: {v}")

        rq1 = timed("RQ1", lambda: rq1_classification_metrics(records, args.top_k, args.use_stub_llm))
        rq2 = timed("RQ2", lambda: rq2_context_precision_recall(records, args.top_k))
        rq4 = timed("RQ4", lambda: rq4_traceability_metrics(records, args.top_k, args.use_stub_llm))
        print_rq_summary(a1, rq1, rq2, rq4)
        print("  per-card confusion (tp/fp/fn/tn):", {c: tuple(v.values()) for c, v in rq1["per_card"].items()})


    if "A7" in args.arms:
        print("\n=== A7: LLM decides vs LLM extracts + code compares ===")
        a7_predictions: dict[str, list[dict[str, Any]]] = {}
        a7 = timed("A7", lambda: a7_decision_modes(records, args.top_k, args.use_stub_llm, a7_predictions))
        for name, rows in a7_predictions.items():
            store.write_predictions(name, rows)
        for name in ("llm_decides", "extract_compare", "extract_compare_ungrounded"):
            print(f"  {name}: {a7[name]}")
        for name in ("llm_vs_extract_compare_mcnemar", "extract_compare_grounded_vs_ungrounded_mcnemar"):
            r = a7[name]
            print(f"  {name}: b={r['b_grounded_fixes']} c={r['c_grounded_breaks']} p={r['mcnemar_pvalue']:.4g}")
        for name in ("extraction_grounded", "extraction_ungrounded"):
            r = a7[name]
            print(f"  {name}: field_accuracy={r['field_accuracy']:.3f} per_field={r['per_field_accuracy']}")

    if "A2" in args.arms:
        print("\n=== A2: top-k sweep (rules kept per card) ===")
        a2 = timed("A2", lambda: a2_top_k_sweep(records, args.use_stub_llm))
        for top_k_val, acc in a2.items():
            print(f"  top_k={top_k_val}: accuracy={acc:.3f}")

    if "A3" in args.arms:
        print("\n=== A3: prompt pattern on/off ===")
        for k, v in timed("A3", lambda: a3_prompt_pattern_on_off(records, args.top_k, args.use_stub_llm)).items():
            print(f"  {k}: {v:.3f}")

    if "A4" in args.arms:
        print("\n=== A4: model selection ===")
        a4 = timed("A4", lambda: a4_model_selection(records, args.top_k, args.use_stub_llm, args.a4_models))
        for k, v in a4.items():
            print(f"  {k}: {v}")

    if "A5" in args.arms:
        print("\n=== A5: chunking granularity ===")
        for k, v in timed("A5", lambda: a5_chunking_granularity(records, args.use_stub_llm, args.top_k)).items():
            print(f"  {k}: {v:.3f}")

    if "A6" in args.arms:
        print("\n=== A6: normal vs reworded profiles ===")
        a6_records = a6_balanced_sample(all_records, min(22, args.limit // 2) if args.limit else 0)
        print(f"  (balanced sample: {len(a6_records)} profiles drawn from the full gold set)")
        a6 = timed("A6", lambda: a6_normal_vs_reworded(a6_records, args.top_k, args.use_stub_llm))
        for k, v in a6.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    core_arms = [] if args.no_core else ["A1", "RQ1", "RQ2", "RQ4"]
    store.data.setdefault("runs", []).append({**meta, "arms_run": [*core_arms, *args.arms]})
    store.path.write_text(json.dumps(store.data, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to {store.path}")


if __name__ == "__main__":
    main()
