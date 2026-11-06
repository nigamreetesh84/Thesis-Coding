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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from statsmodels.stats.contingency_tables import mcnemar

from config.settings import settings
from evaluation.metrics import classification_metrics, context_precision_recall
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
from src.model import GroundedPromptBuilder, RAGRecommender, Retriever, parse_llm_response

STRETCH_TARGET_ACCURACY = 0.90
ALPHA = 0.05


@dataclass(frozen=True)
class GoldRecord:
    """One deserialized row of data/gold_test_set.jsonl."""

    profile: UserProfile
    expected: list[Decision]
    variant: str
    difficulty: str


def load_gold_set(path: str | Path = settings.gold_test_set_path) -> list[GoldRecord]:
    """Load the gold evaluation set produced by evaluation/build_gold_set.py."""
    records: list[GoldRecord] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            profile = UserProfile(**row["profile"])
            expected = [Decision(**d) for d in row["expected"]]
            records.append(
                GoldRecord(profile=profile, expected=expected, variant=row["variant"], difficulty=row["difficulty"])
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


def run_arm(recommender: RAGRecommender, records: list[GoldRecord]) -> list[bool]:
    """Run one ablation arm over the gold set, returning per-profile correctness."""
    return [profile_correct(recommender.recommend(record.profile)[0], record.expected) for record in records]


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

    return {
        "grounded_accuracy": grounded_acc,
        "ungrounded_accuracy": ungrounded_acc,
        "b_grounded_fixes": b,
        "c_grounded_breaks": c,
        "mcnemar_statistic": float(result.statistic),
        "mcnemar_pvalue": float(result.pvalue),
        "relative_error_reduction_vs_ungrounded": relative_error_reduction,
        "meets_stretch_target": grounded_acc >= STRETCH_TARGET_ACCURACY,
        "verdict": verdict,
    }


def make_llm_clients(use_stub: bool) -> tuple[LLMClient, LLMClient]:
    """Return (grounded_client, ungrounded_client), stub or real depending on `use_stub`."""
    if use_stub:
        return StubLLM(), UngroundedStubLLM()
    return LocalLLMClient(settings.llm_model_name), UngroundedLLMClient(settings.llm_model_name)


def a1_grounded_vs_ungrounded(records: list[GoldRecord], top_k: int, use_stub: bool) -> dict[str, Any]:
    """A1: grounded (RAG) vs ungrounded (B1, parametric-memory-only)."""
    retriever = build_retriever(top_k)
    grounded_llm, ungrounded_llm = make_llm_clients(use_stub)
    prompt_builder = GroundedPromptBuilder()

    grounded_recommender = RAGRecommender(retriever, grounded_llm, prompt_builder)
    ungrounded_recommender = RAGRecommender(retriever, ungrounded_llm, prompt_builder)

    grounded_correct = run_arm(grounded_recommender, records)
    ungrounded_correct = run_arm(ungrounded_recommender, records)
    return mcnemar_report(grounded_correct, ungrounded_correct)


def a2_top_k_sweep(records: list[GoldRecord], use_stub: bool, ks: list[int] | None = None) -> dict[int, float]:
    """A2 (STRONGEST NOVEL CONTRIBUTION): top-k sweep k=2,4,6,8,12.

    Sweeps the number of retrieved chunks and reports grounded accuracy at
    each k, characterising the retrieval-depth/accuracy trade-off.
    """
    ks = ks or [2, 4, 6, 8, 12]
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
            return base.replace(
                "Never invent a threshold, number, or condition that is not explicitly "
                "present in the retrieved rules. For every card you do NOT recommend, give "
                "a specific, principal reason naming the exact criterion that fails. ",
                ""
            )

    off_recommender = RAGRecommender(retriever, grounded_llm, MinimalPromptBuilder())
    off_correct = run_arm(off_recommender, records)

    return {
        "prompt_pattern_on_accuracy": sum(on_correct) / len(on_correct),
        "prompt_pattern_off_accuracy": sum(off_correct) / len(off_correct),
    }


def a4_model_selection(records: list[GoldRecord], top_k: int, use_stub: bool) -> dict[str, Any]:
    """A4: model selection, Mistral-7B-Instruct vs Llama-3.1-8B-Instruct.

    Requires a GPU-backed LocalLLMClient for each model name; when running
    the GPU-free StubLLM path, both arms would collapse onto the same
    stub, so this arm is reported as a not-run placeholder instead of
    fabricating a comparison.
    """
    if use_stub:
        return {"note": "A4 skipped under StubLLM: no distinct local model weights are loaded without a GPU."}

    retriever = build_retriever(top_k)
    prompt_builder = GroundedPromptBuilder()
    results: dict[str, float] = {}
    for model_name in ["mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Meta-Llama-3.1-8B-Instruct"]:
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


def a6_normal_vs_reworded(records: list[GoldRecord], top_k: int, use_stub: bool) -> dict[str, float]:
    """A6: normal-phrasing vs reworded-phrasing profile accuracy."""
    retriever = build_retriever(top_k)
    grounded_llm, _ = make_llm_clients(use_stub)
    recommender = RAGRecommender(retriever, grounded_llm, GroundedPromptBuilder())

    normal = [r for r in records if r.variant == "normal"]
    reworded = [r for r in records if r.variant == "reworded"]

    normal_correct = run_arm(recommender, normal) if normal else []
    reworded_correct = run_arm(recommender, reworded) if reworded else []

    return {
        "normal_accuracy": sum(normal_correct) / len(normal_correct) if normal_correct else 0.0,
        "reworded_accuracy": sum(reworded_correct) / len(reworded_correct) if reworded_correct else 0.0,
    }


def rq1_classification_metrics(records: list[GoldRecord], top_k: int, use_stub: bool) -> dict[str, float]:
    """RQ1: per-card precision/recall/F1/accuracy for the grounded arm vs the oracle."""
    retriever = build_retriever(top_k)
    grounded_llm, _ = make_llm_clients(use_stub)
    recommender = RAGRecommender(retriever, grounded_llm, GroundedPromptBuilder())

    y_true: list[bool] = []
    y_pred: list[bool] = []
    for record in records:
        decisions, _ = recommender.recommend(record.profile)
        actual_map = _decisions_to_map(decisions)
        for expected in record.expected:
            y_true.append(expected.recommended)
            y_pred.append(actual_map.get((expected.bank, expected.card), False))
    return classification_metrics(y_true, y_pred)


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
        retrieved = retriever.retrieve(record.profile.to_query())
        retrieved_ids = [r["chunk_id"] for r in retrieved]
        metrics = context_precision_recall(retrieved_ids, relevant_ids)
        precisions.append(metrics["context_precision"])
        recalls.append(metrics["context_recall"])
    return {
        "mean_context_precision": sum(precisions) / len(precisions),
        "mean_context_recall": sum(recalls) / len(recalls),
    }


def rq4_traceability_metrics(records: list[GoldRecord], top_k: int, use_stub: bool) -> dict[str, float]:
    """Measure profile-level label consistency and per-decision citation validity."""
    if not records:
        return {"label_consistency_rate": 0.0, "citation_validity_rate": 0.0}

    retriever = build_retriever(top_k)
    grounded_llm, _ = make_llm_clients(use_stub)
    prompt_builder = GroundedPromptBuilder()
    consistent_profiles = 0
    valid_citations = 0
    decision_count = 0

    for record in records:
        retrieved = retriever.retrieve_for_profile(record.profile)
        prompt = prompt_builder.build(record.profile, retrieved)
        decisions = parse_llm_response(grounded_llm.generate(prompt))
        consistent_profiles += profile_correct(decisions, record.expected)
        decision_count += len(decisions)
        valid_citations += sum(citation_is_valid(decision, retrieved) for decision in decisions)

    return {
        "label_consistency_rate": consistent_profiles / len(records),
        "citation_validity_rate": valid_citations / decision_count if decision_count else 0.0,
    }


def print_rq_summary(
    a1: dict[str, Any], rq1: dict[str, float], rq2: dict[str, float], rq4: dict[str, float]
) -> None:
    """Print each research question label next to its mapped metric."""
    print("\n=== Research Question summary ===")
    print(
        f"RQ1 (faithfulness via RAGAS + gold-set accuracy): grounded accuracy = "
        f"{a1['grounded_accuracy']:.3f}; per-card precision={rq1['precision']:.3f} "
        f"recall={rq1['recall']:.3f} f1={rq1['f1']:.3f} accuracy={rq1['accuracy']:.3f}"
    )
    print(
        f"RQ2 (context precision/recall): mean_context_precision={rq2['mean_context_precision']:.3f} "
        f"mean_context_recall={rq2['mean_context_recall']:.3f}"
    )
    print(
        f"RQ3 (hallucination reduction via McNemar): p={a1['mcnemar_pvalue']:.4f}; "
        f"relative error reduction={a1['relative_error_reduction_vs_ungrounded']:.1%}; verdict={a1['verdict']}"
    )
    print(
        f"RQ4 (explanation correctness): label_consistency_rate={rq4['label_consistency_rate']:.3f}; "
        f"citation_validity_rate={rq4['citation_validity_rate']:.3f} "
        "(cited chunk retrieved for the query and matching the decision's bank/card)"
    )


def main() -> None:
    """CLI entry point: `python -m evaluation.run_ablation`."""
    parser = argparse.ArgumentParser(description="Run the A1-A6 ablation matrix.")
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--use-stub-llm", action="store_true", default=settings.use_stub_llm)
    parser.add_argument("--gold-set", type=Path, default=settings.gold_test_set_path)
    parser.add_argument(
        "--arms", nargs="*", default=["A1", "A2", "A3", "A5", "A6"], help="Which arms to run (A4 requires a GPU)"
    )
    args = parser.parse_args()

    records = load_gold_set(args.gold_set)
    print(f"Loaded {len(records)} gold profiles from {args.gold_set}")
    print(
        "CAVEAT: figures below are an OPTIMISTIC UPPER BOUND computed on a synthetic "
        "gold set against curated, representative (not live) documents for exactly "
        "two banks (SBI, HDFC); there is no runtime validation gate in the live "
        "system -- these are the only faithfulness checks that ever happen."
    )

    a1 = a1_grounded_vs_ungrounded(records, args.top_k, args.use_stub_llm)
    print("\n=== A1: grounded vs ungrounded (B1) ===")
    for k, v in a1.items():
        print(f"  {k}: {v}")

    rq1 = rq1_classification_metrics(records, args.top_k, args.use_stub_llm)
    rq2 = rq2_context_precision_recall(records, args.top_k)
    rq4 = rq4_traceability_metrics(records, args.top_k, args.use_stub_llm)
    print_rq_summary(a1, rq1, rq2, rq4)

    if "A2" in args.arms:
        print("\n=== A2: top-k sweep (STRONGEST NOVEL CONTRIBUTION) ===")
        for top_k_val, acc in a2_top_k_sweep(records, args.use_stub_llm).items():
            print(f"  top_k={top_k_val}: accuracy={acc:.3f}")

    if "A3" in args.arms:
        print("\n=== A3: prompt pattern on/off ===")
        for k, v in a3_prompt_pattern_on_off(records, args.top_k, args.use_stub_llm).items():
            print(f"  {k}: {v:.3f}")

    if "A4" in args.arms:
        print("\n=== A4: model selection ===")
        for k, v in a4_model_selection(records, args.top_k, args.use_stub_llm).items():
            print(f"  {k}: {v}")

    if "A5" in args.arms:
        print("\n=== A5: chunking granularity (STRONGEST NOVEL CONTRIBUTION) ===")
        for k, v in a5_chunking_granularity(records, args.use_stub_llm, args.top_k).items():
            print(f"  {k}: {v:.3f}")

    if "A6" in args.arms:
        print("\n=== A6: normal vs reworded profiles ===")
        for k, v in a6_normal_vs_reworded(records, args.top_k, args.use_stub_llm).items():
            print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()