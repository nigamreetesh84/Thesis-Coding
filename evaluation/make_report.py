"""Turn results/results.json (+ predictions) into a readable RESULTS.md.

Run: `python -m evaluation.make_report --results-dir results`
"""
from __future__ import annotations

import argparse
import math
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _fmt(v: Any) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def _table(rows: list[tuple[str, Any]]) -> str:
    rows = [(k, v) for k, v in rows if v is not None]
    return "| metric | value |\n|---|---|\n" + "\n".join(f"| {k} | {_fmt(v)} |" for k, v in rows) + "\n"


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial test on the discordant pairs, p = 0.5).

    Recomputed here from b and c so results produced by earlier code versions
    (which used the chi-square approximation at >= 25 discordant pairs) are
    reported with the same test as every other arm.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def _difficulty_and_card_breakdown(pred_path: Path) -> str:
    if not pred_path.exists():
        return ""
    rows = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_diff: dict[str, list[bool]] = defaultdict(list)
    cell: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    for r in rows:
        by_diff[r["difficulty"]].append(bool(r["correct"]))
        for card, exp in r["expected"].items():
            act = r["actual"].get(card, False)
            cell[card]["tp" if exp and act else "fn" if exp else "fp" if act else "tn"] += 1
    out = "### Error analysis (grounded arm)\n\n| difficulty | profiles | profile accuracy |\n|---|---|---|\n"
    for d, vals in sorted(by_diff.items()):
        out += f"| {d} | {len(vals)} | {sum(vals) / len(vals):.3f} |\n"
    out += "\n| card | tp | fp | fn | tn |\n|---|---|---|---|---|\n"
    for card, c in sorted(cell.items()):
        out += f"| {card.split('|')[1]} | {c['tp']} | {c['fp']} | {c['fn']} | {c['tn']} |\n"
    return out + "\n"


def build_report(results_dir: Path) -> str:
    data = json.loads((results_dir / "results.json").read_text(encoding="utf-8"))
    arms: dict[str, Any] = data.get("arms", {})

    def res(name: str) -> Any:
        return arms.get(name, {}).get("result")

    def meta(name: str) -> dict[str, Any]:
        return arms.get(name, {}).get("meta", {})

    lines = ["# Results\n"]
    m = meta("A1") or next(iter(arms.values()), {}).get("meta", {})
    if m:
        lines.append(
            f"*LLM:* `{m.get('llm')}` | *embedding:* `{m.get('embedding_model')}` | *per-card prompting:* "
            f"{m.get('per_card_prompting')} | *split:* {m.get('split')} | *profiles (core):* {m.get('n_profiles')} | "
            f"*run:* {m.get('timestamp_utc')}\n"
        )
        if m.get("llm") == "StubLLM":
            lines.append("> **Stub run: these numbers only test plumbing (the stub reads the oracle), not model quality.**\n")

    a1, rq1, rq2, rq4 = res("A1"), res("RQ1"), res("RQ2"), res("RQ4")
    if a1:
        lines.append("## A1 / RQ3: grounded vs ungrounded\n")
        lines.append(_table([
            ("profiles (n)", a1["n_profiles"]),
            ("decision mode", a1.get("decision_mode")),
            ("grounded profile accuracy", a1["grounded_accuracy"]),
            ("grounded 95% CI (Wilson)", a1.get("grounded_accuracy_ci95")),
            ("ungrounded profile accuracy", a1["ungrounded_accuracy"]),
            ("ungrounded 95% CI (Wilson)", a1.get("ungrounded_accuracy_ci95")),
            ("accuracy difference", a1.get("accuracy_difference")),
            ("profiles fixed by grounding (b)", a1["b_grounded_fixes"]),
            ("profiles broken by grounding (c)", a1["c_grounded_breaks"]),
            ("McNemar p-value (exact binomial)", f"{exact_mcnemar_p(a1['b_grounded_fixes'], a1['c_grounded_breaks']):.3g}"),
            ("relative error reduction", a1["relative_error_reduction_vs_ungrounded"]),
            ("grounded cards returned", a1["grounded_cards_returned_rate"]),
            ("ungrounded cards returned", a1["ungrounded_cards_returned_rate"]),
            ("grounded card accuracy (returned cards)", a1.get("grounded_card_accuracy_on_returned")),
            ("ungrounded card accuracy (returned cards)", a1.get("ungrounded_card_accuracy_on_returned")),
            ("ungrounded replies kept despite card-name mismatch", a1.get("ungrounded_name_recoveries")),
            ("grounded decisions flagged needs-review (guard)", a1.get("grounded_needs_review_decisions")),
            ("meets 0.90 stretch target", a1["meets_stretch_target"]),
            ("verdict", a1["verdict"]),
        ]))
        if a1.get("relative_error_reduction_note"):
            lines.append(f"> Note: relative error reduction {a1['relative_error_reduction_note']}.\n")
        if a1["ungrounded_cards_returned_rate"] < 0.95:
            lines.append("> Warning: the ungrounded arm returned fewer than 95% of the cards; part of its error may be output-format failure, not wrong knowledge.\n")
    if rq1:
        lines.append("## RQ1: gold-set classification (RAGAS faithfulness not computed)\n")
        lines.append(_table([(k, rq1[k]) for k in ("precision", "recall", "f1", "accuracy", "majority_class_accuracy",
                                                     "positive_rate_gold", "positive_rate_predicted", "cards_returned_rate")]))
    if rq2:
        lines.append("## RQ2: retrieval (proxy relevance = all cards of requested banks; saturated, see EVALUATION_NOTES.md)\n")
        lines.append(_table(list(rq2.items())))
    if rq4:
        lines.append("## RQ4: explanation traceability\n")
        lines.append(_table(list(rq4.items())))

    a7 = res("A7")
    if a7:
        lines.append(f"## A7: LLM decides vs LLM extracts + code compares (n={a7['n_profiles']})\n")
        lines.append("| arm | profile accuracy | 95% CI | cards returned | card accuracy (returned) | needs review |\n|---|---|---|---|---|---|")
        for key, label in (("llm_decides", "LLM decides (14 Oct design)"), ("extract_compare", "extract + compare (grounded)"),
                           ("extract_compare_ungrounded", "extract + compare (ungrounded)")):
            r = a7[key]
            lines.append(f"| {label} | {r['profile_accuracy']:.3f} | {r['profile_accuracy_ci95']} | "
                         f"{r['cards_returned_rate']:.3f} | {r['card_accuracy_on_returned']:.3f} | {r.get('needs_review_decisions', '-')} |")
        lines.append("")
        for key, label in (("llm_vs_extract_compare_mcnemar", "extract+compare vs LLM decides"),
                           ("extract_compare_grounded_vs_ungrounded_mcnemar", "extract+compare grounded vs ungrounded")):
            r = a7[key]
            lines.append(f"- McNemar, {label}: b={r['b_grounded_fixes']} (first arm right only), "
                         f"c={r['c_grounded_breaks']} (second arm right only), exact p={exact_mcnemar_p(r['b_grounded_fixes'], r['c_grounded_breaks']):.3g}")
        for key, label in (("extraction_grounded", "grounded (reads the documents)"),
                           ("extraction_ungrounded", "ungrounded (from memory = threshold hallucination)")):
            r = a7[key]
            lines.append(f"- Requirement extraction accuracy, {label}: {r['field_accuracy']:.3f} of "
                         f"{r['cards_extracted']} cards x 5 fields; per field: "
                         + ", ".join(f"{f}={v:.2f}" for f, v in r["per_field_accuracy"].items()))
        lines.append("")

    for name, title in (("A2", "A2: rules kept per card (top-k)"), ("A3", "A3: prompt pattern on/off"),
                        ("A4", "A4: model selection"), ("A5", "A5: chunking granularity"), ("A6", "A6: normal vs reworded profiles")):
        r = res(name)
        if r:
            n = meta(name).get("n_profiles")
            lines.append(f"## {title} (core subset n={n})\n")
            lines.append(_table(list(r.items())))

    lines.append(_difficulty_and_card_breakdown(results_dir / "predictions_A1_grounded.jsonl"))
    lines.append("## Limitations\n\nSee `EVALUATION_NOTES.md` (synthetic oracle-derived gold set, borderline-heavy profiles, hedged documents, "
                 "class imbalance, tiny corpus, small quantised model, RAGAS not computed).\n")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RESULTS.md from results.json")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = build_report(args.results_dir)
    (args.results_dir / "RESULTS.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
