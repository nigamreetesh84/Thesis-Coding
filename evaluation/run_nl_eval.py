"""Evaluate the natural-language interface (src/nl_interface.py) WITHOUT touching
the frozen round 5.5 results.

The decision path (retrieval -> extraction -> deterministic comparison -> guard)
is unchanged, so only the two new ends are measured:

1. Query parsing, field level: income, age, CIBIL, employment, existing
   relationship (5 fields per message), LLM parser vs a regex baseline.
   The reference is what the MESSAGE says (e.g. "about 28.0 lakh" -> 2,800,000),
   not the unrounded gold income, so a parser is not penalised for the
   message's own rounding.
2. End to end from free text: all card decisions correct vs the gold labels
   (the same profile-accuracy metric as Chapter 4). Profiles whose message
   rounding alone changes a gold label are counted and reported separately.
   A message with a missing field triggers a follow-up question (counted, and
   scored as not correct).
3. Answer writer: share of LLM sentences accepted by the guard, and agreement
   of the final answer with the engine's verdicts (1.000 by construction:
   rejected sentences fall back to the engine's reason).

Two message sets per profile:
  gold_text  = the gold set's own profile_text (98 templated + 22 reworded)
  user_style = first-person messages generated here, rotating income styles
               (rupees per annum, lakh with decimals, monthly salary, LPA) and
               everyday wording for job and bank relationship.

Usage (Colab, same environment variables as the A8 notebook):
    python -m evaluation.run_nl_eval --split holdout --out-dir $OUT/nl_holdout
    python -m evaluation.run_nl_eval --split holdout --use-stub-llm   # offline smoke test
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

from config.settings import settings
from evaluation.metrics import wilson_interval
from evaluation.run_ablation import (
    GoldRecord,
    build_retriever,
    load_gold_set,
    profile_correct,
    select_records,
)
from src.domain.eligibility import EligibilityOracle
from src.domain.entities import UserProfile
from src.domain.query_parsing import ParsedQuery, rule_based_fields
from src.infrastructure.llm_client import LocalLLMClient, StubLLM
from src.model import RAGRecommender, extract_json_items
from src.nl_interface import AnswerWriter, QueryParser

FIELDS = ("income", "age", "cibil", "employment", "existing_relationship")


# ---------------------------------------------------------------------------
# Messages and what they actually state
# ---------------------------------------------------------------------------
def _inr(value: float) -> str:
    digits = str(int(round(value)))
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups) + "," + tail


def gold_text_message(record: GoldRecord) -> tuple[str, float]:
    """The gold set's own text and the income it states (see build_gold_set.py)."""
    income = record.profile.annual_income
    if record.variant == "reworded" and income >= 100_000:
        stated = round(income / 100_000, 1) * 100_000
    else:
        stated = float(round(income))
    return record.profile_text, stated


_JOB = {
    "salaried": ["I'm a salaried employee", "I work full-time at an IT company on a monthly salary",
                 "I have a regular job with a private bank"],
    "self-employed": ["I'm self-employed", "I work as a freelance consultant", "I run my own practice as a CA"],
    "business owner": ["I run my own business", "I own a small trading company", "I'm an entrepreneur"],
}
_REL_YES = ["I already have a savings account with SBI.", "I've banked with HDFC for years.",
            "I hold a salary account with SBI."]
_REL_NO = ["I don't have any account with SBI or HDFC.", "I'm not an existing customer of either bank."]
_WANTS = ["Which credit cards can I get?", "Suggest a travel card for me.",
          "I want a card with cashback on shopping. What am I eligible for?", "What are my options?"]


def user_style_message(record: GoldRecord, i: int, seed: int = 2026) -> tuple[str, float]:
    p, rng = record.profile, random.Random(seed * 1000 + i)
    style = ("rupees", "lakh", "monthly", "lpa")[i % 4]
    if style == "lakh" and p.annual_income >= 100_000:
        stated = round(p.annual_income / 1_000) * 1_000
        income = f"{stated / 100_000:.2f} lakh a year"
    elif style == "monthly":
        month = round(p.annual_income / 12)
        stated, income = month * 12, f"Rs {_inr(month)} per month"
    elif style == "lpa" and p.annual_income >= 100_000:
        stated = round(p.annual_income / 10_000) * 10_000
        income = f"{stated / 100_000:.1f} LPA"
    else:
        stated, income = round(p.annual_income), f"Rs {_inr(p.annual_income)} per annum"
    job = rng.choice(_JOB[p.employment])
    rel = rng.choice(_REL_YES if p.existing_relationship else _REL_NO)
    want = rng.choice(_WANTS)
    template = rng.choice([
        "{job}, {age} years old, earning {income}. My CIBIL score is {cibil}. {rel} {want}",
        "Hi, I'm {age}. {job} and I make {income}. CIBIL {cibil}. {rel} {want}",
        "{want} Age {age}, credit score {cibil}, income {income}. {job}. {rel}",
    ])
    text = template.format(job=job, age=p.age, income=income, cibil=p.cibil, rel=rel, want=want)
    return text, float(stated)


def field_matches(parsed: ParsedQuery, truth: UserProfile, stated_income: float) -> dict[str, bool]:
    return {
        "income": parsed.income is not None and abs(parsed.income - stated_income) <= 1.0,
        "age": parsed.age == truth.age,
        "cibil": parsed.cibil == truth.cibil,
        "employment": parsed.employment == truth.employment,
        "existing_relationship": parsed.existing_relationship == truth.existing_relationship,
    }


def _ci(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else None, "ci95": [round(lo, 3), round(hi, 3)]}


# ---------------------------------------------------------------------------
def evaluate(records: list[GoldRecord], llm: Any, recommender: RAGRecommender, with_writer: bool) -> dict[str, Any]:
    oracle = EligibilityOracle(settings.eligibility_table_path)
    parser = QueryParser(llm)
    writer = AnswerWriter(llm)
    results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for set_name in ("gold_text", "user_style"):
        messages = [gold_text_message(r) if set_name == "gold_text" else user_style_message(r, i)
                    for i, r in enumerate(records)]
        generate_batch = getattr(llm, "generate_batch", None)
        prompts = [QueryParser.build_prompt(m) for m, _ in messages]
        raws = list(generate_batch(prompts)) if callable(generate_batch) else [llm.generate(p) for p in prompts]

        stats: dict[str, Any] = {arm: {f: 0 for f in FIELDS} | {"all": 0, "followup": 0, "e2e": 0}
                                 for arm in ("llm", "regex")}
        rounding_flips = 0
        writer_sent = writer_ok = writer_agree = 0
        writer_inputs: list[tuple[list[Any], UserProfile]] = []

        for i, (record, (message, stated), raw) in enumerate(zip(records, messages, raws, strict=True)):
            items = extract_json_items(raw)
            arms = {"llm": ParsedQuery.from_llm_dict(items[0] if items else {}),
                    "regex": ParsedQuery.from_llm_dict(rule_based_fields(message))}
            stated_profile = replace(record.profile, annual_income=stated)
            flips = not profile_correct(oracle.evaluate(stated_profile), record.expected)
            rounding_flips += flips
            row: dict[str, Any] = {"set": set_name, "i": i, "variant": record.variant,
                                   "difficulty": record.difficulty, "message": message,
                                   "stated_income": stated, "rounding_changes_label": flips}
            for arm, parsed in arms.items():
                ok = field_matches(parsed, record.profile, stated)
                for f in FIELDS:
                    stats[arm][f] += ok[f]
                stats[arm]["all"] += all(ok.values())
                e2e = False
                if parsed.missing:
                    stats[arm]["followup"] += 1
                else:
                    decisions, _ = recommender.recommend(parsed.to_profile())
                    e2e = profile_correct(decisions, record.expected)
                    if arm == "llm" and with_writer:
                        writer_inputs.append((decisions, parsed.to_profile()))
                stats[arm]["e2e"] += e2e
                row[arm] = {"parsed": {f: getattr(parsed, f) for f in FIELDS}, "fields_ok": ok,
                            "missing": parsed.missing, "e2e_correct": e2e, "income_text": parsed.income_text}
            rows.append(row)

        if with_writer and writer_inputs:
            w_prompts = [AnswerWriter.build_prompt(d) for d, _ in writer_inputs]
            w_raws = list(generate_batch(w_prompts)) if callable(generate_batch) else [llm.generate(p) for p in w_prompts]
            cache = dict(zip(w_prompts, w_raws, strict=True))

            class _Replay:  # hands each prompt its batched reply
                def generate(self, prompt: str) -> str:
                    return cache[prompt]

            writer.llm_client = _Replay()
            for decisions, profile in writer_inputs:
                for answer, decision in zip(writer.write(decisions, profile), decisions, strict=True):
                    writer_sent += 1
                    writer_ok += answer.source == "llm"
                    writer_agree += (answer.recommended == decision.recommended
                                     and answer.needs_review == decision.needs_review)

        n = len(records)
        set_res: dict[str, Any] = {"n_messages": n, "messages_whose_rounding_changes_a_gold_label": rounding_flips}
        for arm in ("llm", "regex"):
            s = stats[arm]
            set_res[f"{arm}_parser"] = {
                "field_accuracy": {f: _ci(s[f], n) for f in FIELDS},
                "fields_correct": _ci(sum(s[f] for f in FIELDS), n * len(FIELDS)),
                "all_five_fields_correct": _ci(s["all"], n),
                "asked_follow_up": _ci(s["followup"], n),
                "e2e_profile_accuracy": _ci(s["e2e"], n),
            }
        if with_writer:
            set_res["answer_writer"] = {
                "llm_sentences_accepted_by_guard": _ci(writer_ok, writer_sent),
                "final_answer_matches_engine_verdict": _ci(writer_agree, writer_sent),
            }
        results[set_name] = set_res
    return {"results": results, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="holdout", choices=["all", "test", "dev", "dev40", "holdout"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--use-stub-llm", action="store_true", help="GPU-free smoke test (regex parser, no writer)")
    ap.add_argument("--no-writer", action="store_true")
    ap.add_argument("--out-dir", default="results_nl")
    args = ap.parse_args()

    records = select_records(load_gold_set(), args.split, args.limit)
    llm = StubLLM() if args.use_stub_llm else LocalLLMClient(settings.llm_model_name)
    recommender = RAGRecommender(build_retriever(settings.top_k), llm, decision_mode="extract_compare")
    out = evaluate(records, llm, recommender, with_writer=not args.no_writer)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"split": args.split, "n_profiles": len(records), "llm": "StubLLM" if args.use_stub_llm
            else settings.llm_model_name, "decision_mode": "extract_compare (frozen round 5.5)"}
    (out_dir / "nl_results.json").write_text(json.dumps(meta | out["results"], indent=2), encoding="utf-8")
    with (out_dir / "nl_rows.jsonl").open("w", encoding="utf-8") as f:
        for row in out["rows"]:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(meta | out["results"], indent=2))


if __name__ == "__main__":
    main()
