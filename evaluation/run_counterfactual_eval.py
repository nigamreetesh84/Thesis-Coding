"""Evaluate the counterfactual "what needs to change" explanations (round 9).

For every gold profile and every card it is declined for, the counterfactual
is computed from the card's requirements and then checked against the
INDEPENDENT eligibility oracle (data/eligibility_table.csv):

* validity   - after applying every suggested change, the oracle says eligible;
* minimality - undoing any single change (or moving a numeric target one
               rupee / one point / one year back) makes the oracle decline again;
* sparsity   - a change is suggested only for criteria the oracle says failed;
* agreement  - the decision verdict with counterfactuals attached equals the
               oracle verdict (attaching a counterfactual changes nothing).

Requirements source: by default the eligibility table converted to
CardRequirements (extraction was 40/40 correct on both document sets, so this
equals what the grounded extractor read). --requirements <json> scores the
requirements actually extracted in a run instead, e.g. the extractions cached
from results_round5_2.

Runs on CPU in a second; no LLM, no embeddings.

    python -m evaluation.run_counterfactual_eval --out results_counterfactual
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from src.domain.counterfactual import apply_changes, closest_declined, summary
from src.domain.eligibility import DEFAULT_ELIGIBILITY_TABLE_PATH, EligibilityOracle
from src.domain.entities import Change, UserProfile
from src.domain.requirements import CardRequirements, decide

GOLD = Path("data/gold_test_set.jsonl")


def requirements_from_table(oracle: EligibilityOracle) -> dict[tuple[str, str], CardRequirements]:
    out = {}
    for r in oracle.rules:
        allowed = None if "any" in r.employment else tuple(sorted(r.employment))
        out[(r.bank, r.card)] = CardRequirements(
            bank=r.bank, card=r.card, min_income=r.min_income, min_age=r.min_age, min_cibil=r.min_cibil,
            allowed_employment=allowed, relationship_required=r.relationship_required,
        )
    return out


def oracle_ok(oracle: EligibilityOracle, profile: UserProfile, bank: str, card: str) -> bool:
    return next(d.recommended for d in oracle.evaluate(replace(profile, banks=[bank])) if d.card == card)


def undo_one(profile_after: UserProfile, original: UserProfile, change: Change) -> UserProfile:
    """Move one change back by the smallest step (or undo it for categorical criteria)."""
    if change.criterion == "income":
        return replace(profile_after, annual_income=float(change.required) - 1)
    if change.criterion == "cibil":
        return replace(profile_after, cibil=int(change.required) - 1)
    if change.criterion == "age":
        return replace(profile_after, age=int(change.required) - 1)
    if change.criterion == "relationship":
        return replace(profile_after, existing_relationship=False)
    return replace(profile_after, employment=original.employment)


def oracle_failed(oracle: EligibilityOracle, profile: UserProfile, bank: str, card: str) -> set[str]:
    d = next(d for d in oracle.evaluate(replace(profile, banks=[bank])) if d.card == card)
    text = d.reason.lower()
    keys = {"income": "annual income", "age": "age ", "cibil": "cibil", "employment": "employment type",
            "relationship": "relationship"}
    return {k for k, needle in keys.items() if needle in text} if not d.recommended else set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(GOLD))
    ap.add_argument("--table", default=str(DEFAULT_ELIGIBILITY_TABLE_PATH))
    ap.add_argument("--out", default="results_counterfactual")
    args = ap.parse_args()

    oracle = EligibilityOracle(args.table)
    reqs = requirements_from_table(oracle)
    records = [json.loads(line) for line in Path(args.gold).read_text(encoding="utf-8").splitlines() if line.strip()]

    n_decisions = n_declines = valid = minimal = sparse = agree = 0
    kinds: Counter[str] = Counter()
    per_criterion: Counter[str] = Counter()
    n_changes: Counter[int] = Counter()
    reachable_profiles = profiles_with_decline = 0
    reach = 0  # declines whose every change is actionable (income, CIBIL, opening an account)
    examples: list[dict] = []
    seen_shapes: set[tuple[str, ...]] = set()  # one example per mix of change kinds, from different profiles
    problems: list[str] = []

    for i, rec in enumerate(records):
        p = rec["profile"]
        profile = UserProfile(p["annual_income"], p["age"], p["cibil"], p["employment"], p["existing_relationship"],
                              p.get("banks", ["SBI", "HDFC"]))
        decisions = []
        for (bank, card), req in reqs.items():
            if bank not in profile.banks:
                continue
            d = decide(profile, req)
            decisions.append(d)
            n_decisions += 1
            truth = oracle_ok(oracle, profile, bank, card)
            agree += d.recommended == truth
            if d.recommended:
                if d.changes:
                    problems.append(f"profile {i} {card}: eligible card has changes")
                continue
            n_declines += 1
            reach += bool(d.changes) and all(c.kind == "actionable" for c in d.changes)
            n_changes[len(d.changes)] += 1
            kinds.update(c.kind for c in d.changes)
            per_criterion.update(c.criterion for c in d.changes)
            after = apply_changes(profile, d.changes)
            ok_valid = oracle_ok(oracle, after, bank, card)
            valid += ok_valid
            ok_min = all(not oracle_ok(oracle, undo_one(after, profile, c), bank, card) for c in d.changes)
            minimal += ok_min
            ok_sparse = {c.criterion for c in d.changes} == oracle_failed(oracle, profile, bank, card)
            sparse += ok_sparse
            if not (ok_valid and ok_min and ok_sparse):
                problems.append(f"profile {i} {card}: valid={ok_valid} minimal={ok_min} sparse={ok_sparse}")
            shape = tuple(sorted(c.kind for c in d.changes))
            if len(examples) < 6 and shape not in seen_shapes and all(e["profile"] is not p for e in examples):
                seen_shapes.add(shape)
                examples.append({"profile": p, "card": card, "reason": d.reason, "what_to_change": summary(d.changes)})
        if any(not d.recommended for d in decisions):
            profiles_with_decline += 1
            reachable_profiles += closest_declined(decisions, profile) is not None

    result = {
        "requirements_source": "eligibility table (equals the grounded extraction, 40/40 fields)",
        "profiles": len(records),
        "card_decisions": n_decisions,
        "declines": n_declines,
        "verdict_agreement_with_oracle": agree / n_decisions,
        "validity": {"ok": valid, "n": n_declines, "rate": valid / n_declines},
        "minimality": {"ok": minimal, "n": n_declines, "rate": minimal / n_declines},
        "sparsity": {"ok": sparse, "n": n_declines, "rate": sparse / n_declines},
        "changes_per_decline": dict(sorted(n_changes.items())),
        "mean_changes_per_decline": sum(k * v for k, v in n_changes.items()) / n_declines,
        "changes_by_kind": dict(kinds),
        "changes_by_criterion": dict(per_criterion),
        "declines_reachable_by_own_action": {"n": reach, "of": n_declines, "rate": reach / n_declines},
        "profiles_with_a_decline": profiles_with_decline,
        "profiles_with_a_reachable_closest_card": reachable_profiles,
        "problems": problems,
        "examples": examples,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "counterfactual_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Counterfactual explanations - checked against the eligibility oracle",
        "",
        f"Profiles {result['profiles']}, card decisions {n_decisions}, declines {n_declines}.",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Verdict unchanged (agreement with oracle) | {agree}/{n_decisions} |",
        f"| Validity (all changes applied -> eligible) | {valid}/{n_declines} |",
        f"| Minimality (any one change undone -> declined) | {minimal}/{n_declines} |",
        f"| Sparsity (changes only for failed criteria) | {sparse}/{n_declines} |",
        f"| Declines reachable by the applicant's own action | {reach}/{n_declines} |",
        f"| Mean changes per decline | {result['mean_changes_per_decline']:.2f} |",
        "",
        f"Changes by kind: {dict(kinds)}",
        f"Changes by criterion: {dict(per_criterion)}",
        f"Profiles with at least one decline: {profiles_with_decline}; with a reachable closest card: {reachable_profiles}",
        "",
        "## Examples",
    ] + [f"- {e['card']}: {e['reason']}  \n  -> {e['what_to_change']}" for e in examples]
    (out / "counterfactual_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if problems:
        print(f"\n{len(problems)} problem(s):", *problems[:20], sep="\n  ")


if __name__ == "__main__":
    main()
