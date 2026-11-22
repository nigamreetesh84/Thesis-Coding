"""Field-level scoring of extractor outputs (same unit as the thesis: card x field).

A field is correct when the extracted value equals the truth; an UNSTATED
criterion (truth None, e.g. Axis publishes no CIBIL threshold) is correct only
when the model also returns null -- inventing a value there is a hallucination.
"""

from __future__ import annotations

from typing import Any

from evaluation.metrics import wilson_interval
from src.domain.requirements import CardRequirements
from src.model import _same_card, extract_json_items

FIELDS = ("min_income", "min_age", "min_cibil", "allowed_employment", "relationship_required")
ALL_TYPES = {"salaried", "self-employed", "business owner"}


def parse_prediction(raw: str, bank: str, card: str) -> tuple[dict[str, Any] | None, bool]:
    """(requirements dict or None, json_valid)."""
    items = extract_json_items(raw)
    if not items:
        return None, False
    own = [i for i in items if _same_card(bank, card, str(i.get("bank", "")), str(i.get("card", "")))]
    item = own[0] if own else items[0]
    return CardRequirements.from_llm_dict(bank, card, item).as_dict(), True


def field_matches(extracted: dict[str, Any] | None, truth: dict[str, Any]) -> dict[str, bool]:
    if extracted is None:
        return dict.fromkeys(FIELDS, False)
    t_emp = set(truth["employment"])
    t_any = "any" in t_emp or t_emp >= ALL_TYPES
    e_emp = extracted.get("allowed_employment")
    e_any = e_emp in (None, "any") or (isinstance(e_emp, list) and set(e_emp) >= ALL_TYPES)
    employment_ok = (t_any and e_any) if (t_any or e_any) else set(e_emp) == t_emp

    def num_ok(key: str) -> bool:
        value, want = extracted.get(key), truth.get(key)
        if want is None:
            return value is None
        return value is not None and abs(float(value) - float(want)) < 0.5

    return {
        "min_income": num_ok("min_income"),
        "min_age": num_ok("min_age"),
        "min_cibil": num_ok("min_cibil"),
        "allowed_employment": employment_ok,
        "relationship_required": bool(extracted.get("relationship_required")) == bool(truth.get("relationship_required")),
    }


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """rows: [{'bank','card','raw','truth'}] -> summary metrics."""
    fields_ok = 0
    cards_ok = 0
    json_ok = 0
    per_field = dict.fromkeys(FIELDS, 0)
    hallucinated_unstated = 0
    unstated = 0
    for row in rows:
        extracted, valid = parse_prediction(row["raw"], row["bank"], row["card"])
        json_ok += valid
        m = field_matches(extracted, row["truth"])
        row["extracted"] = extracted
        row["matches"] = m
        fields_ok += sum(m.values())
        cards_ok += all(m.values())
        for f, ok in m.items():
            per_field[f] += ok
        for key in ("min_income", "min_cibil"):
            if row["truth"].get(key) is None:
                unstated += 1
                hallucinated_unstated += bool(extracted and extracted.get(key) is not None)
    n = len(rows)
    lo, hi = wilson_interval(fields_ok, n * len(FIELDS)) if n else (0.0, 0.0)
    clo, chi = wilson_interval(cards_ok, n) if n else (0.0, 0.0)
    return {
        "n_cards": n,
        "fields_correct": fields_ok,
        "fields_total": n * len(FIELDS),
        "field_accuracy": fields_ok / (n * len(FIELDS)) if n else 0.0,
        "field_accuracy_ci95": [round(lo, 3), round(hi, 3)],
        "card_all_correct": cards_ok,
        "card_accuracy": cards_ok / n if n else 0.0,
        "card_accuracy_ci95": [round(clo, 3), round(chi, 3)],
        "json_valid_rate": json_ok / n if n else 0.0,
        "per_field_accuracy": {f: v / n if n else 0.0 for f, v in per_field.items()},
        "unstated_thresholds": unstated,
        "hallucinated_unstated_thresholds": hallucinated_unstated,
    }


def mcnemar_cards(a: list[bool], b: list[bool]) -> dict[str, Any]:
    """Exact McNemar on paired per-card all-correct outcomes (b = only A right, c = only B right)."""
    from statsmodels.stats.contingency_tables import mcnemar

    b_only = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    c_only = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    both = sum(1 for x, y in zip(a, b, strict=True) if x and y)
    neither = len(a) - b_only - c_only - both
    p = float(mcnemar([[both, b_only], [c_only, neither]], exact=True).pvalue) if (b_only + c_only) else 1.0
    return {"b_first_only": b_only, "c_second_only": c_only, "p_value": p, "n": len(a)}
