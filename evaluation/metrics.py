"""Evaluation metrics used across the ablation and RAGAS scripts."""
from __future__ import annotations

from collections.abc import Sequence


def classification_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> dict[str, float]:
    """Precision, recall, F1 and accuracy for the per-card recommend decision.

    RQ1: accuracy against the deterministic oracle's expected label.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")

    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if not t and not p)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def context_precision_recall(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> dict:
    """Context precision/recall of retrieval against known-relevant chunk ids.

    RQ2: how much of what was retrieved was actually relevant, and how much
    of what was relevant got retrieved.
    """
    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)
    if not retrieved_set and not relevant_set:
        return {"context_precision": 1.0, "context_recall": 1.0}

    hits = retrieved_set & relevant_set
    precision = len(hits) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(hits) / len(relevant_set) if relevant_set else 0.0
    return {"context_precision": precision, "context_recall": recall}

def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (well behaved at 0 and 1 and small n).

    With n = 20 or 40 profiles the uncertainty is large (e.g. 8/20 -> 0.22-0.61),
    so every accuracy in the dissertation should carry this interval.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# Fields compared when scoring the LLM's extracted requirements against the table.
EXTRACTION_FIELDS = ("min_income", "min_age", "min_cibil", "allowed_employment", "relationship_required")


def extraction_field_matches(extracted: dict, truth: dict) -> dict[str, bool]:
    """Field-by-field comparison of one card's extracted requirements with the table row.

    `extracted` is CardRequirements.as_dict(); `truth` holds min_income, min_age,
    min_cibil, employment (tuple, ("any",) for any) and relationship_required.
    A missing (None) relationship is read as "not required", matching how the
    decision procedure applies it.
    """
    truth_emp = truth["employment"]
    truth_any = "any" in truth_emp or set(truth_emp) >= {"salaried", "self-employed", "business owner"}
    ext_emp = extracted.get("allowed_employment")
    ext_any = ext_emp in (None, "any") or (isinstance(ext_emp, list) and set(ext_emp) >= {"salaried", "self-employed", "business owner"})
    if truth_any or ext_any:
        employment_ok = truth_any and ext_any
    else:
        employment_ok = set(ext_emp) == set(truth_emp)

    def num_ok(key: str) -> bool:
        value = extracted.get(key)
        return value is not None and abs(float(value) - float(truth[key])) < 0.5

    return {
        "min_income": num_ok("min_income"),
        "min_age": num_ok("min_age"),
        "min_cibil": num_ok("min_cibil"),
        "allowed_employment": employment_ok,
        "relationship_required": bool(extracted.get("relationship_required")) == bool(truth["relationship_required"]),
    }
