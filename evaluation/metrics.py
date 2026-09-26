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