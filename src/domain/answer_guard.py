"""Guard for the natural-language answer writer.

The writer LLM only rephrases decisions that the deterministic engine has
already made. This module checks each sentence it writes before the applicant
sees it. A sentence is rejected if it

* states the wrong verdict (an eligible card described negatively, a declined
  card not described as declined, a needs-review card described as either), or
* contains a number that does not appear in the engine's reason or the
  applicant's profile (a hallucinated threshold or value).

A rejected sentence is replaced by the engine's own reason, so the final
answer can never contradict the decision. This is a runtime check on the
PHRASING only; decisions themselves are not re-validated against the
eligibility table (which is never read at inference time).
"""

from __future__ import annotations

import re

from src.domain.entities import Decision, UserProfile

_NEGATIVE_RE = re.compile(
    r"\b(not|n't|cannot|ineligible|unfortunately|below|under|short of|fails?|failed|lacks?|unable|no longer)\b",
    re.IGNORECASE,
)
_REVIEW_RE = re.compile(r"\b(confirm|could not|couldn't|check|verify|review)\b", re.IGNORECASE)
_AMOUNT_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|k\b)?", re.IGNORECASE)


def numbers_in(text: str) -> set[int]:
    """All amounts in a text, in rupees where a unit is given ('6 lakh' -> 600000)."""
    found: set[int] = set()
    for match in _AMOUNT_RE.finditer(text):
        value = float(match.group(1).replace(",", ""))
        unit = (match.group(2) or "").lower()
        if unit.startswith(("lakh", "lac")):
            value *= 1e5
        elif unit.startswith("crore"):
            value *= 1e7
        elif unit == "k":
            value *= 1e3
        found.add(int(round(value)))
    return found


def allowed_numbers(decision: Decision, profile: UserProfile) -> set[int]:
    allowed = numbers_in(decision.reason) | numbers_in(f"{decision.bank} {decision.card}")
    allowed |= {int(round(profile.annual_income)), profile.age, profile.cibil}
    # the same amount written in lakh with 1-2 decimals ("28.04 lakh") is the same number
    widened = set()
    for n in allowed:
        if n >= 100_000:
            widened |= {int(round(n / 1e3) * 1e3), int(round(n / 1e4) * 1e4), int(round(n / 1e5) * 1e5)}
    return allowed | widened


def check_sentence(sentence: object, decision: Decision, profile: UserProfile) -> tuple[bool, str]:
    """Return (accepted, reason)."""
    if not isinstance(sentence, str) or not sentence.strip():
        return False, "missing"
    negative = bool(_NEGATIVE_RE.search(sentence))
    if decision.needs_review:
        if not _REVIEW_RE.search(sentence):
            return False, "needs-review card not described as unconfirmed"
    elif decision.recommended and negative:
        return False, "eligible card described negatively"
    elif not decision.recommended and not negative:
        return False, "declined card not described as declined"
    extra = numbers_in(sentence) - allowed_numbers(decision, profile)
    if extra:
        return False, f"numbers not in the engine's reason: {sorted(extra)}"
    return True, "ok"
