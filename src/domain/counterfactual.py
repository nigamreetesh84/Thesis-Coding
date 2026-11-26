"""Counterfactual explanations: what would have to change for a declined card.

A decline reason says WHICH rule failed ("annual income of Rs 4,50,000 is
below the Rs 6,00,000 minimum"). This module adds the counterfactual form
proposed by Wachter, Mittelstadt and Russell (2018): the smallest change to the
applicant's facts that would turn the decision into an approval ("raise annual
income by Rs 1,50,000, about Rs 12,500 a month").

Design rules
------------
* Computed by deterministic code from the SAME requirements that produced the
  decision (extracted by the LLM, or approved by a bank employee). No LLM and
  no eligibility table at inference time, so a counterfactual can never
  disagree with the verdict it explains.
* Only failed criteria get a change (sparse), and each numeric target is the
  stated minimum itself (minimal: one rupee or one point less still fails).
* All changes for one card are needed together: eligibility is a conjunction
  of criteria, so the list is the whole set, not a menu.
* Changes are labelled by whether the applicant can act on them: income, CIBIL
  and opening an account are actionable; age only changes with time; an
  employment type the card does not accept is treated as fixed, because a
  recommender should not advise someone to change job to get a credit card.
* A decision the fail-closed guard marked needs_review gets no counterfactual:
  its requirements could not be read reliably.

Validity and minimality are checked against the independent eligibility table
by evaluation/run_counterfactual_eval.py.
"""

from __future__ import annotations

from dataclasses import replace

from src.domain.entities import Change, Decision, UserProfile
from src.domain.requirements import CardRequirements, _rupees, check_requirements, normalise_employment

KIND_ORDER = {"actionable": 0, "time": 1, "fixed": 2}


def what_needs_to_change(profile: UserProfile, req: CardRequirements) -> tuple[Change, ...]:
    """The changes that would make the applicant meet every stated requirement of the card.

    Returns an empty tuple when the applicant already qualifies.
    """
    failed = {r.criterion for r in check_requirements(profile, req) if not r.passed}
    changes: list[Change] = []

    if "income" in failed and req.min_income is not None:
        gap = req.min_income - profile.annual_income
        changes.append(
            Change(
                "income",
                "actionable",
                profile.annual_income,
                req.min_income,
                gap,
                f"raise annual income by {_rupees(gap)} (about {_rupees(gap / 12)} a month) "
                f"to the {_rupees(req.min_income)} minimum",
            )
        )
    if "cibil" in failed and req.min_cibil is not None:
        gap = req.min_cibil - profile.cibil
        changes.append(
            Change(
                "cibil",
                "actionable",
                profile.cibil,
                req.min_cibil,
                gap,
                f"raise the CIBIL score by {gap} point{'s' if gap != 1 else ''} to {req.min_cibil}",
            )
        )
    if "relationship" in failed:
        changes.append(
            Change(
                "relationship",
                "actionable",
                False,
                True,
                None,
                f"open an account with {req.bank} first (the card is offered to existing customers only)",
            )
        )
    if "age" in failed and req.min_age is not None:
        gap = req.min_age - profile.age
        changes.append(
            Change(
                "age",
                "time",
                profile.age,
                req.min_age,
                gap,
                f"wait until age {req.min_age} (in {gap} year{'s' if gap != 1 else ''})",
            )
        )
    if "employment" in failed and req.allowed_employment:
        accepted = ", ".join(req.allowed_employment)  # machine-readable list, used by apply_changes
        types = list(req.allowed_employment)
        spoken = types[0] if len(types) == 1 else ", ".join(types[:-1]) + " or " + types[-1]
        changes.append(
            Change(
                "employment",
                "fixed",
                profile.employment,
                accepted,
                None,
                f"the card accepts only {spoken} applicants, so a change in income or score alone "
                f"would not make a {profile.employment} applicant eligible",
            )
        )
    return tuple(changes)


def attach(decision: Decision, profile: UserProfile, req: CardRequirements) -> Decision:
    """Return the decision with its counterfactual attached (declines only; verdict untouched)."""
    if decision.recommended or decision.needs_review:
        return decision
    return replace(decision, changes=what_needs_to_change(profile, req))


def apply_changes(profile: UserProfile, changes: tuple[Change, ...]) -> UserProfile:
    """The profile the applicant would have after making every change (used to verify validity)."""
    p = profile
    for c in changes:
        if c.criterion == "income":
            p = replace(p, annual_income=float(c.required))
        elif c.criterion == "cibil":
            p = replace(p, cibil=int(c.required))
        elif c.criterion == "age":
            p = replace(p, age=int(c.required))
        elif c.criterion == "relationship":
            p = replace(p, existing_relationship=True)
        elif c.criterion == "employment":
            first = str(c.required).split(",")[0].strip()
            p = replace(p, employment=normalise_employment(first) or first)
    return p


def summary(changes: tuple[Change, ...]) -> str:
    """One applicant-facing sentence, e.g. 'To qualify: raise ...; and raise ....'"""
    if not changes:
        return ""
    if any(c.kind == "fixed" for c in changes):
        fixed = next(c for c in changes if c.kind == "fixed")
        return "Not reachable by changing income or score: " + fixed.text + "."
    parts = [c.text for c in changes]
    joined = parts[0] if len(parts) == 1 else "; ".join(parts[:-1]) + "; and " + parts[-1]
    return "To qualify: " + joined + "."


def effort(changes: tuple[Change, ...], profile: UserProfile) -> tuple[int, int, float]:
    """Sort key for 'closest card': fewest blocked criteria first, then fewest changes, then smallest relative gap."""
    worst = max((KIND_ORDER[c.kind] for c in changes), default=0)
    relative = 0.0
    for c in changes:
        if c.criterion == "income" and c.gap is not None and float(c.required) > 0:
            relative += c.gap / float(c.required)
        elif c.criterion == "cibil" and c.gap is not None:
            relative += c.gap / 600  # CIBIL range 300-900
    return worst, len(changes), relative


def closest_declined(decisions: list[Decision], profile: UserProfile) -> Decision | None:
    """The declined card the applicant is nearest to, among those reachable by their own actions."""
    reachable = [d for d in decisions if d.changes and all(c.kind == "actionable" for c in d.changes)]
    if not reachable:
        return None
    return min(reachable, key=lambda d: effort(d.changes, profile))
