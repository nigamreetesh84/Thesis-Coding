"""Builds the gold evaluation set: ~120 synthetic applicant profiles with
expected outcomes, stratified across income bands, CIBIL ranges,
employment types and CIBIL ranges, and balanced between easy/borderline
difficulty and normal/reworded phrasing.

Run as a script: python -m evaluation.build_gold_set
"""
from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from src.domain.eligibility import EligibilityOracle
from src.domain.entities import UserProfile

RANDOM_SEED = 42
N_PROFILES = 120
N_REWORDED = 22

INCOME_BANDS = [150_000, 300_000, 600_000, 900_000, 1_200_000, 1_500_000, 2_000_000, 2_500_000]
EMPLOYMENT_TYPES = ["salaried", "self-employed", "business owner"]
CIBIL_RANGES = [(650, 690), (691, 750), (751, 800), (801, 850), (851, 900)]

# Thresholds used purely to construct realistic borderline profiles for
# testing the recommender; NOT used to compute expected labels (only the
# oracle, reading data/eligibility_table.csv, ever computes those).
BORDERLINE_THRESHOLDS = {
    "income": [250_000, 300_000, 350_000, 600_000, 1_200_000, 1_500_000, 2_000_000, 2_500_000],
    "cibil": [650, 700, 750, 780, 800],
    "age": [18, 21],
}

DEFAULT_GOLD_SET_PATH = Path("data/gold_test_set.jsonl")


def cohens_kappa(labels_a: Sequence[bool], labels_b: Sequence[bool]) -> float:
    """Cohen's kappa inter-rater agreement between two boolean label sequences."""
    if len(labels_a) != len(labels_b):
        raise ValueError("labels_a and labels_b must be the same length")
    n = len(labels_a)
    if n == 0:
        return 1.0
    po = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b) / n
    p_a_true = sum(labels_a) / n
    p_b_true = sum(labels_b) / n
    pe = p_a_true * p_b_true + (1 - p_a_true) * (1 - p_b_true)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def random_profile(rng: random.Random, difficulty: str) -> UserProfile:
    employment = rng.choice(EMPLOYMENT_TYPES)
    banks = ["SBI", "HDFC"]

    if difficulty == "borderline":
        income_threshold = rng.choice(BORDERLINE_THRESHOLDS["income"])
        delta = rng.uniform(-0.05, 0.05) * income_threshold
        income = max(0.0, income_threshold + delta)

        cibil_threshold = rng.choice(BORDERLINE_THRESHOLDS["cibil"])
        cibil = max(300, min(900, cibil_threshold + rng.randint(-5, 5)))

        age_threshold = rng.choice(BORDERLINE_THRESHOLDS["age"])
        age = max(18, age_threshold + rng.randint(-1, 1))
    else:
        income = rng.choice(INCOME_BANDS) * rng.uniform(0.8, 1.4)
        cibil_lo, cibil_hi = rng.choice(CIBIL_RANGES)
        cibil = rng.randint(cibil_lo, cibil_hi)
        age = rng.randint(21, 65)

    existing_relationship = rng.random() < 0.35
    return UserProfile(
        annual_income=round(income, 2),
        age=age,
        cibil=int(cibil),
        employment=employment,
        existing_relationship=existing_relationship,
        banks=banks,
    )


def _reword_profile_text(profile: UserProfile, rng: random.Random) -> str:
    """Render an everyday-language, reworded restatement of a profile.

    Used for the 20-25 'reworded' variant profiles (RQ4 / A6 ablation):
    same underlying facts, different surface wording (e.g. '8 lakh' instead
    of '800000', synonyms for job type, reordered conditions).
    """
    lakhs = profile.annual_income / 100_000
    income_phrase = (
        f"about {lakhs:.1f} lakh rupees a year" if lakhs >= 1 else f"{round(profile.annual_income):,} rupees a year"
    )
    job_synonyms = {
        "salaried": ["a salaried employee", "someone drawing a monthly salary", "working a regular job"],
        "self-employed": ["self-employed", "a freelancer", "working independently"],
        "business owner": ["a business owner", "someone who runs their own company", "an entrepreneur"],
    }
    job_phrase = rng.choice(job_synonyms[profile.employment])
    relationship_phrase = (
        "who already banks with them" if profile.existing_relationship else "who is not an existing customer"
    )
    return (
        f"I'm {profile.age} years old, {job_phrase}, earning {income_phrase}, with a credit "
        f"score near {profile.cibil}, and {relationship_phrase}. Which of {', '.join(profile.banks)}'s "
        f"cards could I get?"
    )


def build_gold_set(
    n_profiles: int = N_PROFILES, n_reworded: int = N_REWORDED, seed: int = RANDOM_SEED
) -> list[dict]:
    """Generate the full list of gold-set records (not yet written to disk)."""
    rng = random.Random(seed)
    oracle = EligibilityOracle()

    records: list[dict] = []
    reworded_indices = set(rng.sample(range(n_profiles), k=min(n_reworded, n_profiles)))

    for i in range(n_profiles):
        difficulty = "borderline" if i % 2 == 0 else "easy"
        profile = random_profile(rng, difficulty)
        expected = [asdict(d) for d in oracle.evaluate(profile)]
        variant = "reworded" if i in reworded_indices else "normal"

        record = {
            "profile": {
                "annual_income": profile.annual_income,
                "age": profile.age,
                "cibil": profile.cibil,
                "employment": profile.employment,
                "existing_relationship": profile.existing_relationship,
                "banks": profile.banks,
            },
            "profile_text": _reword_profile_text(profile, rng) if variant == "reworded" else profile.to_query(),
            "expected": expected,
            "variant": variant,
            "difficulty": difficulty,
        }
        records.append(record)
    return records


def finalize(records: list[dict], kappa_a: Sequence[bool], kappa_b: Sequence[bool], min_kappa: float = 0.80) -> float:
    """Validate the gold set is only finalized once annotator agreement is high.

    Raises:
        AssertionError: if Cohen's kappa between the two annotation passes
            falls below min_kappa.
    """
    kappa = cohens_kappa(kappa_a, kappa_b)
    assert kappa >= min_kappa, f"Gold set annotation kappa ({kappa:.3f}) is below the required {min_kappa}"
    return kappa


def write_gold_set(records: list[dict], path: str | Path = DEFAULT_GOLD_SET_PATH) -> None:
    """Write gold set records to a JSON Lines file, one profile per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def main() -> None:
    """CLI entry point: python -m evaluation.build_gold_set."""
    records = build_gold_set()

    # Self-consistency smoke test: re-running the (deterministic) oracle
    # over the same profiles must reproduce the stored expected labels.
    # This is NOT a substitute for real dual-human annotation kappa, which
    # a dissertation write-up should report from an actual annotation pass.
    oracle = EligibilityOracle()
    labels_a = [d["recommended"] for r in records for d in r["expected"]]
    labels_b = [
        d.recommended
        for r in records
        for d in oracle.evaluate(
            UserProfile(
                annual_income=r["profile"]["annual_income"],
                age=r["profile"]["age"],
                cibil=r["profile"]["cibil"],
                employment=r["profile"]["employment"],
                existing_relationship=r["profile"]["existing_relationship"],
                banks=r["profile"]["banks"],
            )
        )
    ]
    kappa = finalize(records, labels_a, labels_b)
    print(f"Gold set generation self-consistency kappa: {kappa:.3f}")

    write_gold_set(records)
    n_reworded = sum(1 for r in records if r["variant"] == "reworded")
    print(f"Wrote {len(records)} profiles to {DEFAULT_GOLD_SET_PATH} ({n_reworded} reworded)")


if __name__ == "__main__":
    main()