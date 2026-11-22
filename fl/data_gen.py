"""Per-bank training and test data for the federated rule extractor.

Every example is (frozen extraction prompt, target JSON) for ONE card:

    prompt = ExtractionPromptBuilder().build(chunks)     # src/model.py, unchanged
    target = [{"bank", "card", "requirements": {...}, "sources": {...}}]

Training cards are SYNTHETIC (invented names and thresholds), rendered in each
bank's house style with the hard-document perturbations of the thesis (A8):
numbers in words, monthly figures, superseded thresholds, other cards' figures,
fees and spends that are not conditions, negated employment, unstated criteria.

Test sets (never trained on):
  * synthetic_test  -- unseen synthetic cards per bank, plus a HELD-OUT phrasing
                       family no bank uses in training (generalisation to new wording);
  * real_thesis     -- the thesis documents data/rule_docs (8 cards) and
                       data/rule_docs_hard (8 cards), truth = eligibility table;
  * real_axis       -- the Axis document compiled from public pages (4 cards),
                       truth = data/axis/eligibility_axis.csv.

Approved-portal corrections (data/portal/banks/<BANK>/corrections.jsonl) are
added to that bank's training data when present: the employee's review is the
bank's own labelled data, which is exactly what federated learning keeps local.
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

from src.model import ExtractionPromptBuilder
from src.pipeline import RuleChunker

BANKS = ("SBI", "HDFC", "AXIS")
EMPLOYMENT_TYPES = ("salaried", "self-employed", "business owner")
INCOME_GRID_LAKH = (1.8, 2.4, 2.5, 3, 3.5, 4.2, 5, 6, 7.2, 8, 9, 10, 12, 15, 18, 20, 24, 25, 30, 36)
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
         10: "ten", 12: "twelve", 15: "fifteen", 18: "eighteen", 20: "twenty", 24: "twenty-four", 25: "twenty-five",
         30: "thirty", 36: "thirty-six"}
NAME_A = ("Aurora", "Nova", "Prime Plus", "Platinum", "Signature", "Select", "Edge", "Zenith", "Swift", "Saver",
          "Elite Plus", "Rise", "Orbit", "Vista", "Pulse", "Crest", "Horizon", "Spark", "Legend", "Infinity")
NAME_B = ("Rewards", "Cashback", "Travel", "Lifestyle", "Fuel", "Shopping", "Business", "Classic", "Gold", "Max")


@dataclass(frozen=True)
class CardSpec:
    bank: str
    card: str
    min_income: float | None
    min_age: int | None
    min_cibil: int | None
    employment: tuple[str, ...]  # ("any",) = any type
    relationship_required: bool

    def truth(self) -> dict[str, Any]:
        return {"min_income": self.min_income, "min_age": self.min_age, "min_cibil": self.min_cibil,
                "employment": self.employment, "relationship_required": self.relationship_required}


@dataclass
class Example:
    bank: str
    card: str
    split: str  # train | synthetic_test | real_thesis | real_thesis_hard | real_axis
    prompt: str
    target: str
    truth: dict[str, Any]
    style: str = ""
    perturbations: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["truth"] = {**self.truth, "employment": list(self.truth["employment"])}
        return d


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def indian(n: float) -> str:
    digits = str(int(round(n)))
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return ",".join(([head] if head else []) + groups) + "," + tail


def lakh(v: float) -> str:
    x = v / 100_000
    return f"{x:g}"


def amount_phrases(v: float, rng: random.Random, allow_words: bool = True) -> str:
    choices = [f"₹{indian(v)}", f"Rs {lakh(v)} lakh", f"₹{lakh(v)} lakh", f"INR {indian(v)}"]
    x = v / 100_000
    if allow_words and x == int(x) and int(x) in WORDS:
        choices.append(f"{WORDS[int(x)]} lakh rupees")
    return rng.choice(choices)


def monthly_ok(v: float) -> bool:
    return v % 12_000 == 0


def employment_text(types: tuple[str, ...]) -> str:
    names = {"salaried": "salaried employees", "self-employed": "self-employed professionals",
             "business owner": "business owners"}
    items = [names[t] for t in types]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# House styles: what each bank's documents look like. "heldout" is used only in tests.
STYLES: dict[str, dict[str, Any]] = {
    "SBI": {
        "headings": {"income": "Income Requirement", "age": "Age Requirement", "cibil": "Credit History Requirement",
                     "employment": "Employment Requirement", "relationship": "Relationship Requirement"},
        "income": ["Applicants should be earning a gross annual income of at least {amt}.",
                   "A minimum gross annual income of {amt} is expected from applicants."],
        "age": ["The primary cardholder must be at least {age} years of age at the time of application.",
                "Applicants must have completed {age} years of age."],
        "cibil": ["An applicant's bureau score should generally sit at {score} or higher.",
                  "A credit bureau score of {score} or above is required."],
    },
    "HDFC": {
        "headings": {"income": "Income Criteria", "age": "Age Criteria", "cibil": "Credit Score Criteria",
                     "employment": "Customer Profile", "relationship": "Existing Relationship"},
        "income": ["Salaried: Net Monthly Income > ₹{monthly}.",
                   "Income: Net Monthly Income greater than ₹{monthly}.",
                   "Income: ITR > {amt} per annum."],
        "age": ["Age: Min {age} years & Max 60 years.", "Age: Minimum {age} years, Maximum 65 years."],
        "cibil": ["CIBIL score of {score}+ is required.", "Applicants need a credit score of at least {score}."],
    },
    "AXIS": {
        "headings": {"income": "Eligibility – Income", "age": "Eligibility – Age", "cibil": "Eligibility – Credit Score",
                     "employment": "Eligibility – Occupation", "relationship": "Eligibility – Relationship"},
        "income": ["Minimum net annual income of {amt} and above.",
                   "Salaried applicants need a minimum net income of ₹{monthly} per month."],
        "age": ["Primary cardholder should be between the age of {age} and 70 years.",
                "Applicant must be between {age} and 70 years of age."],
        "cibil": ["A credit score of {score} or above is required for this card.",
                  "Applicants should have a CIBIL score of at least {score}."],
    },
    "heldout": {
        "headings": {"income": "Who can apply – earnings", "age": "Who can apply – age",
                     "cibil": "Who can apply – credit record", "employment": "Who can apply – work",
                     "relationship": "Who can apply – banking with us"},
        "income": ["To qualify, your yearly earnings must not fall below {amt}.",
                   "We accept applications only where take-home pay is {amt} a year or more."],
        "age": ["Applicants younger than {age} cannot apply.", "You need to be {age} or older."],
        "cibil": ["Your credit record should show a CIBIL of {score} or better.",
                  "We look for a bureau rating no lower than {score}."],
    },
}


def render_card(spec: CardSpec, style: str, rng: random.Random, perturb: bool,
                others: list[CardSpec]) -> tuple[list[dict[str, Any]], dict[str, str], list[str]]:
    """Rule chunks (one per stated criterion) for one card, the chunk ids per criterion, perturbations used."""
    s = STYLES[style]
    chunks: list[tuple[str, str]] = []
    used: list[str] = []

    if spec.min_income is not None:
        templates = [t for t in s["income"] if "{monthly}" not in t or monthly_ok(spec.min_income)]
        t = rng.choice(templates)
        text = t.format(amt=amount_phrases(spec.min_income, rng), monthly=indian(spec.min_income / 12))
        if perturb and rng.random() < 0.3:
            old = rng.choice([g * 100_000 for g in INCOME_GRID_LAKH if g * 100_000 != spec.min_income])
            text += f" This replaces the earlier limit of {amount_phrases(old, rng, False)}, which no longer applies."
            used.append("superseded")
        if perturb and rng.random() < 0.25 and others:
            o = rng.choice([c for c in others if c.min_income] or others)
            if o.min_income:
                text += f" (For comparison, the {o.card} asks for {amount_phrases(o.min_income, rng, False)}.)"
                used.append("other_card")
        if perturb and rng.random() < 0.25:
            text += f" The annual fee of ₹{indian(rng.choice([499, 500, 999, 2500, 10000]))} is waived on spends above ₹{indian(rng.choice([1, 2, 3, 5]) * 100_000)}."
            used.append("fee_spend")
        chunks.append(("income", text))
    elif perturb and rng.random() < 0.5:
        chunks.append(("income", "Applicants should have a stable source of income."))
        used.append("income_unstated")

    if spec.min_age is not None:
        chunks.append(("age", rng.choice(s["age"]).format(age=spec.min_age)))

    if spec.min_cibil is not None:
        chunks.append(("cibil", rng.choice(s["cibil"]).format(score=spec.min_cibil)))
    elif perturb and rng.random() < 0.4:
        chunks.append(("cibil", "No specific credit score is stated; applicants with a history of default are not approved."))
        used.append("cibil_unstated")

    if spec.employment == ("any",):
        emp = rng.choice(["Applicants of any employment type may apply.",
                          "Salaried, self-employed and business owners are all welcome."])
    else:
        emp = f"Open to {employment_text(spec.employment)}."
        excluded = [t for t in EMPLOYMENT_TYPES if t not in spec.employment]
        if perturb and excluded and rng.random() < 0.5:
            emp += f" {employment_text((rng.choice(excluded),)).capitalize()} are not eligible for this card."
            used.append("negated_employment")
    chunks.append(("employment", emp))

    if spec.relationship_required:
        rel = rng.choice([f"Applicants must already hold an account with {spec.bank}.",
                          f"Available only to existing {spec.bank} customers."])
        chunks.append(("relationship", rel))
    elif rng.random() < 0.6:
        rel = rng.choice([f"{spec.bank} does not insist on an existing account.",
                          "No existing banking relationship is required."])
        chunks.append(("relationship", rel))

    rng.shuffle(chunks)
    stem = f"{spec.bank.lower()}_cards"
    base = rng.randint(0, 40)
    out, sources = [], {}
    for i, (criterion, text) in enumerate(chunks):
        cid = f"{stem}#{base + i}"
        out.append({"chunk_id": cid, "bank": spec.bank, "card": spec.card,
                    "text": f"### {spec.card} – {s['headings'][criterion]}\n{text}"})
        stated = not (criterion == "income" and spec.min_income is None) and not (criterion == "cibil" and spec.min_cibil is None)
        if stated:
            sources[criterion] = cid
    return out, sources, used


def target_json(spec: CardSpec, sources: dict[str, str]) -> str:
    emp: Any = "any" if spec.employment == ("any",) else list(spec.employment)
    obj = {"bank": spec.bank, "card": spec.card,
           "requirements": {"min_annual_income_inr": int(spec.min_income) if spec.min_income is not None else None,
                            "min_age": spec.min_age, "min_cibil": spec.min_cibil, "allowed_employment": emp,
                            "existing_relationship_required": bool(spec.relationship_required)},
           "sources": {k: sources.get(k) for k in ("income", "age", "cibil", "employment", "relationship")}}
    return json.dumps([obj], ensure_ascii=False)


# ---------------------------------------------------------------------------
# Card specs
# ---------------------------------------------------------------------------
def random_spec(bank: str, rng: random.Random, used_names: set[str]) -> CardSpec:
    while True:
        name = f"{bank if bank != 'AXIS' else 'Axis'} {rng.choice(NAME_A)} {rng.choice(NAME_B)}"
        if name not in used_names:
            used_names.add(name)
            break
    if rng.random() < 0.15:
        employment: tuple[str, ...] = ("any",)
    else:
        k = rng.choice([1, 2, 2, 3])
        employment = tuple(sorted(rng.sample(EMPLOYMENT_TYPES, k)))
        if len(employment) == 3:
            employment = ("any",)
    return CardSpec(
        bank=bank, card=name,
        min_income=None if rng.random() < 0.08 else rng.choice(INCOME_GRID_LAKH) * 100_000,
        min_age=rng.choice([18, 18, 21, 21, 21, 23, 25]),
        min_cibil=None if rng.random() < 0.25 else rng.choice([650, 680, 700, 720, 730, 750, 760, 780, 800]),
        employment=employment,
        relationship_required=rng.random() < 0.2,
    )


def example_from_spec(spec: CardSpec, style: str, split: str, rng: random.Random, others: list[CardSpec],
                      perturb: bool = True) -> Example:
    chunks, sources, used = render_card(spec, style, rng, perturb, others)
    return Example(bank=spec.bank, card=spec.card, split=split, prompt=ExtractionPromptBuilder().build(chunks),
                   target=target_json(spec, sources), truth=spec.truth(), style=style, perturbations=used)


def corrections_examples(portal_dir: Path, bank: str) -> list[Example]:
    """Employee-approved values from the portal = the bank's own labelled data."""
    path = portal_dir / "banks" / bank / "corrections.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        t = row["target"]
        emp = t.get("allowed_employment")
        spec = CardSpec(bank=bank, card=row["card"], min_income=t.get("min_income"), min_age=t.get("min_age"),
                        min_cibil=t.get("min_cibil"),
                        employment=("any",) if emp in (None, "any") else tuple(sorted(emp)),
                        relationship_required=bool(t.get("relationship_required")))
        sources = {}
        for c in row["chunks"]:
            head = c["text"].splitlines()[0].lower()
            for k in ("income", "age", "employment", "relationship"):
                if k in head:
                    sources.setdefault(k, c["chunk_id"])
            if any(w in head for w in ("credit", "cibil", "score")):
                sources.setdefault("cibil", c["chunk_id"])
        out.append(Example(bank=bank, card=row["card"], split="train", prompt=ExtractionPromptBuilder().build(row["chunks"]),
                           target=target_json(spec, sources), truth=spec.truth(), style="portal_correction"))
    return out


def real_examples(rule_docs: Path, table_path: Path, split: str) -> list[Example]:
    """Thesis documents with the eligibility table as truth (never used for training)."""
    from src.domain.eligibility import EligibilityOracle

    chunks = [vars(c) for c in RuleChunker({"sbi_cards.md": "SBI", "hdfc_cards.md": "HDFC"}).chunk_directory(rule_docs)]
    truth = {(r.bank, r.card): r for r in EligibilityOracle(table_path).rules}
    out = []
    for (bank, card), rule in truth.items():
        own = [c for c in chunks if c["bank"] == bank and c["card"] == card]
        if not own:
            continue
        spec = CardSpec(bank, card, rule.min_income, rule.min_age, rule.min_cibil, tuple(rule.employment),
                        rule.relationship_required)
        out.append(Example(bank=bank, card=card, split=split, prompt=ExtractionPromptBuilder().build(own),
                           target="", truth=spec.truth(), style="thesis_document"))
    return out


def axis_examples(pdf: Path, table: Path) -> list[Example]:
    """The Axis public-pages document (converted exactly as the portal converts it)."""
    from src.portal.ingest import ingest
    from src.portal.samples import CARD_NAMES

    import tempfile

    result = ingest(pdf.read_bytes(), pdf.name, "AXIS", CARD_NAMES.split(", "))
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir) / "axis_cards.md"
        tmp.write_text(result.markdown, encoding="utf-8")
        chunks = [vars(c) for c in RuleChunker({tmp.name: "AXIS"}).chunk_file(tmp)]
    out = []
    with table.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            emp = tuple(e.strip() for e in row["employment"].split(",")) if row["employment"] else ("any",)
            spec = CardSpec("AXIS", row["card"], float(row["min_income"]) if row["min_income"] else None,
                            int(row["min_age"]) if row["min_age"] else None,
                            int(row["min_cibil"]) if row["min_cibil"] else None, emp,
                            row["relationship_required"].strip().upper() == "TRUE")
            own = [c for c in chunks if c["card"] == row["card"]]
            out.append(Example(bank="AXIS", card=row["card"], split="real_axis", prompt=ExtractionPromptBuilder().build(own),
                               target="", truth=spec.truth(), style="axis_document"))
    return out


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def build_datasets(seed: int = 7, train_cards_per_bank: int = 60, test_cards_per_bank: int = 12,
                   variants_per_card: int = 3, portal_dir: Path | None = Path("data/portal"),
                   repo: Path = Path(".")) -> dict[str, list[Example]]:
    """{'train/SBI': [...], 'train/HDFC': [...], 'train/AXIS': [...], 'synthetic_test/<BANK>': [...],
    'heldout_style/<BANK>': [...], 'real_thesis': [...], 'real_thesis_hard': [...], 'real_axis': [...]}"""
    rng = random.Random(seed)
    data: dict[str, list[Example]] = {}
    used: set[str] = set()
    for bank in BANKS:
        train_specs = [random_spec(bank, rng, used) for _ in range(train_cards_per_bank)]
        test_specs = [random_spec(bank, rng, used) for _ in range(test_cards_per_bank)]
        train = [example_from_spec(s, bank, "train", rng, train_specs) for s in train_specs for _ in range(variants_per_card)]
        if portal_dir is not None:
            train += corrections_examples(portal_dir, bank)
        rng.shuffle(train)
        data[f"train/{bank}"] = train
        data[f"synthetic_test/{bank}"] = [example_from_spec(s, bank, "synthetic_test", rng, test_specs) for s in test_specs]
        data[f"heldout_style/{bank}"] = [example_from_spec(s, "heldout", "heldout_style", rng, test_specs) for s in test_specs]
    data["real_thesis"] = real_examples(repo / "data/rule_docs", repo / "data/eligibility_table.csv", "real_thesis")
    data["real_thesis_hard"] = real_examples(repo / "data/rule_docs_hard", repo / "data/eligibility_table.csv", "real_thesis_hard")
    axis_pdf = repo / "data/portal_samples/axis_eligibility_2026-11.pdf"
    if axis_pdf.exists():
        data["real_axis"] = axis_examples(axis_pdf, repo / "data/axis/eligibility_axis.csv")
    poisoned = repo / "data/portal_samples/prompt_injection_test.pdf"
    if poisoned.exists():
        # same truth as the Axis page: a model that obeys the injected instruction gets ACE wrong
        data["injection_axis"] = [dataclass_replace(e, split="injection_axis")
                                  for e in axis_examples(poisoned, repo / "data/axis/eligibility_axis.csv")]
    return data


def save(data: dict[str, list[Example]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in data.items():
        path = out_dir / f"{name.replace('/', '__')}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for ex in rows:
                f.write(json.dumps(ex.to_json(), ensure_ascii=False) + "\n")


def summary(data: dict[str, list[Example]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, rows in data.items():
        perts: dict[str, int] = {}
        for ex in rows:
            for p in ex.perturbations:
                perts[p] = perts.get(p, 0) + 1
        out[name] = {"n": len(rows), "perturbations": perts}
    return out


def iter_all(data: dict[str, list[Example]], prefix: str) -> Iterable[Example]:
    for name, rows in data.items():
        if name.startswith(prefix):
            yield from rows


if __name__ == "__main__":
    ds = build_datasets()
    save(ds, Path("results_fl/data"))
    print(json.dumps(summary(ds), indent=1))
