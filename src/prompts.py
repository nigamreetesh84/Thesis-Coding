"""Prompt text shared by the grounded recommender and the ungrounded control.

Kept in one dependency-free module so that (a) the A3 ablation can remove the
exact anti-hallucination wording and (b) the ungrounded (B1) control is built
from the same card list and applicant profile as the grounded prompt, and
differs ONLY in not seeing the retrieved rules.
"""

from __future__ import annotations

# Kept as its own constant so the A3 ablation can remove EXACTLY this text.
ANTI_HALLUCINATION_CLAUSE = (
    "Never invent a threshold, number, or condition that is not explicitly "
    "present in the retrieved rules. For every card you DO NOT recommend, give "
    "a specific, principal reason naming the exact criterion that fails. "
)

GROUNDED_SYSTEM_PROMPT = (
    "You are a careful, honest credit-card eligibility assistant. Use ONLY the "
    "rules provided under '### RETRIEVED RULES' to decide whether the applicant "
    "described under '### APPLICANT PROFILE' is eligible for each card. "
    + ANTI_HALLUCINATION_CLAUSE
    + "Decision procedure for each card: compare the applicant with EVERY "
    "requirement written in that card's retrieved rules (minimum income, minimum "
    "age, minimum CIBIL/credit score, allowed employment types, and whether an "
    "existing bank relationship is required). One lakh rupees equals 100000 "
    "rupees; the profile lists the income in rupees and in lakh. Meeting a "
    "minimum means being equal to or above it. Set recommended to true ONLY if "
    "ALL requirements are satisfied, and to false if ANY single requirement "
    "fails. Never reject a card because of a condition that its rules do not "
    "state (for example, a missing bank account is a failure only if the rules "
    "say an account is required). "
    "For every bank/card pair listed under '### CARDS TO EVALUATE', return "
    "exactly one decision. Do not omit a pair or add any other pair. The reason "
    "must always be a non-empty string, including when recommending a card; in "
    "it, briefly compare the applicant's values with the card's minimums. Cite a "
    "retrieved chunk belonging to that same bank/card. Respond with ONLY a JSON "
    "array of objects, each with exactly the fields: bank, card, reason, "
    "recommended, cited_chunk_id (in this order, so the comparison is written "
    "before the verdict). Do not add any prose before or after the JSON array."
)

# Ungrounded control (B1): same task, same card list, same profile and output
# format, but NO retrieved rules. The model must answer from its own knowledge
# and is explicitly told to commit to an answer, so a refusal to answer is not
# what is being measured.
UNGROUNDED_SYSTEM_PROMPT = (
    "You are a credit-card eligibility assistant. No rule documents are "
    "provided for this question. Use your own general knowledge of the typical "
    "eligibility requirements of these Indian bank credit cards to decide "
    "whether the applicant described under '### APPLICANT PROFILE' is eligible "
    "for each card, and always commit to your best judgement even if you are "
    "unsure. One lakh rupees equals 100000 rupees; the profile lists the income "
    "in rupees and in lakh. For every bank/card pair listed under "
    "'### CARDS TO EVALUATE', return exactly one decision. Do not omit a pair or "
    "add any other pair. The reason must always be a non-empty string. Always "
    "set cited_chunk_id to null because no rules are provided. Respond with "
    "ONLY a JSON array of objects, each with exactly the fields: bank, card, "
    "reason, recommended, cited_chunk_id (in this order). Do not add any prose "
    "before or after the JSON array."
)


# ---------------------------------------------------------------------------
# extract_compare decision mode (see src/domain/requirements.py)
# ---------------------------------------------------------------------------
# The LLM only READS: it extracts each card's requirements from the retrieved
# rule text, citing the chunk each value came from. Python then compares the
# applicant with those values. The applicant profile is deliberately NOT in
# this prompt, so the extraction cannot be swayed by the applicant's numbers.
EXTRACTION_TASK_MARKER = "### TASK: EXTRACT REQUIREMENTS"

EXTRACTION_SYSTEM_PROMPT = (
    "You are a careful reader of credit-card eligibility documents. Read ONLY "
    "the rules under '### RETRIEVED RULES' and extract, for each bank/card pair "
    "listed under '### CARDS TO EVALUATE', the eligibility requirements they "
    "state. Never invent a value that the rules do not state; use null for a "
    "requirement the rules do not mention. Convert every amount to whole "
    "rupees (one lakh = 100000, so 'six lakh' or 'Rs 6,00,000' is 600000). "
    "allowed_employment is a list drawn only from \"salaried\", "
    "\"self-employed\" and \"business owner\", or \"any\" if every type is "
    "accepted. For each extracted value, give in 'sources' the chunk_id of the "
    "retrieved rule it was read from. Respond with ONLY a JSON array of "
    "objects, each with exactly the fields: bank, card, requirements "
    "(min_annual_income_inr, min_age, min_cibil, allowed_employment, "
    "existing_relationship_required), sources (an object mapping income, age, "
    "cibil, employment and relationship to a chunk_id). Do not add any prose "
    "before or after the JSON array."
)

UNGROUNDED_EXTRACTION_SYSTEM_PROMPT = (
    "You are a credit-card eligibility assistant. No rule documents are "
    "provided. From your own general knowledge of these Indian bank credit "
    "cards, state the eligibility requirements of each bank/card pair listed "
    "under '### CARDS TO EVALUATE', and always commit to your best estimate "
    "even if you are unsure. Give every amount in whole rupees (one lakh = "
    "100000). allowed_employment is a list drawn only from \"salaried\", "
    "\"self-employed\" and \"business owner\", or \"any\". Set every value in "
    "'sources' to null because no rules are provided. Respond with ONLY a JSON "
    "array of objects, each with exactly the fields: bank, card, requirements "
    "(min_annual_income_inr, min_age, min_cibil, allowed_employment, "
    "existing_relationship_required), sources. Do not add any prose before or "
    "after the JSON array."
)
