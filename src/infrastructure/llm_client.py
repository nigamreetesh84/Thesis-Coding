"""Local LLM client adapters.

Implements the local, GPU-optional generation backends used by the RAG
recommender: a 4-bit quantized local model for grounded generation, an
"ungrounded" variant used as an ablation control (B1) that strips the
retrieved-rules context before generating, and GPU-free stub equivalents
so the whole system remains runnable end-to-end without a GPU. No paid API
is ever used anywhere in this module.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

from src.domain.eligibility import DEFAULT_ELIGIBILITY_TABLE_PATH, EligibilityOracle
from src.domain.entities import UserProfile

RETRIEVED_RULES_HEADING = "### RETRIEVED RULES"
APPLICANT_PROFILE_HEADING = "### APPLICANT PROFILE"


class LLMClient(Protocol):
    """Common interface implemented by every generation backend."""

    def generate(self, prompt: str) -> str:
        """Generate a raw text completion for the given prompt."""
        ...


class LocalLLMClient:
    """Local, 4-bit quantized instruction-tuned LLM (e.g. Mistral-7B-Instruct).

    The underlying `transformers` model/tokenizer are loaded lazily on the
    first call to `generate`, so importing this module or constructing the
    client never requires a GPU or downloads any weights up front.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        max_new_tokens: int = 512,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._pipeline: Any = None

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map="auto",
        )
        self._pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def generate(self, prompt: str) -> str:
        if self._pipeline is None:
            self._load()
        assert self._pipeline is not None
        outputs = self._pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            return_full_text=False,
        )
        return str(outputs[0]["generated_text"])


def strip_retrieved_rules(prompt: str) -> str:
    """Remove the '### RETRIEVED RULES' section from a grounded prompt."""
    pattern = re.compile(
        rf"{re.escape(RETRIEVED_RULES_HEADING)}.*?(?={re.escape(APPLICANT_PROFILE_HEADING)})",
        re.DOTALL,
    )
    return pattern.sub("", prompt)


class UngroundedLLMClient(LocalLLMClient):
    """Ablation control (B1): answers from parametric memory only.

    Strips the '### RETRIEVED RULES' section out of the prompt before
    generation, so the underlying model can no longer see the retrieved,
    grounded context -- isolating the effect of retrieval anchoring.
    """

    def generate(self, prompt: str) -> str:
        return super().generate(strip_retrieved_rules(prompt))


_CHUNK_LINE_RE = re.compile(r"chunk_id=(?P<chunk_id>\S+)\s+bank=(?P<bank>.+?)\s+card=(?P<card>.+)")
_PROFILE_FIELD_RE = re.compile(r"(?P<key>\w+)=(?P<value>.*)$", re.MULTILINE)


def _section(prompt: str, heading: str, next_heading: str) -> str:
    pattern = re.compile(
        rf"### {re.escape(heading)}\s*\n(?P<body>.*?)(?=### {re.escape(next_heading)}\s*$)",
        re.DOTALL | re.MULTILINE,
    )
    match = pattern.search(prompt)
    return match.group("body") if match else ""


def _parse_retrieved(prompt: str) -> dict[tuple[str, str], str]:
    """Extract {(bank, card): chunk_id} for every chunk cited in the prompt."""
    section = _section(prompt, "RETRIEVED RULES", "APPLICANT PROFILE")
    mapping: dict[tuple[str, str], str] = {}
    for match in _CHUNK_LINE_RE.finditer(section):
        mapping[(match.group("bank").strip(), match.group("card").strip())] = match.group("chunk_id").strip()
    return mapping


def _parse_profile(prompt: str) -> UserProfile:
    """Reconstruct the UserProfile encoded in the '### APPLICANT PROFILE' block."""
    section = _section(prompt, "APPLICANT PROFILE", "OUTPUT INSTRUCTIONS")
    if not section:
        raise ValueError("prompt does not contain an APPLICANT PROFILE section")
    fields = {m.group("key"): m.group("value").strip() for m in _PROFILE_FIELD_RE.finditer(section)}
    return UserProfile(
        annual_income=float(fields["annual_income"]),
        age=int(fields["age"]),
        cibil=int(fields["cibil"]),
        employment=fields["employment"],
        existing_relationship=fields["existing_relationship"].strip().lower() == "true",
        banks=[b.strip() for b in fields["banks"].split(",") if b.strip()],
    )


class StubLLM:
    """Deterministic, GPU-free stand-in used for tests, CI and demos.

    Grounded path: when the prompt still carries a '### RETRIEVED RULES'
    section, the stub looks up the TRUE eligibility outcome (via the
    domain oracle) for every requested card, citing a retrieved chunk when
    one is available -- simulating a well-grounded model that faithfully
    applies the cited rule. Note this only reads which cards were retrieved
    from the prompt text; it never reads rule_docs files.

    Ungrounded path: once '### RETRIEVED RULES' has been stripped (see
    'UngroundedStubLLM' below, the B1 ablation arm), there is no retrieved
    context to fall back on.
    """

    def __init__(self, table_path: str | Path = DEFAULT_ELIGIBILITY_TABLE_PATH) -> None:
        self._oracle = EligibilityOracle(table_path)

    def generate(self, prompt: str) -> str:
        try:
            profile = _parse_profile(prompt)
        except (ValueError, KeyError):
            return "[]"

        retrieved = _parse_retrieved(prompt)
        if retrieved:
            decisions = self._oracle.evaluate(profile)
            payload = [
                {
                    "bank": d.bank,
                    "card": d.card,
                    "recommended": d.recommended,
                    "reason": d.reason,
                    "cited_chunk_id": retrieved.get((d.bank, d.card)),
                }
                for d in decisions
            ]
            return json.dumps(payload)

        return self._ungrounded_guess(profile)

    def _ungrounded_guess(self, profile: UserProfile) -> str:
        """Crude, generic rule of thumb applied without any retrieved rule.

        Deliberately imprecise relative to the oracle -- simulating an LLM
        answering from vague parametric memory instead of a specific,
        retrieved threshold -- so grounded-vs-ungrounded ablations remain
        meaningful even when no GPU (and therefore no real LLM) is present.
        """
        payload = []
        for bank, card in self._oracle.cards_for_banks(profile.banks):
            naive_ok = profile.annual_income >= 500_000 and profile.cibil >= 700 and profile.age >= 21
            payload.append(
                {
                    "bank": bank,
                    "card": card,
                    "recommended": naive_ok,
                    "reason": (
                        "Approximate judgement from general knowledge of typical card "
                        "requirements, without access to this bank's specific published rule."
                    ),
                    "cited_chunk_id": None,
                }
            )
        return json.dumps(payload)


class UngroundedStubLLM(StubLLM):
    """GPU-free ungrounded counterpart to `StubLLM`, used by run_ablation.py.

    Strips the retrieved-rules section from the prompt (mirroring
    `UngroundedLLMClient`) before delegating to `StubLLM.generate`, which
    then falls back to the crude, generic heuristic.
    """

    def generate(self, prompt: str) -> str:
        return super().generate(strip_retrieved_rules(prompt))