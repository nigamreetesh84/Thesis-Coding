"""Local LLM client adapters.

Implements the local, GPU-optional generation backends used by the RAG
recommender: a 4-bit quantized local model for grounded generation, an
"ungrounded" variant used as an ablation control (B1) that strips the
retrieved-rules context before generating, and GPU-free stub equivalents
so the whole system remains runnable end-to-end without a GPU. No paid API
is ever used anywhere in this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, ClassVar, Protocol

from src.domain.eligibility import DEFAULT_ELIGIBILITY_TABLE_PATH, EligibilityOracle
from src.domain.entities import UserProfile
from src.prompts import (
    EXTRACTION_TASK_MARKER,
    UNGROUNDED_EXTRACTION_SYSTEM_PROMPT,
    UNGROUNDED_SYSTEM_PROMPT,
)

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

    _shared_model_name: ClassVar[str | None] = None
    _shared_pipeline: ClassVar[Any] = None
    _cache: ClassVar[dict[str, str] | None] = None
    _cache_path: ClassVar[Path | None] = None

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        max_new_tokens: int = 1536,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._pipeline: Any = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        if LocalLLMClient._shared_model_name == self.model_name and LocalLLMClient._shared_pipeline is not None:
            self._pipeline = LocalLLMClient._shared_pipeline
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

        if LocalLLMClient._shared_pipeline is not None:
            LocalLLMClient._shared_pipeline = None
            LocalLLMClient._shared_model_name = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
        LocalLLMClient._shared_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        LocalLLMClient._shared_model_name = self.model_name
        self._pipeline = LocalLLMClient._shared_pipeline

    def _format_prompt(self, prompt: str) -> str:
        assert self._pipeline is not None
        tokenizer = self._pipeline.tokenizer
        if getattr(tokenizer, "chat_template", None):
            system_marker = "SYSTEM:\n"
            messages = [{"role": "user", "content": prompt}]
            if prompt.startswith(system_marker):
                body_start = prompt.find("\n\n### ", len(system_marker))
                if body_start != -1:
                    messages = [
                        {"role": "system", "content": prompt[len(system_marker):body_start].strip()},
                        {"role": "user", "content": prompt[body_start:].strip()},
                    ]
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:  # e.g. templates without a system role (Mistral-7B-Instruct-v0.2)
                merged = [{"role": "user", "content": prompt}]
                return tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)
        if "mistral" in self.model_name.lower():
            return f"<s>[INST] {prompt.strip()} [/INST]"
        return prompt

    # ---- persistent generation cache -------------------------------------
    # Decoding is greedy (do_sample=False), so the same (model, prompt) always
    # yields the same text. Caching it on disk means the ablation arms that
    # re-run an identical prompt (A1, RQ1, RQ4, A2@k>=4, A3-on, A5-fine, A6)
    # generate once, and a run interrupted by a Colab disconnect resumes where
    # it stopped. Set LLM_CACHE=0 to disable.
    @classmethod
    def _cache_enabled(cls) -> bool:
        return os.environ.get("LLM_CACHE", "1").lower() not in {"0", "false", "no"}

    @classmethod
    def _load_cache(cls) -> dict[str, str]:
        path = Path(os.environ.get("LLM_CACHE_PATH", "artifacts/llm_cache.jsonl"))
        if cls._cache is not None and cls._cache_path == path:
            return cls._cache
        cache: dict[str, str] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    cache[row["key"]] = row["text"]
                except (json.JSONDecodeError, KeyError):
                    continue  # ignore a partially written last line
        cls._cache, cls._cache_path = cache, path
        return cache

    def _cache_key(self, formatted_prompt: str) -> str:
        raw = f"{self.model_name}|{self.max_new_tokens}|{formatted_prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _prepare(self, prompt: str) -> str:
        """Hook for subclasses to rewrite the prompt before formatting."""
        return prompt

    def _run_pipeline(self, formatted: list[str]) -> list[str]:
        """Greedy-generate for already formatted prompts, batched with a safe fallback."""
        assert self._pipeline is not None
        tokenizer = self._pipeline.tokenizer
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        batch_size = max(1, int(os.environ.get("LLM_BATCH_SIZE", "8")))
        kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens, "do_sample": False, "return_full_text": False}

        def one(text: str) -> str:
            out = self._pipeline(text, **kwargs)
            return str(out[0]["generated_text"])

        if len(formatted) == 1 or batch_size == 1:
            return [one(t) for t in formatted]
        try:
            outputs = self._pipeline(formatted, batch_size=batch_size, **kwargs)
            texts = [str(o[0]["generated_text"] if isinstance(o, list) else o["generated_text"]) for o in outputs]
            if len(texts) == len(formatted):
                return texts
        except Exception:  # most commonly CUDA out-of-memory: fall back to one prompt at a time
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        return [one(t) for t in formatted]

    def _generate_many(self, prompts: list[str]) -> list[str]:
        if self._pipeline is None:
            self._load()
        assert self._pipeline is not None
        formatted = [self._format_prompt(self._prepare(p)) for p in prompts]
        keys = [self._cache_key(f) for f in formatted]
        use_cache = self._cache_enabled()
        cache = self._load_cache() if use_cache else {}

        results: dict[int, str] = {i: cache[k] for i, k in enumerate(keys) if k in cache}
        missing = [i for i in range(len(formatted)) if i not in results]
        if missing:
            for i, text in zip(missing, self._run_pipeline([formatted[i] for i in missing]), strict=True):
                results[i] = text
                if use_cache:
                    cache[keys[i]] = text
                    path = self._cache_path
                    if path is not None:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps({"key": keys[i], "text": text}) + "\n")
        return [results[i] for i in range(len(formatted))]

    def generate(self, prompt: str) -> str:
        return self._generate_many([prompt])[0]

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate for several prompts at once (used for per-card prompting)."""
        return self._generate_many(prompts)


_RULES_BLOCK_RE = re.compile(
    rf"^{re.escape(RETRIEVED_RULES_HEADING)}[ \t]*\n.*?(?=^### CARDS TO EVALUATE[ \t]*$|^{re.escape(APPLICANT_PROFILE_HEADING)}[ \t]*$)",
    re.DOTALL | re.MULTILINE,
)
_SYSTEM_BLOCK_RE = re.compile(r"\ASYSTEM:\n.*?(?=\n\n### )", re.DOTALL)


def strip_retrieved_rules(prompt: str) -> str:
    """Remove ONLY the '### RETRIEVED RULES' section from a grounded prompt.

    The '### CARDS TO EVALUATE' list and the applicant profile are kept, so the
    ungrounded control is asked the same question about the same cards.
    """
    return _RULES_BLOCK_RE.sub("", prompt)


def to_ungrounded_prompt(prompt: str) -> str:
    """Build the B1 control prompt: no rules, and a system prompt that does not
    tell the model to rely on rules that are not there."""
    stripped = strip_retrieved_rules(prompt)
    system = UNGROUNDED_EXTRACTION_SYSTEM_PROMPT if EXTRACTION_TASK_MARKER in prompt else UNGROUNDED_SYSTEM_PROMPT
    return _SYSTEM_BLOCK_RE.sub(lambda _: "SYSTEM:\n" + system, stripped, count=1)


class UngroundedLLMClient(LocalLLMClient):
    """Ablation control (B1): answers from parametric memory only.

    Removes the retrieved rules (keeping the card list and profile) and swaps
    in a system prompt that asks for a best-effort answer from general
    knowledge, isolating the effect of retrieval anchoring.
    """

    def _prepare(self, prompt: str) -> str:
        return to_ungrounded_prompt(prompt)


_CHUNK_LINE_RE = re.compile(r"chunk_id=(?P<chunk_id>\S+)\s+bank=(?P<bank>.+?)\s+card=(?P<card>.+)")
_PROFILE_FIELD_RE = re.compile(r"(?P<key>\w+)=(?P<value>.*)$", re.MULTILINE)
_CARD_LINE_RE = re.compile(r"^- bank=(.+?); card=(.+)$", re.MULTILINE)
# Words in a rule chunk's heading that identify which criterion it states.
_CRITERION_WORDS: dict[str, tuple[str, ...]] = {
    "income": ("income",),
    "age": ("age",),
    "cibil": ("credit score", "credit history", "cibil", "score"),
    "employment": ("employment",),
    "relationship": ("relationship",),
}


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
        if EXTRACTION_TASK_MARKER in prompt:
            return self._extract(prompt)
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

    def _extract(self, prompt: str) -> str:
        """Extraction prompt (extract_compare mode).

        Grounded: report the table's requirements for each listed card, citing
        the retrieved chunk whose heading names the criterion (simulating a
        perfect reader). Ungrounded: crude generic guesses, like
        `_ungrounded_guess`, so the A1 comparison stays meaningful offline.
        """
        cards = [(m.group(1).strip(), m.group(2).strip()) for m in _CARD_LINE_RE.finditer(prompt)]
        chunks = [(m.group("chunk_id"), m.group("bank").strip(), m.group("card").strip(), m.end())
                  for m in _CHUNK_LINE_RE.finditer(prompt)]
        grounded = RETRIEVED_RULES_HEADING in prompt and bool(chunks)
        by_card = {(r.bank, r.card): r for r in self._oracle.rules}
        payload = []
        for bank, card in cards:
            rule = by_card.get((bank, card))
            if rule is None:
                continue
            if not grounded:
                payload.append({"bank": bank, "card": card, "requirements": {
                    "min_annual_income_inr": 500000, "min_age": 21, "min_cibil": 700,
                    "allowed_employment": "any", "existing_relationship_required": False}, "sources": {}})
                continue
            sources: dict[str, str] = {}
            for chunk_id, c_bank, c_card, end in chunks:
                if (c_bank, c_card) != (bank, card):
                    continue
                heading = prompt[end:prompt.find("\n", end + 1)].lower()
                for criterion, words in _CRITERION_WORDS.items():
                    if any(w in heading for w in words):
                        sources.setdefault(criterion, chunk_id)
            employment = "any" if "any" in rule.employment else list(rule.employment)
            payload.append({"bank": bank, "card": card, "requirements": {
                "min_annual_income_inr": int(rule.min_income), "min_age": rule.min_age, "min_cibil": rule.min_cibil,
                "allowed_employment": employment, "existing_relationship_required": rule.relationship_required},
                "sources": sources})
        return json.dumps(payload)

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
        return super().generate(to_ungrounded_prompt(prompt))