# VS Code Copilot — Verification & Gap-Closing Prompts
Thesis: Eligibility-Aware, Explainable Multi-Bank Credit Card Recommendation Using RAG

Run these in order. Do not batch fixes — one punch-list item per Prompt 2 cycle.

---

## Prompt 1 — Full Repository Audit (run first)

```
ROLE: You are a strict code auditor for a dissertation-grade RAG repository.
Do NOT trust file names or comments — open and read every file before making a claim.

TASK: Audit this repository against the specification below and produce a table:
| Item | Status (PASS/FAIL/PARTIAL) | Evidence (file:line) | Fix required |

CHECKS TO RUN, IN ORDER:

1. REPO STRUCTURE
   - Diff the actual file tree against this required layout: [paste the "REPOSITORY LAYOUT" block from Part B]
   - Flag any missing file, any extra/dead file, any empty/stub file, any file containing "TODO" or "pass  # placeholder".

2. ARCHITECTURAL RULES (non-negotiable)
   - Open src/domain/eligibility.py. Search its imports and body for any reference to data/rule_docs,
     vector_store, embedder, or faiss. It must ONLY read eligibility_table.csv. If it touches rule_docs, this is a FAIL — report exact line.
   - Confirm dependency direction: domain/ imports nothing from application/ or infrastructure/;
     application/ imports domain/ only; infrastructure/ may import both. Run a grep for "from src.infrastructure"
     inside src/domain/ and src/application/ — must return zero matches.
   - Confirm there is NO runtime validation/guard function called between LLM output and the API response
     in src/api.py or src/model.py (faithfulness checks must exist only in evaluation/, never in the live path).

3. FUNCTIONAL END-TO-END RUN
   - Actually execute: `python -m src.pipeline` and confirm artifacts/faiss.index and artifacts/chunks.parquet are created.
   - Load the FAISS index and assert vectors are 384-dim float32 with L2 norm ≈ 1.0 (print actual norms for 5 samples).
  - Start the API with StubLLM (no GPU) via `uvicorn src.api:app` and curl POST /recommend, GET /status,
     GET /telemetry, GET /history — print actual JSON responses, not assumed ones.
   - Open dashboard/index.html and confirm it calls /recommend and renders latency, retrieved_k, p95,
     and colour-coded cards with reason + cited_chunk_id (inspect the JS, don't assume).

4. TESTS & STATIC ANALYSIS
   - Run `pytest -v` — paste full output, not a summary.
   - Run `pytest tests/test_benchmark.py -v` — confirm p95 ≤ 750ms and throughput > 5 req/s with ACTUAL numbers.
   - Run `mypy src/` and `ruff check src/` — paste full output. Zero errors required, not "mostly clean."

5. EVALUATION RIGOUR (this is where most marks are lost)
   - Open evaluation/build_gold_set.py: confirm ~120 profiles, confirm cohens_kappa() exists and
     finalize() actually asserts kappa >= 0.80 (not just logs a warning).
   - Open evaluation/run_ablation.py: confirm McNemar test uses statsmodels with exact=True when
     discordant pairs < 25, and confirm it prints a HYPOTHESIS SUPPORTED/NOT SUPPORTED verdict at alpha=0.05.
   - Confirm A1–A6 ablations are each independently runnable via CLI flags — list the exact commands.
   - Open evaluation/ragas_faithfulness.py: confirm a SECOND kappa calculation exists — judge-model vs
     human-labelled subset (~20 items) — distinct from the gold-set annotation kappa in step above.
     This is the #1 examiner gap-closer; if missing, flag as CRITICAL.
   - Confirm all 4 RQs are printed with their mapped metric names in the evaluation output (RQ1..RQ4).

6. DATA REALISM
   - Open data/rule_docs/*.md and data/eligibility_table.csv side by side. Confirm the numeric thresholds
     are consistent but WORDED DIFFERENTLY (not copy-pasted numbers) — this proves comprehension over string-matching.

OUTPUT:
- The status table from above, fully filled in with evidence.
- A prioritized "TO REACH 95%" punch-list, ordered by: (a) architectural rule violations,
  (b) missing examiner gap-closers, (c) failing tests, (d) static analysis errors, (e) documentation gaps.
- For each punch-list item, the exact follow-up prompt I should give you to fix it.
```

---

## Prompt 2 — Targeted fix loop (use after Prompt 1 flags something)

```
Fix ONLY [paste one row from the punch-list].
Do not touch any other file. After the fix, re-run the relevant check from the audit
(pytest / mypy / curl / grep) and paste the actual output proving it now passes.
```

---

## Prompt 3 — Final grade-readiness check (run last)

```
Re-read Part C of the thesis reference document (the acceptance checklist table).
For every row, state PASS/FAIL with evidence. Then answer directly:
"Is this repository currently sufficient for a 95%+ distinction, or what specifically caps it below that,
and how many marks (roughly, against typical MSc rubric weightings for architecture/40%, evaluation-rigour/35%,
documentation/15%, code-quality/10%) is each remaining gap likely costing?"
```

---

## Common failure patterns to watch for
- **Judge-validation kappa** (item 16) skipped or faked with dummy numbers — this is the top examiner gap-closer named in the spec itself.
- **McNemar exact-test threshold logic** (`exact=True` only when discordant pairs < 25) implemented sloppily or hardcoded to always True/False.
- **Oracle/rule_docs leakage** — Copilot commonly has the oracle silently import from the retriever "for convenience," which breaks the core architectural claim.
- **Ablation A2/A5 not wired to a CLI** — often left as functions never actually exposed or callable.
- **Static analysis silently "passed"** — check for `# type: ignore` or `# noqa` added just to satisfy mypy/ruff instead of fixing the underlying issue.

## Execution checklist
- [ ] Run Prompt 1, capture full punch-list
- [ ] Fix items one at a time with Prompt 2, commit after each (`git add -A && git commit -m "fix: <item>"`)
- [ ] Re-run Prompt 1 in full every 3–4 fixes (a fix can silently break an earlier-passing check)
- [ ] Run Prompt 3 once punch-list is clear