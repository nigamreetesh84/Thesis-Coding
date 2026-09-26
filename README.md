

# Eligibility-Aware, Explainable Multi-Bank Credit Card Recommendation (RAG)

A reference implementation of an eligibility-aware, explainable credit-card recommendation system for two Indian banks (SBI, HDFC) built with Retrieval-Augmented Generation (RAG). It is fully local: local Sentence-BERT embeddings, local FAISS search, a local 4-bit quantized LLM (with a GPU-free stub for CI/demos), and a local open-source judge model for evaluation. **No paid API is used anywhere, including at evaluation time.**

## Architecture

Clean Architecture, dependencies point inward only:
domain/ framework-free entities + deterministic eligibility oracle
entities.py UserProfile, Decision
eligibility.py EligibilityOracle -- reads ONLY data/eligibility_table.csv
application/ use-case orchestration (no web/framework imports)
recommend_service.py
infrastructure/ FAISS, Sentence-BERT, LLM adapters behind interfaces
embedder.py, vector_store.py, llm_client.py, repositories.py
src/pipeline.py offline ETL: chunk rule docs -> embed -> persist FAISS + parquet
src/model.py Retriever, GroundedPromptBuilder, RAGRecommender, persist_config
src/api.py FastAPI service

text

`domain` depends on nothing. `application` depends on `domain`. `infrastructure` depends on both. `src/model.py`, `src/pipeline.py` and `src/api.py` wire everything together at the outermost layer.

### Non-negotiable architectural rules

1. **No runtime validation gate between the LLM and the user.** Correctness is driven by retrieval anchoring plus a grounded prompt. Faithfulness is measured only offline, at development time (`evaluation/`), never on live traffic. `src/model.py::RAGRecommender` returns the parsed LLM output directly.
2. **The oracle is isolated from the search corpus.** `src/domain/eligibility.py` reads only `data/eligibility_table.csv`. It never imports, opens, or reads anything under `data/rule_docs/`. That directory is the retriever's corpus only.
3. **No paid APIs.** Local Sentence-BERT (`all-MiniLM-L6-v2`, 384-dim), local FAISS, a local 4-bit quantized LLM (`Mistral-7B-Instruct` / `Llama-3.1-8B-Instruct` via bitsandbytes), and a local open-source judge model for evaluation.
4. **GPU-free path.** The whole system runs end-to-end via `StubLLM` / `UngroundedStubLLM` / `StubJudgeModel` when no GPU is present.

### Architecture note: multi-hop decisions over atomic rule chunks

Each retrieved chunk encodes exactly **one atomic eligibility rule** for one card (income, OR age, OR CIBIL, OR employment/relationship). The final recommendation **decision** is inherently **multi-hop**: it requires composing several atomic rules (income AND age AND CIBIL AND employment AND relationship) before a card can be labelled eligible. This composition happens in the **prompt** (the LLM reasons over several retrieved atomic chunks at once), not in the retrieval layer. This keeps chunking simple -- one rule per chunk, trivially easy to cite and audit -- and avoids the engineering overhead of GraphRAG-style multi-hop retrieval machinery, which would be unnecessary complexity for a handful of independent per-card predicates that never depend on each other's retrieval order. See the matching docstring in `src/model.py`.

## Quickstart

```bash
# 1. Install dependencies (Python 3.11)
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Build the FAISS index + chunk store from data/rule_docs/
python -m src.pipeline

# 3. Launch the API (StubLLM by default -- no GPU required)
uvicorn src.api:app --reload

# 4. Open the dashboard
#    Open dashboard/index.html directly in a browser (it targets
#    http://localhost:8000 automatically when opened via file:// or a
#    dev server on port 5500).

# 5. Build the gold evaluation set (~120 profiles)
python -m evaluation.build_gold_set

# 6. Run the ablation matrix (A1-A6) with McNemar + RQ summary
python -m evaluation.run_ablation

# 7. Run RAGAS faithfulness + judge-validation kappa
python -m evaluation.ragas_faithfulness

# 8. Run the full test suite
pytest

# 9. Static analysis
ruff check .
mypy .
To use a real local LLM instead of the stub (requires a CUDA GPU and the model weights), set USE_STUB_LLM=false in .env (copy from .env.example).

API surface
POST /recommend -- body: {annual_income, age, cibil, employment, existing_relationship, banks}; returns {decisions, latency_ms, retrieved_k}.

GET /status -- liveness + current configuration.

GET /telemetry -- {count, mean_latency_ms, p50_latency_ms, p95_latency_ms, p99_latency_ms} over the last 500 requests.

GET /history?limit= -- the most recent recommend results.

Research questions and mapped metrics
RQ	Question	Metric	Where
RQ1	Is the grounded explanation faithful and accurate?	RAGAS faithfulness + FactScore-style precision (evaluation/ragas_faithfulness.py) and gold-set classification accuracy (evaluation/run_ablation.py::rq1_classification_metrics)	run_ablation.py, ragas_faithfulness.py
RQ2	Does retrieval surface the right rules?	Context precision/recall (evaluation/metrics.py::context_precision_recall)	run_ablation.py::rq2_context_precision_recall
RQ3	Does grounding reduce hallucinated eligibility decisions?	Paired McNemar test, grounded vs ungrounded (B1)	run_ablation.py::a1_grounded_vs_ungrounded
RQ4	Are explanations correct AND traceable?	Consistent-with-expected-oracle-label AND cites a valid chunk_id	run_ablation.py::profile_correct, Decision.cited_chunk_id
All four RQ labels are printed alongside their metric by evaluation/run_ablation.py::print_rq_summary.

Ablation matrix (A1-A6)
Run via python -m evaluation.run_ablation --arms A1 A2 A3 A5 A6 (A4 requires a GPU and is skipped, with an explicit note, under StubLLM):

A1 grounded vs ungrounded (B1, parametric-memory-only) with McNemar significance, discordant counts, relative error reduction vs a 90% stretch target, and a HYPOTHESIS SUPPORTED/NOT SUPPORTED verdict.

A2 (strongest novel contribution) top-k sweep, k = 2, 4, 6, 8, 12.

A3 recommend-explain-cite prompt pattern on vs off.

A4 model selection, Mistral-7B-Instruct vs Llama-3.1-8B-Instruct (GPU only).

A5 (strongest novel contribution) chunking granularity: one rule per chunk vs one merged chunk per card.

A6 normal-phrasing vs reworded-phrasing profile accuracy.

Gold evaluation set
python -m evaluation.build_gold_set generates ~120 synthetic profiles, stratified across income bands, employment types (salaried / self-employed / business owner) and CIBIL ranges, and balanced across easy/borderline difficulty and normal/reworded phrasing (22 reworded profiles use everyday language, "8 lakh" instead of "800000", reordered conditions, and job-type synonyms). Each record stores {profile, profile_text, expected, variant, difficulty}, where expected is computed only by the isolated EligibilityOracle. The script asserts Cohen's kappa between two label passes is >= 0.80 (evaluation/build_gold_set.py::finalize).

Honesty / threats to validity
Figures are computed on a synthetic gold set against curated, representative (not live) rule documents for exactly two banks (SBI, HDFC). Treat all reported numbers as an optimistic upper bound, not a claim of generalisation to other banks, live documents, or production traffic (broader scaling is future work).

There is no runtime validation gate in the live system -- offline evaluation is the only place faithfulness is ever checked. This is a deliberate design choice (see architectural rule 1) with an inherent risk that a live, ungrounded hallucination would reach a user unchecked; the mitigation is retrieval anchoring + the grounded prompt, validated offline.

The synthetic-profile gold set has a usability ceiling: it cannot capture every real-world phrasing or edge case a live user might type.

The judge model used in evaluation/ragas_faithfulness.py is itself validated against a small (~20-example) human-labelled subset via Cohen's kappa (judge_validation_kappa) precisely because an unvalidated automated judge cannot otherwise be trusted as a fair referee.

Scope note
This is a dissertation-grade prototype. Explicitly out of scope (named as future work): production deployment at scale, live banking system integration, fraud/default risk prediction, use of real customer data, and demographic fairness auditing. Dockerfile / docker-compose.yml are provided only as an illustrative enterprise extension, not a requirement of the prototype.

Repository layout
text
config/settings.py, config/logging.yaml
data/rule_docs/sbi_cards.md, data/rule_docs/hdfc_cards.md
data/eligibility_table.csv
data/gold_test_set.jsonl        (generated by evaluation/build_gold_set.py)
artifacts/faiss.index, artifacts/chunks.parquet   (produced by src/pipeline.py)
src/domain/entities.py, src/domain/eligibility.py
src/application/recommend_service.py
src/infrastructure/embedder.py, vector_store.py, llm_client.py, repositories.py
src/pipeline.py, src/model.py, src/api.py
dashboard/index.html, dashboard/react/RecommenderPanel.tsx
evaluation/build_gold_set.py, run_ablation.py, ragas_faithfulness.py, metrics.py
notebooks/01_retrieval_tuning.ipynb
tests/test_oracle.py, test_pipeline.py, test_model.py, test_benchmark.py
README.md, requirements.txt, pyproject.toml
docker-compose.yml, Dockerfile, .env.example
text
proide downlable file of readme.md
I can't directly host or provide a downloadable file link, but you can create the README.md file yourself in seconds using either of these methods:

Option 1: Copy from a code block (easiest)
I've placed the full content in a single fenced code block below. Click the copy icon in the top-right of the block, then paste it into a new file named README.md in your project folder.

Option 2: One-line terminal command
Paste this into your terminal (in your project root) and it will write the file for you:

PowerShell (Windows):

powershell
notepad README.md
Then paste the content and save.

Bash / Git Bash / macOS / Linux:

bash
cat > README.md << 'EOF'
# Eligibility-Aware, Explainable Multi-Bank Credit Card Recommendation (RAG)
... (paste full content) ...
EOF
Full README.md content
markdown
# Eligibility-Aware, Explainable Multi-Bank Credit Card Recommendation (RAG)

A reference implementation of an eligibility-aware, explainable credit-card recommendation system for two Indian banks (SBI, HDFC) built with Retrieval-Augmented Generation (RAG). It is fully local: local Sentence-BERT embeddings, local FAISS search, a local 4-bit quantized LLM (with a GPU-free stub for CI/demos), and a local open-source judge model for evaluation. **No paid API is used anywhere, including at evaluation time.**

## Architecture

Clean Architecture, dependencies point inward only:

```
domain/          framework-free entities + deterministic eligibility oracle
  entities.py      UserProfile, Decision
  eligibility.py   EligibilityOracle -- reads ONLY data/eligibility_table.csv
application/     use-case orchestration (no web/framework imports)
  recommend_service.py
infrastructure/  FAISS, Sentence-BERT, LLM adapters behind interfaces
  embedder.py, vector_store.py, llm_client.py, repositories.py
src/pipeline.py  offline ETL: chunk rule docs -> embed -> persist FAISS + parquet
src/model.py     Retriever, GroundedPromptBuilder, RAGRecommender, persist_config
src/api.py       FastAPI service
```

`domain` depends on nothing. `application` depends on `domain`. `infrastructure` depends on both. `src/model.py`, `src/pipeline.py` and `src/api.py` wire everything together at the outermost layer.

### Non-negotiable architectural rules

1. **No runtime validation gate between the LLM and the user.** Correctness is driven by retrieval anchoring plus a grounded prompt. Faithfulness is measured only offline, at development time (`evaluation/`), never on live traffic. `src/model.py::RAGRecommender` returns the parsed LLM output directly.
2. **The oracle is isolated from the search corpus.** `src/domain/eligibility.py` reads only `data/eligibility_table.csv`. It never imports, opens, or reads anything under `data/rule_docs/`. That directory is the retriever's corpus only.
3. **No paid APIs.** Local Sentence-BERT (`all-MiniLM-L6-v2`, 384-dim), local FAISS, a local 4-bit quantized LLM (`Mistral-7B-Instruct` / `Llama-3.1-8B-Instruct` via bitsandbytes), and a local open-source judge model for evaluation.
4. **GPU-free path.** The whole system runs end-to-end via `StubLLM` / `UngroundedStubLLM` / `StubJudgeModel` when no GPU is present.

### Architecture note: multi-hop decisions over atomic rule chunks

Each retrieved chunk encodes exactly **one atomic eligibility rule** for one card (income, OR age, OR CIBIL, OR employment/relationship). The final recommendation **decision** is inherently **multi-hop**: it requires composing several atomic rules (income AND age AND CIBIL AND employment AND relationship) before a card can be labelled eligible. This composition happens in the **prompt** (the LLM reasons over several retrieved atomic chunks at once), not in the retrieval layer. This keeps chunking simple -- one rule per chunk, trivially easy to cite and audit -- and avoids the engineering overhead of GraphRAG-style multi-hop retrieval machinery, which would be unnecessary complexity for a handful of independent per-card predicates that never depend on each other's retrieval order. See the matching docstring in `src/model.py`.

## Quickstart

```bash
# 1. Install dependencies (Python 3.11)
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Build the FAISS index + chunk store from data/rule_docs/
python -m src.pipeline

# 3. Launch the API (StubLLM by default -- no GPU required)
uvicorn src.api:app --reload

# 4. Open the dashboard
#    Open dashboard/index.html directly in a browser (it targets
#    http://localhost:8000 automatically when opened via file:// or a
#    dev server on port 5500).

# 5. Build the gold evaluation set (~120 profiles)
python -m evaluation.build_gold_set

# 6. Run the ablation matrix (A1-A6) with McNemar + RQ summary
python -m evaluation.run_ablation

# 7. Run RAGAS faithfulness + judge-validation kappa
python -m evaluation.ragas_faithfulness

# 8. Run the full test suite
pytest

# 9. Static analysis
ruff check .
mypy .
```

To use a real local LLM instead of the stub (requires a CUDA GPU and the model weights), set `USE_STUB_LLM=false` in `.env` (copy from `.env.example`).

## API surface

- `POST /recommend` -- body: `{annual_income, age, cibil, employment, existing_relationship, banks}`; returns `{decisions, latency_ms, retrieved_k}`.
- `GET /status` -- liveness + current configuration.
- `GET /telemetry` -- `{count, mean_latency_ms, p50_latency_ms, p95_latency_ms, p99_latency_ms}` over the last 500 requests.
- `GET /history?limit=` -- the most recent recommend results.

## Research questions and mapped metrics

| RQ | Question | Metric | Where |
|----|----------|--------|-------|
| RQ1 | Is the grounded explanation faithful and accurate? | RAGAS faithfulness + FactScore-style precision (`evaluation/ragas_faithfulness.py`) and gold-set classification accuracy (`evaluation/run_ablation.py::rq1_classification_metrics`) | `run_ablation.py`, `ragas_faithfulness.py` |
| RQ2 | Does retrieval surface the right rules? | Context precision/recall (`evaluation/metrics.py::context_precision_recall`) | `run_ablation.py::rq2_context_precision_recall` |
| RQ3 | Does grounding reduce hallucinated eligibility decisions? | Paired McNemar test, grounded vs ungrounded (B1) | `run_ablation.py::a1_grounded_vs_ungrounded` |
| RQ4 | Are explanations correct AND traceable? | Consistent-with-expected-oracle-label AND cites a valid `chunk_id` | `run_ablation.py::profile_correct`, `Decision.cited_chunk_id` |

All four RQ labels are printed alongside their metric by `evaluation/run_ablation.py::print_rq_summary`.

## Ablation matrix (A1-A6)

Run via `python -m evaluation.run_ablation --arms A1 A2 A3 A5 A6` (A4 requires a GPU and is skipped, with an explicit note, under `StubLLM`):

- **A1** grounded vs ungrounded (B1, parametric-memory-only) with McNemar significance, discordant counts, relative error reduction vs a 90% stretch target, and a HYPOTHESIS SUPPORTED/NOT SUPPORTED verdict.
- **A2 (strongest novel contribution)** top-k sweep, k = 2, 4, 6, 8, 12.
- **A3** recommend-explain-cite prompt pattern on vs off.
- **A4** model selection, Mistral-7B-Instruct vs Llama-3.1-8B-Instruct (GPU only).
- **A5 (strongest novel contribution)** chunking granularity: one rule per chunk vs one merged chunk per card.
- **A6** normal-phrasing vs reworded-phrasing profile accuracy.

## Gold evaluation set

`python -m evaluation.build_gold_set` generates ~120 synthetic profiles, stratified across income bands, employment types (salaried / self-employed / business owner) and CIBIL ranges, and balanced across easy/borderline difficulty and normal/reworded phrasing (22 reworded profiles use everyday language, "8 lakh" instead of "800000", reordered conditions, and job-type synonyms). Each record stores `{profile, profile_text, expected, variant, difficulty}`, where `expected` is computed **only** by the isolated `EligibilityOracle`. The script asserts Cohen's kappa between two label passes is >= 0.80 (`evaluation/build_gold_set.py::finalize`).

## Honesty / threats to validity

- Figures are computed on a **synthetic** gold set against **curated, representative (not live)** rule documents for **exactly two banks** (SBI, HDFC). Treat all reported numbers as an **optimistic upper bound**, not a claim of generalisation to other banks, live documents, or production traffic (broader scaling is future work).
- There is **no runtime validation gate** in the live system -- offline evaluation is the only place faithfulness is ever checked. This is a deliberate design choice (see architectural rule 1) with an inherent risk that a live, ungrounded hallucination would reach a user unchecked; the mitigation is retrieval anchoring + the grounded prompt, validated offline.
- The synthetic-profile gold set has a **usability ceiling**: it cannot capture every real-world phrasing or edge case a live user might type.
- The judge model used in `evaluation/ragas_faithfulness.py` is itself validated against a small (~20-example) human-labelled subset via Cohen's kappa (`judge_validation_kappa`) precisely because an unvalidated automated judge cannot otherwise be trusted as a fair referee.

## Scope note

This is a dissertation-grade **prototype**. Explicitly out of scope (named as future work): production deployment at scale, live banking system integration, fraud/default risk prediction, use of real customer data, and demographic fairness auditing. `Dockerfile` / `docker-compose.yml` are provided only as an illustrative enterprise extension, not a requirement of the prototype.

## Repository layout

```
config/settings.py, config/logging.yaml
data/rule_docs/sbi_cards.md, data/rule_docs/hdfc_cards.md
data/eligibility_table.csv
data/gold_test_set.jsonl        (generated by evaluation/build_gold_set.py)
artifacts/faiss.index, artifacts/chunks.parquet   (produced by src/pipeline.py)
src/domain/entities.py, src/domain/eligibility.py
src/application/recommend_service.py
src/infrastructure/embedder.py, vector_store.py, llm_client.py, repositories.py
src/pipeline.py, src/model.py, src/api.py
dashboard/index.html, dashboard/react/RecommenderPanel.tsx
evaluation/build_gold_set.py, run_ablation.py, ragas_faithfulness.py, metrics.py
notebooks/01_retrieval_tuning.ipynb
tests/test_oracle.py, test_pipeline.py, test_model.py, test_benchmark.py
README.md, requirements.txt, pyproject.toml
docker-compose.yml, Dockerfile, .env.example
```
Option 3: Ask me to output as a single "copy-paste" block
If you'd prefer, tell me your OS (Windows/macOS/Linux) and I'll give you a single heredoc command you can paste directly into your terminal to write README.md to disk without needing to open a text editor.

Just to set expectations: as a chat assistant, I can't generate a hosted download link (like a .md file attachment) for you — but one of the three options above will get the file onto your machine in under 30 seconds.