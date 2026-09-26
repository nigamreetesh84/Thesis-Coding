# Eligibility-Aware Credit Card Recommender

A local retrieval-augmented recommendation prototype for SBI and HDFC cards. Rule documents are chunked and indexed with Sentence-BERT and FAISS; the API uses a local LLM when configured, or `StubLLM` by default for GPU-free use.

## Architecture

- `src/domain/` contains framework-independent entities and the deterministic eligibility oracle. The oracle reads only `data/eligibility_table.csv`.
- `src/application/` contains the recommendation use case and its recommender protocol.
- `src/infrastructure/` contains the embedding, vector-store, LLM, and repository adapters.
- `src/model.py`, `src/pipeline.py`, and `src/api.py` form the outer composition and execution layer.
- Faithfulness validation is offline only. The live API does not apply a validation gate between model output and its response.

Each retrieved chunk contains one atomic rule for a card. The final decision composes several independent rules in the prompt, so decision-making is multi-hop even though retrieval operates over atomic chunks. This small rule set does not require GraphRAG-style retrieval machinery.

## Quickstart

Use Python 3.11 or newer. In PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.pipeline
uvicorn src.api:app --reload
```

Open `dashboard/index.html` in a browser. It calls the live API by default. Append `?mock=true` to the URL to opt into its mock demonstration responses.

## API

- `POST /recommend` accepts `annual_income`, `age`, `cibil`, `employment`, `existing_relationship`, and `banks`; it returns decisions, `latency_ms`, and `retrieved_k`.
- `GET /status` reports service status and configuration.
- `GET /telemetry` reports request count and latency percentiles.
- `GET /history?limit=` returns recent recommendation responses.

## Evaluation

Run the test suite and static checks:

```powershell
pytest -v
pip install mypy ruff
mypy src/
ruff check src/
```

The benchmark includes a fresh API startup and first recommendation as well as repeated requests; it asserts p95 latency at or below 750 ms and throughput above 5 requests per second.

```powershell
python -m evaluation.run_ablation --gold-set data/gold_test_set.jsonl --use-stub-llm --arms A2 A3 A5 A6
```

A1 runs as the primary grounded-versus-ungrounded comparison. A2, A3, A5, and A6 are independently selectable with `--arms`; A4 requires distinct GPU-backed model weights and is skipped under `StubLLM`.

Gold-set finalization requires independent annotation rows at `data/gold_set_independent_labels.jsonl`, keyed by `profile_index`, `bank`, and `card`, with a boolean `recommended` value. Without those annotations, the builder refuses to report kappa or write a finalized set. Judge-versus-human kappa likewise requires `data/human_labelled_explanations.jsonl`; it is reported as not computed when that file is absent.

Run the offline faithfulness and judge-validation script with:

```powershell
python -m evaluation.ragas_faithfulness
```

## Research Questions

| Question | Evaluation metric |
|---|---|
| RQ1: Is the grounded explanation faithful and accurate? | RAGAS faithfulness and gold-set classification precision, recall, F1, and accuracy |
| RQ2: Does retrieval surface the relevant rules? | Context precision and recall |
| RQ3: Does grounding reduce incorrect decisions? | Paired McNemar test, discordant counts, and relative error reduction |
| RQ4: Are explanations correct and traceable? | Profile-level label consistency and citation validity against retrieved chunk ID, bank, and card |

## Limitations

Evaluation uses synthetic profiles and curated rule documents for SBI and HDFC, not live bank data or production traffic. Results are an optimistic upper bound; the profile set may miss real-world phrasing and edge cases. Because the live system has no runtime validation gate, an ungrounded live response can reach the user. Broader bank coverage, live integrations, and production deployment remain future work.

## Repository Layout

```text
artifacts/faiss.index, artifacts/chunks.parquet
config/settings.py, config/logging.yaml
data/eligibility_table.csv
data/gold_test_set.jsonl
data/rule_docs/hdfc_cards.md, data/rule_docs/sbi_cards.md
dashboard/index.html, dashboard/react/RecommenderPanel.tsx
evaluation/build_gold_set.py, evaluation/metrics.py
evaluation/ragas_faithfulness.py, evaluation/run_ablation.py
src/api.py, src/model.py, src/pipeline.py
src/application/recommend_service.py
src/domain/eligibility.py, src/domain/entities.py
src/infrastructure/embedder.py, src/infrastructure/llm_client.py
src/infrastructure/repositories.py, src/infrastructure/vector_store.py
tests/conftest.py, tests/test_benchmark.py, tests/test_model.py, tests/test_pipeline.py
.env.example, pyproject.toml, requirements.txt
```

The gold-set fixture and reported evaluation metrics are synthetic and cover curated rules for two banks only; they do not establish performance on live banking data or production traffic. The repository does not include Docker deployment files.
