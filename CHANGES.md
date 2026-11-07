# Changes in this version

Root causes found from the Colab run output, and what was changed.

| File | Change | Why |
|---|---|---|
| `evaluation/run_ablation.py` (RQ2) | `retriever.retrieve(...)` -> `retriever.retrieve_for_profile(...)` | RQ2 used plain top-4 retrieval, so recall was 4/25 = 0.160 regardless of quality. It now measures the retrieval the system actually uses. |
| `evaluation/run_ablation.py` (A3) | Removes `ANTI_HALLUCINATION_CLAUSE` (exact text, asserted) | The old replace string never matched the prompt (`do NOT` vs `DO NOT`), so prompt-on and prompt-off were identical. |
| `evaluation/run_ablation.py` (A6) | Reworded/normal profile text is now what the LLM sees (real LLM only) | `profile_text` was never used, so A6 compared identical inputs. |
| `evaluation/run_ablation.py` | `--limit N` (evenly spaced subsample), tqdm progress bar, `profile_text` loaded in `GoldRecord` | Full run is hours on a T4; makes staged runs possible. |
| `src/model.py` | New `SYSTEM_PROMPT` (explicit all-criteria procedure, lakh conversion, reason before verdict); `annual_income_in_lakh` line in profile block | The 3B model rejected eligible cards (recall 0.167); rules are written in words ("six lakh") while the profile is in rupees. |
| `src/model.py` | `parse_llm_response` skips malformed items, handles `null` reason / string booleans | One bad item used to discard all 8 decisions; `null` became the text "None". |
| `src/infrastructure/llm_client.py` | `max_new_tokens` 1536; persistent greedy-generation cache (`LLM_CACHE_PATH`, `LLM_CACHE=0` disables) | Identical prompts recur across arms (A1, RQ1, RQ4, A2 k>=4, A3-on, A5, A6). Cuts unique generations by ~3x and makes runs resumable after a disconnect. |
| `tests/test_fixes.py` | 6 regression tests | Parser, prompt, A3 clause, A6 builder, gold `profile_text`, cache. |

## Round 2 (after the 7B real-model run)

| File | Change | Why |
|---|---|---|
| `src/model.py` (`RAGRecommender`) | **Per-card prompting is now the default** (one short grounded prompt per card; `per_card=False` or `RAG_PER_CARD=0` restores the joint prompt) | Real 7B run: profile accuracy 0.175, per-card recall 0.365. The model mis-compared numbers (e.g. "age 30 is below 21") when judging 8 cards in one JSON answer. |
| `evaluation/run_ablation.py` | `--joint-prompt` flag | Lets you report joint vs per-card prompting as an extra comparison. |
| `tests/test_fixes.py` | per-card test (20 tests total) | Checks one prompt per card and that extra decisions are dropped. |

Note for the write-up: each card has only 3-4 rule chunks, so `top_k >= 4` retrieves identical context. A2 therefore only has power at k = 2 and 3.

## Round 3 (after the per-card run)

| File | Change | Why |
|---|---|---|
| `evaluation/run_ablation.py` (RQ4) | Now goes through `RAGRecommender` (per-card) | RQ4 still built one joint prompt, so its label-consistency (0.175) was the old joint accuracy, not the system's. |
| `evaluation/run_ablation.py` (A6) | Balanced sample of normal and reworded profiles (`a6_balanced_sample`) | Only 22 of 120 gold profiles are reworded; an evenly spaced subsample of 20 contained 1, so A6 was meaningless. |
| `tests/test_fixes.py` | 2 more tests (22 total) | A6 sampling; RQ4 issues one prompt per card. |

Prompts and retrieval were deliberately NOT changed in this round, so cached generations from the previous run are reused (re-running is fast except A6).

## Round 4: full audit (final version)

| Area | Finding | Fix |
|---|---|---|
| A1 / RQ3 (critical) | The ungrounded control prompt had its `### CARDS TO EVALUATE` list removed together with the rules, and still told the model to "use ONLY the retrieved rules". It could never be correct (0.0 accuracy in every run), so the McNemar significance was an artifact. | `strip_retrieved_rules` removes only the rules; new `UNGROUNDED_SYSTEM_PROMPT` (best-effort from general knowledge); `src/prompts.py` holds all prompt text. |
| Diagnostics | Parse failures/refusals silently counted as "wrong". | `cards_returned_rate` for grounded/ungrounded/RQ1; per-profile `predictions_*.jsonl`; per-card confusion counts; majority-class baseline. |
| Speed | 8 sequential generations per profile. | Per-card prompts are sent as one batched pipeline call (`LLM_BATCH_SIZE`, automatic sequential fallback on error); greedy generations cached on disk. |
| Mistral-style chat templates | A system role raises an error. | Falls back to a single user message. |
| A2 | k >= 4 identical (3-4 chunks per card). | Default sweep k = 1..4. |
| A4 | Hard-coded gated models. | `--a4-models` (default Qwen2.5-3B vs 7B). |
| RQ1 label | Said "RAGAS" although only gold accuracy is computed. | Relabelled. |
| Data | Truncated sentence in `hdfc_cards.md` ("already have a HDFC."). | Fixed. |
| Dependencies | `ragas` is never imported but caused langchain conflicts. | Removed; `tqdm` added. |
| Reproducibility | Results only printed. | `results/results.json` (merged across runs) with model, split, prompting mode and timings; `--split`; embedder-fallback guard. |
| Integrity | No automated check that docs, table, gold set and prompts agree. | `python -m evaluation.verify_setup` + tests (32 total). |

## Round 5 (6 Nov 2026): diagnosis from the 14 Oct predictions, and fixes

### What the predictions showed
* **The grounded system's errors are number comparisons, not retrieval.** Of the grounded false negatives, 29/33 (test) and 47/49 (dev) have a reason that says a criterion the applicant PASSES has failed, e.g. "income of Rs 28,04,342 is below the required Rs 6,00,000", "age 56 is below the minimum of 21", "20 years old, which is below the minimum age of 18". 19 of the 24 wrong test profiles are wrong ONLY because of such comparisons.
* **The ungrounded arm's 0.0 is partly a parsing artefact.** It returned 72% (test) / 67.5% (dev) of cards; per-card replies whose card name differed from the requested one, or that were a single JSON object, were dropped.
* **A6 is confounded by units.** "Normal" free-text profiles state income in rupees, "reworded" ones in lakh (the unit of the rule documents), so reworded > normal reflects number formatting, not robustness to wording.
* **40 dev profiles were never evaluated** in A1/RQ1-RQ4 (`--split dev --limit 40` took every second profile). 12 of them appeared in the A6 sample.

### Changes
| File | Change | Why |
|---|---|---|
| `src/model.py` | Tolerant JSON extraction (`extract_json_items`): single object, code fences, prose, wrapper objects, trailing commas, brackets inside strings | Replies that were valid in substance were discarded |
| `src/model.py` | `coerce_recommended`: "Yes"/"Eligible" -> True; unreadable verdicts are skipped instead of guessed | Previously "Yes" became False |
| `src/model.py` | Per-card name recovery (`_own_decision`), counted in `last_name_recoveries` | Ungrounded replies naming the card differently were silently dropped |
| `src/domain/requirements.py` (new) | Deterministic comparison of the applicant with requirements EXTRACTED by the LLM; specific counterfactual reason citing the failing criterion's chunk | Fixes the comparison errors without reading the eligibility table at inference |
| `src/model.py`, `src/prompts.py` | `decision_mode="extract_compare"` (`--decision-mode`, `RAG_DECISION_MODE`); extraction prompt holds no profile and lists chunks in a fixed order | LLM reads, code compares. Default stays `llm`, so the 14 Oct results reproduce exactly |
| `src/infrastructure/llm_client.py` | Ungrounded control and stub understand extraction prompts | Keeps A1 a like-for-like comparison in both modes |
| `evaluation/run_ablation.py` | **A7** arm: llm vs extract_compare vs ungrounded extract_compare, with requirement-extraction accuracy (threshold hallucination rate for the ungrounded arm) | Measures the fix and gives RQ3 a direct hallucination measure |
| `evaluation/run_ablation.py` | A1: Wilson 95% CIs, accuracy difference, exact-test flag, card accuracy on returned cards, name recoveries; relative error reduction flagged when baseline is 0 | Separates wrong knowledge from format failure; honest effect sizes |
| `evaluation/run_ablation.py` | RQ2: dense precision/recall@k WITHOUT the card filter | The old proxy is 1.0 by construction |
| `evaluation/run_ablation.py` | RQ4: `decline_citation_governs_rate` - does a decline cite a chunk stating a criterion the applicant actually fails? | Reasoning-level check (first marker, F5) |
| `evaluation/run_ablation.py` | `--split holdout` (the 40 never-evaluated dev profiles) and `--split dev40` | Clean final estimate |
| `evaluation/make_report.py` | Reports all of the above | |
| `tests/test_robustness.py` | Regression tests for every change | |

Parser and name-recovery changes do not alter prompts, so re-running A1 on test/dev40 with the Drive cache replays the 14 Oct generations and re-scores them in minutes. The extract_compare mode issues new (short) prompts; because the extraction prompt contains no profile, each card is read about once and cached, so it is fast.

### Round 5.1 (6 Nov 2026, after the round-5 Colab run)
| File | Change | Why |
|---|---|---|
| `src/domain/requirements.py` | `criterion_for_key`: source labels such as `min_annual_income_inr` map to `income`, etc. | Qwen2.5-7B labelled sources by field name, so declines got no citation (holdout citation validity 0.225, decline-governs 0.000). Decisions unaffected: re-scoring the saved extractions changes 0 of 320 verdicts per split and gives citation validity 1.000, decline-governs 248/248 (holdout) and 246/246 (test) |
| `src/model.py` | Card-name matching treats `+`, `Plus` and its absence as equal | Ungrounded extraction never returned HDFC MoneyBack+ (cards returned 0.875); likely a name variant |
| `tests/test_robustness.py` | 2 regression tests (64 total) | |

### Round 5.2 (6 Nov 2026, from the raw llm_cache.jsonl)
* **Ungrounded arm's missing cards explained:** 90 of 320 ungrounded per-card replies contain a reason but NO `recommended` field (230/320 = the 0.719 cards-returned rate). This is the model declining to commit without documents, not a parsing failure; it stays scored as "no answer" and is reported alongside card accuracy on answered cards (0.652).
* `src/model.py`: `_loads_lenient` also accepts Python-style `True`/`False`/`None` outside strings (the ungrounded MoneyBack+ extraction used `False`). Every cached reply now parses. Ungrounded extraction accuracy with that card included: 10/40 fields (0.25).
* 1 regression test (65 total).

### Round 5.3 (6 Nov 2026, after the 5.2 re-scoring run)
* **Reproducibility finding.** The 5.2 run replayed the 14 Oct cached generations exactly (320/320 reasons, 40/40 verdicts). The 6 Nov round-5 run had regenerated them instead; those fresh greedy generations differed in 131/320 reasons and 2/40 profile verdicts (LLM-decides test accuracy 0.425 vs 0.400; holdout 0.400 vs 0.375). Batched 4-bit generation on a GPU is not bit-reproducible; `llm_cache.jsonl` is the reproducibility record and must be archived with the code. The extract_compare verdicts were identical in every run.
* `evaluation/run_ablation.py`: McNemar always uses the exact binomial test (previously chi-square at >= 25 discordant pairs, which mixed two tests across arms).
* `evaluation/make_report.py`: reports the exact p-value recomputed from b and c, so results from earlier runs are reported with the same test without re-running.
* 1 regression test (66 total).

### Round 5.4 (6 Nov 2026): fail-closed guard and A8 harder documents
| File | Change | Why |
|---|---|---|
| `src/domain/requirements.py`, `src/model.py` | Fail-closed guard (`decide_guarded`, default on, `RAG_GUARD=0` disables): if the retrieved rules state an income/age/CIBIL requirement that the model did not read, or a value read is implausible, the card is NOT recommended and is flagged `needs_review` with "check the bank's criteria before applying" | A missed threshold was read as "no requirement", turning a reading error into a false recommendation (the costliest error for an applicant: a rejection and a hard CIBIL enquiry). Replayed over all 1,280 saved extract_compare decisions: fired 0 times, changed 0 verdicts |
| `src/domain/entities.py`, `src/api.py` | `Decision.needs_review` / `DecisionOut.needs_review` (default False) | Expose the flag to the API and evaluation |
| `evaluation/run_ablation.py`, `evaluation/make_report.py` | Count and report needs-review decisions | |
| `data/rule_docs_hard/` (new, DRAFT for review) | A8: same cards, sections and thresholds, harder wording: monthly income, numbers in words, superseded old thresholds, other cards' thresholds mentioned, negated conditions, fee/spend figures that are not conditions | The 40/40 extraction was measured on one clean, self-authored document set (second-marker point S5) |
| `data/A8_HARD_DOCS_NOTES.md` | Trap-by-trap answer key (not indexed) | |
| `tests/test_robustness.py` | 5 tests (71 total) | Guard failure modes; hard documents keep the same structure |

### Round 5.5 (7 Nov 2026, after the A8 run): DESIGN FROZEN
* A8 (holdout, harder documents): extraction 40/40 fields; extract+compare 1.000; LLM decides 0.600; ungrounded extract 0.125.
* `src/domain/requirements.py`: a source given as several ids ("sbi_cards#7,sbi_cards#8") or a list keeps the first id. Re-scoring the saved A8 extractions: 0/320 verdicts changed; citation validity 287/320 -> 320/320; declines citing the failing rule 217/248 -> 248/248.
* Caveat for the write-up: the hard documents add traps AND remove all 9 hedge words of the original set ("about", "roughly", "generally", ...). The LLM-decides gain (0.375 -> 0.600) cannot be attributed to either change alone.
* 1 regression test (72 total). No further design changes: every gold profile has now been evaluated.
