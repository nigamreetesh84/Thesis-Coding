# Evaluation notes (read before interpreting any number)

## What is measured
* **Task.** For each synthetic applicant profile and each of the 8 SBI/HDFC cards, decide *recommended / not recommended*. A profile counts as correct only if **all 8 cards** are right (`profile accuracy`). Per-card precision/recall/F1/accuracy are reported alongside it.
* **System.** Retrieval (Sentence-BERT + FAISS) returns the rule chunks of each requested card; the local LLM (4-bit, greedy decoding) decides one card per prompt, writes a reason and cites a chunk. No runtime validation gate exists.
* **Gold labels** come from the deterministic oracle over `data/eligibility_table.csv`. The rule documents are paraphrased independently of that table (checked by `verify_setup`).
* **A1 / RQ3 control (B1).** Same prompt, same card list, same profile and output format, but **no retrieved rules** and a system prompt asking for a best-effort answer from general knowledge. Significance is a paired McNemar test over profiles.

## Known limitations (state these in the thesis)
1. **Synthetic, oracle-derived gold set.** Labels are produced by the same table the documents paraphrase; no independent human annotation is included (see README). Numbers are an optimistic upper bound.
2. **Hard by construction.** Half of the 120 profiles are "borderline" and 46% of card decisions lie within ~10% income / 20 CIBIL / 1 year of a threshold, while several rules are worded with hedges ("about", "around", "roughly", "generally"). The oracle is strict, so a careful reader of the hedged text can legitimately disagree on boundary cases.
3. **Class imbalance.** About 23% of card decisions are positive, so a model that rejects everything scores about 77% per-card accuracy. Always read recall and profile accuracy next to the `majority_class_accuracy` baseline reported in RQ1.
4. **Tiny corpus.** 25 chunks, 3-4 per card. Any `top_k >= 4` retrieves identical context, so RQ2 (proxy relevance = all cards of the requested banks) is saturated at 1.0 and A2 only has power for k = 1..3.
5. **Small model.** A 4-bit 7B model makes arithmetic/comparison slips and occasionally invents conditions; this limits absolute accuracy. A4 compares model sizes.
6. **RAGAS faithfulness is not computed.** `evaluation/ragas_faithfulness.py` needs a human-labelled subset and a judge model; RQ1 is therefore reported as gold-set classification quality, and RQ4 as profile-level label consistency plus citation validity.
7. **Stub mode is circular.** `StubLLM` reads the oracle; stub results only test plumbing.
8. **Dev/test discipline.** `--split test` (every 3rd profile) is the held-out set; prompt or design changes should be tuned on `--split dev`. Earlier development iterations looked at results on the test profiles, so disclose that, or confirm the final configuration on `--split dev`.

## Defects found and fixed during the audit (all verified by tests)
See `CHANGES.md`. The most important: the ungrounded control prompt previously lost the card list (it could never be correct), RQ2 and RQ4 used a different retrieval/prompting path than the system, A3 removed text that did not exist in the prompt, A6 never showed the reworded text to the model, and per-card prompting/batching/caching were added.
