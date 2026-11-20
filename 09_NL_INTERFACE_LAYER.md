# 09 - Natural-Language Interface Layer (round 6, 18 Nov 2026)

Supervisor feedback (17 Nov): after the vector index, create a vector database storing all card details; user requirement in natural language -> LLM -> vector DB -> comparison engine -> natural-language answer.

**Superseded:** the standalone nl_layer.zip from 17 Nov. Round 6 is built inside the repo (github.com/nigamreetesh84/Thesis-Coding, branch made locally as `nl-interface`; delivered as Thesis-Coding-round6-files.zip + round6_nl_interface.patch, not pushed).

## Decision
Wrap the FROZEN extract_compare design; change nothing in the decision path. No existing prompt constant changed (new ones appended), so llm_cache entries and all round 5.5 numbers in 06 stand. 100 tests pass (72 existing + 28 new).

## Flow (src/nl_interface.py)
1. QueryParser (LLM #1, new prompt QUERY_PARSER_SYSTEM_PROMPT): copies facts, income phrase verbatim. src/domain/query_parsing.py converts lakh/crore/monthly/LPA to annual rupees (LLM never converts units). Missing/implausible field -> follow-up question (fail-closed).
2. CardRanker: the EXISTING FAISS index + chunks.parquet (the vector database) ranks cards by stated preferences; ordering only, every card still decided.
3. RAGRecommender(decision_mode="extract_compare"): unchanged.
4. AnswerWriter (LLM #2, ANSWER_WRITER_SYSTEM_PROMPT) + src/domain/answer_guard.py: sentence rejected if wrong verdict or a number not in the engine reason -> engine reason shown instead.
- API: POST /ask {message, previous}. Dashboard: plain-language box.
- Relationship stays one boolean (frozen UserProfile); per-bank relationship = limitation/future work.

## Evaluation: evaluation/run_nl_eval.py (NOT yet run on Qwen; numbers = [NEEDS DATA])
Holdout 40 profiles x 2 message sets: gold_text (gold profile_text, 98 templated + 22 reworded) and user_style (generated first-person: rupees / lakh / monthly / LPA, everyday job and bank wording). Arms: LLM parser vs regex baseline. Metrics: field accuracy (5 fields x 40 = 200), all-5-correct, follow-up rate, end-to-end profile accuracy vs gold labels, messages whose own rounding changes a gold label, writer guard acceptance, final verdict agreement (1.000 by construction).
Offline stub smoke run (regex only, not a result): gold_text 200/200; user_style 181/200, regex fails on "freelance consultant", "own practice as a CA" -> asks a follow-up; 1 user_style message (3.5 LPA) changes a gold label by rounding alone.

Colab: same env vars as the A8 notebook, then
`python -m evaluation.run_nl_eval --split holdout --out-dir $OUT/nl_holdout`

## Thesis changes
- Diagrams: end_to_end_application_architecture, detailed_design_2_online_recommendation_flow (add parser -> follow-up -> ranker -> engine -> writer+guard).
- Ch 3: NL interface section (also covers proposal Objective 2, input interface). Deviations row: added after design freeze on supervisor feedback.
- Ch 4: one table from nl_results.json (holdout).
- Ch 5: LLM used for language at both ends, decisions deterministic in the middle; regex vs LLM parser contrast.
- Limitations: messages are template-generated, not real users; ranking not evaluated against a relevance gold set; single relationship flag.
