# Round 7 – Bank portal and federated learning (22–23 Nov 2026)

Added after the supervisor meeting of 22 Nov 2026: bank administrators (SBI, HDFC, Axis)
upload their rule documents; customers choose two or three banks and get explained
decisions; federated learning of the rule extractor across banks as the new research
contribution (RQ5).

**Nothing in the frozen thesis path changed**: `src/model.py`, `src/prompts.py`,
`data/rule_docs*`, `POST /recommend`, `POST /ask` and every round 5.5 number stand.
(Two small NL-interface additions: a bare number such as "760" answers a single
missing CIBIL/age question, and the relationship regex also recognises "Axis";
the NL-evaluation smoke numbers are unchanged: 200/200 and 181/200.)

---

## 1. Run it (Windows)

```bat
call mydev\Scripts\activate.bat
pip install -r requirements.txt
set USE_STUB_LLM=true
set RAG_DECISION_MODE=extract_compare
python -m src.portal.samples
python -m uvicorn src.api:app
```
(PowerShell: `$env:USE_STUB_LLM="true"` instead of `set`.)

- Portal: http://127.0.0.1:8000/portal
- Original dashboard (unchanged): http://127.0.0.1:8000/

On first start SBI and HDFC are loaded as version v001 from the thesis rule documents.
Axis has no rules until its administrator uploads and approves them.

### Demo accounts (prototype only)

| User | Password | Role |
|---|---|---|
| admin_sbi | sbi@123 | SBI employee |
| admin_hdfc | hdfc@123 | HDFC employee |
| admin_axis | axis@123 | Axis employee |
| customer | customer@123 | customer |

Change one: `python -m src.portal.auth set-password admin_axis NewPass123`

To reset the demo, stop the server and delete `data/portal/`, `artifacts/portal/` and
`config/portal_users.json`.

## 2. Five-minute demo script

1. **Axis employee** logs in → *Upload a rule document* → click *Axis eligibility page*
   to download it → upload it with the card names shown → the review screen shows every
   value with the chunk it came from; Neo's income is flagged (Axis publishes none).
2. Correct one value (e.g. set ACE minimum CIBIL to 750) → *Approve and publish*.
   The version list shows v001 LIVE, "1 correction saved for federated learning";
   the audit log shows *chain intact*.
3. Upload the *illustrative update* → the diff shows ACE income ₹6,00,000 → ₹7,20,000
   (and warns that the reviewer's CIBIL correction is not in the new document).
4. **Customer** logs in → ticks SBI, HDFC and Axis → types
   "I'm 34, salaried, earn 18 lakh a year, CIBIL 780, I already bank with Axis" →
   12 cards across 3 banks, each with its reason, cited rule and rule version.
   SBI Elite says an *SBI* relationship is needed (per-bank relationship).
5. Leave out the CIBIL score → the system asks for it; reply "780".
6. Upload the *prompt-injection test* file as Axis → the rule-based reader is fooled into
   CIBIL 300 by the injected sentence; the review step is where a human catches it
   (and RQ5 measures whether the LLM extractors obey it).

## 3. What was built

| Part | Files |
|---|---|
| Role logins (PBKDF2 hashes, signed tokens) | `src/portal/auth.py` |
| PDF / Markdown / text ingest → one rule per chunk | `src/portal/ingest.py` |
| Extractors: frozen LLM prompt, and a rule-based baseline | `src/portal/rule_extractor.py` |
| Versioned, approved knowledge base; diff; corrections | `src/portal/registry.py` |
| Hash-chained, tamper-evident audit log | `src/portal/audit.py` |
| Decisions from approved rules, per-bank relationship | `src/portal/approved_recommender.py` |
| Service + API routes (`/portal/...`) | `src/portal/service.py`, `src/portal/routes.py` |
| Portal UI | `dashboard/portal.html` |
| Axis documents and answer key (**verify before use**) | `data/axis/`, `data/portal_samples/`, `src/portal/samples.py` |
| Federated learning (RQ5) | `fl/`, `notebooks/fl_colab.ipynb` |
| Tests (16 new; 124 total pass) | `tests/test_portal.py`, `tests/test_fl.py` |

Design choices to state in Chapter 3:
- **Human in the loop**: only employee-approved values reach customers; no LLM sits
  between an approved rule and a decision (same deterministic comparison as extract_compare).
- **Audit**: every upload, approval and decision is appended to a hash chain; a customer
  decision stores only a salted hash of the request, never the applicant's details.
  This gives the tamper-evidence a blockchain is used for, without a distributed network,
  which a single-operator prototype does not need (say so if asked about blockchain).
- **Bank tie-up is simulated**; Axis text is compiled from public pages (sources in
  `data/axis/SOURCES.md`).

## 4. Baseline result already available (CPU, no GPU needed)

The rule-based reader on the thesis documents (truth = eligibility table, 40 fields each):

| Extractor | Original documents | Hard documents (A8) |
|---|---|---|
| Rule-based reader (regex) | 34/40 | **18/40** |
| Qwen2.5-7B, frozen prompt (round 5.5) | 40/40 | 40/40 |

A hand-written reader collapses on realistic wording; this is the motivation for an LLM
extractor (and the non-learned baseline for RQ5). Test: `tests/test_portal.py::test_regex_baseline_scores_on_thesis_documents`.

## 5. Federated learning (RQ5)

**RQ5.** Can banks jointly fine-tune a small rule extractor without sharing their documents,
and how close does it get to centralised training, including on a bank left out of training?

- Clients = SBI, HDFC, Axis. Each bank's training documents are written in its own house
  style (SBI gross annual income; HDFC *net monthly* income, as on its real pages; Axis
  "Rs X lakh and above"), so the data is non-IID, as between real banks.
- Model: Qwen2.5-1.5B-Instruct, 4-bit, LoRA r=16 on attention projections. Only adapter
  weights are averaged (FedAvg, weighted by client data size). The prompt and output
  format are the frozen extraction prompt, so the result is a drop-in extractor.
- Approved-portal corrections are added to that bank's local data automatically.
- Arms (compute-matched): zero-shot · local-only per bank · centralised (pooled) ·
  **FedAvg** · FedAvg leave-one-bank-out (Axis unseen).
- Test sets (never trained on): unseen synthetic cards per bank; a held-out phrasing
  style no bank trains on; the thesis documents (original + hard, 16 cards); the real
  Axis page (4 cards); the prompt-injection file.
- Metrics: field accuracy (card × 5 fields, Wilson CI), cards with all fields correct,
  JSON validity, invented thresholds where the document states none; exact McNemar on cards.

### Run on Colab (T4)
Open `notebooks/fl_colab.ipynb`; it clones this branch, so **push `round7-admin-fl` first**.
Full run about 1.5–2.5 h; resumable after a disconnect. Quick check first:
`python -m fl.run_fl train --preset quick --out-dir <drive folder>`.
Then upload `results.json`, `report.md` and `predictions_*.jsonl`.

### Honest limitations to state
- Simulation on one GPU, not three banks' servers; no differential privacy or secure
  aggregation (adapter updates could leak information in principle).
- Training cards are synthetic; real public rule pages are few (that is why the real
  documents are used only as test sets).
- Public rule pages weaken the privacy motive; the motivation is banks' internal,
  unpublished underwriting rules and their employees' corrections.
- Axis answer key is a draft until verified against the live pages.
- One income per card: Flipkart Axis has separate salaried/self-employed thresholds
  (schema limitation, documented in `data/axis/SOURCES.md`).
