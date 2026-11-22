"""Bank portal (round 7, supervisor feedback 22 Nov 2026).

Adds a multi-bank layer ON TOP of the frozen system:

* bank administrators (SBI, HDFC, Axis) log in and upload their rule documents;
* the LLM extracts each card's requirements with a cited chunk per value;
* the administrator reviews, corrects and approves them (human in the loop);
* every approved version is stored, diffed against the previous one and written
  to a tamper-evident, hash-chained audit log;
* customers log in, choose two or three banks and get decisions from the
  APPROVED knowledge base, each tied to the rule version it used.

The frozen evaluation path (src/model.py, data/rule_docs, POST /recommend and
POST /ask) is not changed, so every round 5.5 result still stands. The bank
tie-up is SIMULATED: Axis documents are drafted from Axis Bank's public
eligibility pages, and no customer details are stored (ethics scope).
"""
