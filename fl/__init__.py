"""Federated learning of the rule extractor (round 7, RQ5).

RQ5: Can banks jointly fine-tune a small rule extractor WITHOUT sharing their
documents, and how close does it get to training on pooled data, including on a
bank left out of training?

* Clients = banks (SBI, HDFC, Axis). Each writes its rules in its own house
  style (SBI: gross annual income; HDFC: net MONTHLY income, as on its real
  pages; Axis: "Rs X lakh and above"), so client data is non-IID.
* Each client fine-tunes LoRA adapters of a small instruction model
  (Qwen2.5-1.5B-Instruct, 4-bit) on its own examples; only adapter weights go
  to the server, which averages them weighted by client data size (FedAvg,
  McMahan et al., 2017). No document or example ever leaves its bank.
* The task and output format are the thesis's frozen extraction prompt, so the
  fine-tuned model is a drop-in replacement for the 7B extractor.

Modules: data_gen (per-bank examples), train (local training, FedAvg, arms),
evaluate (field-level scoring), run_fl (CLI used by notebooks/fl_colab.ipynb).
"""
