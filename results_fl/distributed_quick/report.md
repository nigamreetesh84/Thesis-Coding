# RQ5 – federated rule extractor: field accuracy (correct fields / total)

Model: Qwen/Qwen2.5-0.5B-Instruct · rounds 3 · local steps 8 · LoRA r=16

| Test set | fedavg | fedavg_lobo |
|---|---|---|
| synthetic_test/AXIS | 28/30 (0.93) | 27/30 (0.90) |
| synthetic_test/HDFC | 29/30 (0.97) | 28/30 (0.93) |
| synthetic_test/SBI | 29/30 (0.97) | 29/30 (0.97) |
| heldout_style/AXIS | 26/30 (0.87) | 28/30 (0.93) |
| heldout_style/HDFC | 22/30 (0.73) | 24/30 (0.80) |
| heldout_style/SBI | 25/30 (0.83) | 26/30 (0.87) |
| real_thesis | 27/30 (0.90) | 28/30 (0.93) |
| real_thesis_hard | 21/30 (0.70) | 23/30 (0.77) |
| real_axis | 17/20 (0.85) | 9/20 (0.45) |
| injection_axis | 16/20 (0.80) | 9/20 (0.45) |

| Pooled | fedavg | fedavg_lobo |
|---|---|---|
| all_unseen | 224/260 (0.86) | 222/260 (0.85) |
| real_documents | 65/80 (0.81) | 60/80 (0.75) |
| synthetic | 159/180 (0.88) | 162/180 (0.90) |

Exact McNemar on cards with all five fields correct (pooled unseen sets; injection set excluded):

- fedavg vs fedavg_lobo: b=5 c=6 p=1 (n=52)

Prompt-injection file (Axis page with an injected instruction): field accuracy
fedavg 16/20, fedavg_lobo 9/20

Invented thresholds where the document states none (income/CIBIL): fedavg 1, fedavg_lobo 0

Data sent per client per round: 8.7 MB of adapter weights; no training example leaves its bank.
