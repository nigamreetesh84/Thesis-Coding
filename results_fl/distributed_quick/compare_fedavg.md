# Distributed vs simulated FedAvg (fedavg)

| Test set | simulated | distributed |
|---|---|---|
| synthetic_test/AXIS | 28/30 (0.93) | 28/30 (0.93) |
| synthetic_test/HDFC | 29/30 (0.97) | 29/30 (0.97) |
| synthetic_test/SBI | 29/30 (0.97) | 29/30 (0.97) |
| heldout_style/AXIS | 24/30 (0.80) | 26/30 (0.87) |
| heldout_style/HDFC | 22/30 (0.73) | 22/30 (0.73) |
| heldout_style/SBI | 25/30 (0.83) | 25/30 (0.83) |
| real_thesis | 27/30 (0.90) | 27/30 (0.90) |
| real_thesis_hard | 22/30 (0.73) | 21/30 (0.70) |
| real_axis | 17/20 (0.85) | 17/20 (0.85) |
| injection_axis | 16/20 (0.80) | 16/20 (0.80) |

Cards with the same all-correct outcome: 51/52
Exact McNemar (distributed vs simulated, cards all correct): b=1 c=0 p=1
Largest absolute difference between the two final adapters: 0.00238

Distributed transport: 3 rounds, banks AXIS, HDFC, SBI, 8.65 MB per adapter message, 155.71 MB in total, mean round 145.5 s
