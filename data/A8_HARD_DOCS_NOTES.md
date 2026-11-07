# A8: harder rule documents (DRAFT - review and edit before use)

`data/rule_docs_hard/` restates exactly the same eligibility rules as
`data/rule_docs/` (and therefore as `data/eligibility_table.csv`), with the
same cards and the same section headings, so the gold labels stay valid and
only the WORDING changes. It tests whether requirement extraction survives
wording that defeats word-matching (second marker point S5; the proposal's
promised "distractor rules and reworded documents").

This file is outside `rule_docs_hard/`, so it is never indexed or shown to
the model.

| Card | Trap | Correct reading |
|---|---|---|
| SBI SimplyCLICK | Income per MONTH (Rs 50,000 x 12); fee and fee-waiver figures | 6,00,000; fee figures ignored |
| SBI SimplyCLICK | Negated age and score ("younger than twenty-one are declined", "under 750 leads to rejection") | 21; 750 |
| SBI SimplyCLICK | Employment stated by exclusion | salaried only |
| SBI SimplySAVE | Words only; superseded older floor (two lakh) | 3,00,000 |
| SBI SimplySAVE | Words only ("twenty-one", "seven hundred") | 21; 700 |
| SBI Prime | Superseded older figure (twelve lakh) before the current one | 15,00,000 |
| SBI Prime | Another card's threshold mentioned (SimplySAVE 700); business owners excluded | 750; salaried, self-employed |
| SBI Elite | Words only; spend figure that is not a condition | 20,00,000 |
| SBI Elite | Another card's threshold mentioned (Prime 750) | 780 |
| SBI Elite | Relationship stated as "existing customers only" | required |
| HDFC MoneyBack+ | Words only ("two and a half lakh"); monthly cashback cap that is not a condition | 2,50,000 |
| HDFC MoneyBack+ | Negated score ("below six hundred and fifty ... will not go ahead") | 650 |
| HDFC Millennia | Another card's income mentioned (Regalia twelve lakh) | 3,50,000 |
| HDFC Millennia | Business owners excluded | salaried, self-employed |
| HDFC Regalia | Income per MONTH (one lakh) plus joining-fee and spend figures | 12,00,000 |
| HDFC Diners Club Black | Words only; superseded older figure (21 lakh) | 25,00,000 |
| HDFC Diners Club Black | Self-employed excluded; relationship as "existing customers only" | salaried, business owner; required |

Run (Colab, T4): see the A8 cell in the round 5.4 notebook. It rebuilds a
separate index from these documents and runs A7 on the holdout split
without touching the main artefacts.
