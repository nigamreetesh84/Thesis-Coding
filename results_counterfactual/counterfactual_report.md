# Counterfactual explanations - checked against the eligibility oracle

Profiles 120, card decisions 960, declines 738.

| Check | Result |
|---|---|
| Verdict unchanged (agreement with oracle) | 960/960 |
| Validity (all changes applied -> eligible) | 738/738 |
| Minimality (any one change undone -> declined) | 738/738 |
| Sparsity (changes only for failed criteria) | 738/738 |
| Declines reachable by the applicant's own action | 255/738 |
| Mean changes per decline | 2.14 |

Changes by kind: {'actionable': 994, 'time': 315, 'fixed': 268}
Changes by criterion: {'income': 471, 'age': 315, 'employment': 268, 'relationship': 162, 'cibil': 361}
Profiles with at least one decline: 118; with a reachable closest card: 80

## Examples
- SBI SimplyCLICK: Not eligible because annual income of ₹2,51,531 is below the ₹6,00,000 minimum; age 18 is below the minimum of 21; employment type 'business owner' is not one of the accepted types (salaried).  
  -> Not reachable by changing income or score: the card accepts only salaried applicants, so a change in income or score alone would not make a business owner applicant eligible.
- SBI Prime: Not eligible because annual income of ₹7,31,330 is below the ₹15,00,000 minimum.  
  -> To qualify: raise annual income by ₹7,68,670 (about ₹64,056 a month) to the ₹15,00,000 minimum.
- SBI SimplyCLICK: Not eligible because annual income of ₹2,87,782 is below the ₹6,00,000 minimum; CIBIL score 650 is below the minimum of 750; employment type 'self-employed' is not one of the accepted types (salaried).  
  -> Not reachable by changing income or score: the card accepts only salaried applicants, so a change in income or score alone would not make a self-employed applicant eligible.
- SBI SimplySAVE: Not eligible because age 18 is below the minimum of 21; CIBIL score 645 is below the minimum of 700.  
  -> To qualify: raise the CIBIL score by 55 points to 700; and wait until age 21 (in 3 years).
- SBI Elite: Not eligible because annual income of ₹19,33,568 is below the ₹20,00,000 minimum; CIBIL score 761 is below the minimum of 780; an existing SBI relationship is required and the applicant does not have one.  
  -> To qualify: raise annual income by ₹66,432 (about ₹5,536 a month) to the ₹20,00,000 minimum; raise the CIBIL score by 19 points to 780; and open an account with SBI first (the card is offered to existing customers only).
- SBI Prime: Not eligible because annual income of ₹12,24,218 is below the ₹15,00,000 minimum; age 19 is below the minimum of 21; CIBIL score 654 is below the minimum of 750; employment type 'business owner' is not one of the accepted types (salaried, self-employed).  
  -> Not reachable by changing income or score: the card accepts only salaried or self-employed applicants, so a change in income or score alone would not make a business owner applicant eligible.
