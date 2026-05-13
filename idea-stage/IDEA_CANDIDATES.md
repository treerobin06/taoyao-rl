# Idea Candidates

| # | Idea | Existing Signal | Novelty | Status |
|---:|---|---|---:|---|
| 1 | Trusted-action transfer failure and release | ATLAS offline strong but O2O degrades | 5.5/10 | RECOMMENDED |
| 2 | Label-quality decomposition | aligned 45.29 vs shuffled 18.78 | 6/10 | SUPPORT |
| 3 | Policy extraction under fixed teacher | IQL/SSAR/ATLAS gap suggests extraction bottleneck | 6.5/10 | EXTENSION |
| 4 | Teacher cost-quality frontier | SSAR expensive, ATLAS reusable | 5/10 | SECONDARY |
| 5 | Cross-algorithm trust reuse | not yet tested | 5.5/10 | DEFER |

## Active Idea

**Trusted-action transfer failure and release**

- Hypothesis: action-level teacher trust is useful offline but over-constrains online adaptation if applied naively.
- Key evidence: ATLAS seed0/seed1 offline near 68-70; shuffled label collapse; ATLAS O2O falls from 45.29 to 37.97 while TD3+BC improves from 22.43 to 39.64.
- Next step: run one narrow O2O mechanism test: fixed vs linear decay vs online-gated trust.
