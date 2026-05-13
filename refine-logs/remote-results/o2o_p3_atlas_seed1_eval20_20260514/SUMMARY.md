# P3 ATLAS Seed1 O2O Eval20 Summary

Date: 2026-05-14  
Setting: `hopper-medium-replay-v2`, seed1, 50k offline + 10k online, eval20.

Raw files:

- results: `refine-logs/remote-results/o2o_p3_atlas_seed1_eval20_20260514/results/`
- logs: `refine-logs/remote-results/o2o_p3_atlas_seed1_eval20_20260514/logs/`

| method | offline final | offline best | online final | online best | best step |
|---|---:|---:|---:|---:|---:|
| `atlas_o2o_eval20_seed1_decay` | 31.21 | 31.21 | 84.35 | 84.35 | 60000 |
| `atlas_o2o_eval20_seed1_fixed` | 31.21 | 31.21 | 23.88 | 49.92 | 57000 |

Takeaway: ATLAS release is strong on seed1, while ATLAS fixed remains poor. Q-gate fixed is better than fixed, but worse than simple release on seed1.
