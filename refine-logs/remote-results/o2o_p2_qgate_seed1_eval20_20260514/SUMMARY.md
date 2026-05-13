# P2 Q-Filtered Trust Seed1 Eval20 Summary

Date: 2026-05-14  
Setting: `hopper-medium-replay-v2`, seed1, 50k offline + 10k online, eval20.

Raw files:

- results: `refine-logs/remote-results/o2o_p2_qgate_seed1_eval20_20260514/results/`
- logs: `refine-logs/remote-results/o2o_p2_qgate_seed1_eval20_20260514/logs/`

| method | offline final | offline best | online final | online best | best step |
|---|---:|---:|---:|---:|---:|
| `td3_bc_o2o_eval20_seed1_decay` | 20.19 | 20.96 | 98.86 | 98.86 | 60000 |
| `atlas_o2o_eval20_seed1_qgate_fixed` | 31.21 | 31.21 | 39.91 | 70.03 | 56000 |

Takeaway: seed1 exposes very high Hopper O2O variance. TD3+BC release reaches 98.86 final, so seed0-only ATLAS q-gate gains cannot be claimed as robust baseline dominance.
