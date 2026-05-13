# P1 Q-Filtered Trust Eval20 Summary

Date: 2026-05-14  
Setting: `hopper-medium-replay-v2`, seed0, 50k offline + 10k online, eval20.

Raw files:

- results: `refine-logs/remote-results/o2o_p1_qgate_eval20_20260514/results/`
- logs: `refine-logs/remote-results/o2o_p1_qgate_eval20_20260514/logs/`

| method | offline final | offline best | online final | online best | best step |
|---|---:|---:|---:|---:|---:|
| `atlas_o2o_eval20_qgate_fixed` | 46.70 | 46.70 | 48.41 | 48.41 | 60000 |
| `ssar_iqlqv_o2o_eval20_qgate_fixed` | 50.71 | 50.71 | 38.88 | 65.73 | 57000 |

Takeaway: Q-filtered trust repairs the seed0 ATLAS online final, turning the fixed teacher constraint from 28.93 to 48.41. It does not stabilize SSAR/IQL-qv: the best is 65.73 but the final is 38.88.
