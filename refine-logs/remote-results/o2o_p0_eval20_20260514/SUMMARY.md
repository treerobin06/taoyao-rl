# P0 Offline-to-Online Eval20 Summary

Date: 2026-05-14  
Remote instance: AutoDL `pro-7785f027d673`  
Setting: `hopper-medium-replay-v2`, seed0, 50k offline + 10k online, eval20, mixed offline/online replay.

Raw files:

- results: `refine-logs/remote-results/o2o_p0_eval20_20260514/results/`
- logs: `refine-logs/remote-results/o2o_p0_eval20_20260514/logs/`
- random-subset control label: `refine-logs/remote-results/o2o_p0_eval20_20260514/random_subset_iqlqv_hopper-medium-replay-v2_seed0.npz`

| method | offline final | offline best | online final | online best | best step | final delta |
|---|---:|---:|---:|---:|---:|---:|
| `td3_bc_o2o_eval20_decay` | 22.20 | 22.20 | 40.06 | 40.06 | 60000 | +17.86 |
| `td3_bc_o2o_eval20_fixed` | 22.20 | 22.20 | 33.35 | 46.12 | 58000 | +11.15 |
| `atlas_o2o_eval20_decay` | 46.70 | 46.70 | 37.50 | 38.79 | 56000 | -9.20 |
| `atlas_o2o_eval20_fixed` | 46.70 | 46.70 | 28.93 | 32.80 | 53000 | -17.77 |
| `random_subset_iqlqv_o2o_eval20_decay` | 12.14 | 19.95 | 35.53 | 44.13 | 57000 | +23.39 |
| `ssar_iqlqv_o2o_eval20_decay` | 50.71 | 50.71 | 28.87 | 31.85 | 58000 | -21.84 |
| `ssar_iqlqv_o2o_eval20_fixed` | 50.71 | 50.71 | 38.61 | 96.22 | 57000 | -12.10 |

Immediate reading:

- TD3+BC remains the best final online result in this short eval20 P0 slice.
- ATLAS and SSAR/IQL-qv labels give much stronger offline initialization, but carrying the teacher constraint into online fine-tuning hurts final score under this runner.
- Random subset control keeps the same IQL-qv hard-trust fraction but loses offline performance, supporting the claim that aligned teacher labels matter.
- SSAR/IQL-qv fixed has a large transient online spike, but the tail drops sharply; report both best and final, and avoid claiming stable O2O improvement from this run alone.
