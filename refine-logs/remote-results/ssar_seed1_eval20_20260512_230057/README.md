# SSAR Seed1 Eval20 Check

Date: 2026-05-12 late night to 2026-05-13 00:14 CST

Purpose: verify whether the strong cached SSAR seed0 result (`92.44` final / `100.98` best with 5 eval episodes) is stable across seed and less noisy evaluation.

Remote run root:

- `/root/autodl-tmp/external_quick_logs/ssar_seed1_eval20_20260512_230057`

Command setting:

- source: external SSAR checkout copied from `/root/autodl-tmp/external_repos/SSAR`
- env: `hopper-medium-replay-v2`
- seed: `1`
- IQL-qv preselection: full `1,000,000` trusted-action seeking steps
- offline steps: `100,000`
- eval frequency: `10,000`
- eval episodes: `20`
- elapsed: `4430s`

Result:

| Step | Raw Return | Normalized Score |
|------|-----------:|-----------------:|
| 10k | 902.266 | 28.346 |
| 20k | 701.852 | 22.188 |
| 30k | 843.393 | 26.537 |
| 40k | 842.768 | 26.518 |
| 50k | 862.118 | 27.112 |
| 60k | 1346.306 | 41.989 |
| 70k | 1884.461 | 58.525 |
| 80k | 2141.792 | 66.432 |
| 90k | 3208.764 | 99.215 |
| 100k | 1961.016 | 60.877 |

Summary:

- final: `60.877`
- best: `99.215 @90k`
- interpretation: SSAR can still spike near 100 on seed1 with 20-episode eval, but the final/tail is much lower than seed0. Treat cached SSAR as a high-variance upper anchor, not as a stable final-score claim.

Seed1 IQL-qv cache:

- remote backup: `/root/autodl-tmp/taoyao-rl-cache/SSAR/iql_qv/hopper-medium-replay-v2/seed1/0.7_model.pth`
- SHA256: `53dd12638216579de50a2449ad7c598ffe9f97f85c7975ee71583fb1694a08fd`
