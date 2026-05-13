# ATLAS Label-Quality Ablation: Shuffled Labels

Date: 2026-05-13

Purpose: test whether ATLAS works because the SSAR/IQL teacher label is aligned with each `(state, action)` transition, or whether any score distribution / weighted BC is enough.

Setup:

- env: `hopper-medium-replay-v2`
- seed: `0`
- steps: `50,000`
- eval frequency: `10,000`
- eval episodes: `5`
- control: shuffle `atlas_score` from `atlas_selector_hopper-medium-replay-v2_seed0.npz`
- label distribution: unchanged (`mean=0.3823`, `std=0.2212`)
- alignment: broken by random permutation

Result:

| Variant | Final | Best | Best Step |
|---------|------:|-----:|----------:|
| ATLAS teacher labels | 45.29 | 45.29 | 50k |
| ATLAS shuffled labels | 18.78 | 19.35 | 40k |

Conclusion: preserving the score distribution is not enough. The teacher-label alignment matters strongly; this supports ATLAS as a real teacher-signal method rather than generic weighted BC.
