# Research Findings

## 2026-05-12: Result-to-Claim Gate For Cheap Trusted Selectors

- Current verdict: partial support.
- Supported: SSAR-style trusted action selection is the most plausible useful mechanism in the current `hopper-medium-replay-v2` exploration.
- Not supported: `trusted_td3_bc_top20` as a final method. It reached `45.13 @50k` but fell to `28.76 @100k`.
- Constraint for future attempts: do not expand naive return-ranked trajectory filtering to more seeds or environments.
- Next action: implement a cheap Q-gap selector before any additional stability runs.

## 2026-05-12: Cheap Online Selectors Failed 50k Gate

- `trusted_td3_bc_qgap_soft`: final `19.94`, best `22.05 @10k`.
- `trusted_td3_bc_consistency`: final `19.77`, best `28.00 @40k`.
- Decision: no 100k, no seed expansion for these exact variants.
- Interpretation: online TD3 critic scores and behavior-consistency weights are too weak/noisy to replace SSAR IQL-qv selection directly.
- Next constraint: redesign the selector around a better offline value source or amortized SSAR/IQL labels before spending more GPU.

## 2026-05-12: ATLAS Passed The Seed0 100k Gate

- `trusted_td3_bc_atlas` 50k: final/best `45.29`.
- `trusted_td3_bc_atlas` 100k: final/best `69.97 @100k`.
- Decision: ATLAS is the current strongest local contribution candidate.
- Interpretation: cached SSAR/IQL-qv trusted labels can be partially amortized into a lightweight selector; this is stronger than online Q-gap/consistency and more stable than return-ranked filtering.
- Next constraint: run one narrow stability check and one label-quality ablation before any broad sweep or paper-facing claim.

## 2026-05-13: SSAR Seed1 Eval20 Anchor Check

- `SSAR_hopper-medium-replay-v2_seed1_100000_eval20`: final `60.88`, best `99.22 @90k`.
- Seed1 IQL-qv cache saved at `/root/autodl-tmp/taoyao-rl-cache/SSAR/iql_qv/hopper-medium-replay-v2/seed1/0.7_model.pth`, SHA256 `53dd12638216579de50a2449ad7c598ffe9f97f85c7975ee71583fb1694a08fd`.
- Decision: do not describe seed0 `92.44 final / 100.98 best` as stable SSAR performance.
- Interpretation: SSAR can produce near-100 spikes across seeds, but tail/final score is unstable even with 20 eval episodes.
- Next constraint: treat SSAR as a high-variance upper anchor; prioritize ATLAS stability/label-quality checks over broader SSAR seed sweeps.

## 2026-05-13: ATLAS Shuffled-Label Control Failed

- `trusted_td3_bc_atlas_shuffle` 50k: final `18.78`, best `19.35 @40k`.
- Aligned ATLAS 50k: final/best `45.29`.
- Decision: the ATLAS gain depends on teacher-label alignment, not just the marginal score distribution.
- Interpretation: this is the first positive novelty-isolation ablation for ATLAS.
- Next constraint: run ATLAS seed1 using the preserved seed1 IQL-qv cache before expanding to another env.

## 2026-05-13: ATLAS Seed1 Stability Passed

- `trusted_td3_bc_atlas` seed1 eval20 100k: final/best `68.11 @100k`.
- ATLAS seed0 100k: final/best `69.97 @100k`.
- SSAR seed1 eval20: final `60.88`, best `99.22 @90k`.
- Decision: ATLAS is no longer just a seed0 artifact; it has passed the first narrow stability check.
- Interpretation: ATLAS does not reproduce SSAR's near-100 spike, but its final score is more stable across seed0/seed1 than SSAR's tail.
- Next constraint: one second replay env is now the next high-value check; do not jump to broad 6-env x 3-seed sweeps.

## 2026-05-13: ATLAS Survived Walker2d Second-Env Check

- SSAR full IQL-qv on `walker2d-medium-replay-v2`, seed0, eval10: final `94.28`, best `94.60 @40k`.
- `trusted_td3_bc_atlas` on the same split: final `71.26`, best `77.86 @90k`.
- Walker teacher cache saved at `/root/autodl-tmp/taoyao-rl-cache/SSAR/iql_qv/walker2d-medium-replay-v2/seed0/0.7_model.pth`, SHA256 `ffe559a043a2f0ef5814d7cfeb18d6ce2960120ebddeb7fffde0eb31ef3aeb64`.
- Decision: ATLAS is not hopper-only, but it leaves a clear gap to full SSAR on walker.
- Interpretation: the next improvement should target teacher-label usage and selector quality, not more baseline collection.
- Next constraint: run only 1-2 cheap ATLAS ablations first, especially soft-vs-hard label weighting and `label_min_weight`.

## 2026-05-14: P0 Offline-to-Online Eval20 Panel Completed

- Setting: `hopper-medium-replay-v2`, 50k offline + 10k online, eval20.
- TD3+BC release seed0: offline `22.20`, online final/best `40.06`.
- ATLAS release seed0: offline `46.70`, online final `37.50`, best `38.79`.
- SSAR/IQL-qv release seed0: offline `50.71`, online final `28.87`, best `31.85`.
- SSAR/IQL-qv fixed seed0: online final `38.61`, best `96.22`, showing a large transient spike but unstable tail.
- Random matched trust subset confirms label alignment matters offline, but final online score remains close to ATLAS release.
- Decision: the defensible C-line claim is now a constraint-transfer gap: teacher labels help offline initialization but do not automatically transfer to online fine-tuning.
- Next constraint: do not claim ATLAS improves O2O as a stable method; any additional C-line run must diagnose online adaptation rather than add another baseline.

## 2026-05-14: Q-Filtered Trust Is Diagnostic, Not Yet A Method Claim

- ATLAS Q-gate fixed seed0: offline `46.70`, online final/best `48.41`, repairing the seed0 fixed-constraint failure.
- SSAR/IQL-qv Q-gate fixed seed0: online final `38.88`, best `65.73`, still unstable.
- Seed1 TD3+BC release: offline `20.19`, online final/best `98.86`.
- Seed1 ATLAS release: offline `31.21`, online final/best `84.35`.
- Seed1 ATLAS Q-gate fixed: online final `39.91`, best `70.03`.
- Decision: Q-filtered trust is a promising hypothesis on seed0, but seed1 exposes high Hopper O2O variance and prevents a robust superiority claim.
- Next constraint: paper framing should emphasize mechanism diagnosis and claim discipline; A/B-line comparable rows are more valuable than another C-line sweep.
