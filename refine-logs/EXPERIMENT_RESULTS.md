# Experiment Results

**Date**: 2026-05-12
**Milestone**: M1 one-env four-system screen
**Setting**: `hopper-medium-replay-v2`, seed 0, 50k gradient steps, eval every 10k, 5 eval episodes.

## Summary Table

| System | Final Normalized Score | Best Normalized Score | Best Step | Final Raw Return | Interpretation |
|--------|------------------------|-----------------------|-----------|------------------|----------------|
| `bc` | 17.86 | 32.26 | 30k | 560.90 | cheap anchor; high variance, strong transient at 30k |
| `td3_bc` | 22.43 | 22.43 | 50k | 709.72 | fixed-loader baseline remains low |
| `td3_bc_alpha5` | 21.95 | 21.95 | 50k | 693.99 | stronger Q term does not help |
| `rebrac_lite` | 34.48 | 34.48 | 50k | 1101.96 | first positive C-track signal; +12.05 over TD3+BC final |

## Official-Source Screen

Same setting unless noted: `hopper-medium-replay-v2`, seed 0, 50k steps, eval every 10k.

| Source Method | Final Normalized Score | Best Normalized Score | Best Step | Status | Interpretation |
|---------------|------------------------|-----------------------|-----------|--------|----------------|
| PRDC official source | 23.54 | 23.54 | 50k | DONE | runs cleanly, but only matches/slightly exceeds TD3+BC at 50k |
| A2PR official source | 22.31 | 22.81 | 40k | DONE | runs cleanly, unstable early curve, no clear uplift over TD3+BC |
| SSAR official source-localized | 38.56 | 43.97 | 40k | DONE | strongest current signal; clean full IQL-qv preselection took about 62 min, total run 4003s |

## Decision

- `rebrac_lite` passes the exploration-stage go gate on this one-seed smoke: it improves final normalized score by more than 10 points over `td3_bc`.
- `td3_bc_alpha5` should not be expanded.
- PRDC/A2PR official-source 50k runs do not yet beat the ReBRAC-lite signal.
- SSAR full-IQL beats ReBRAC-lite on this one-seed screen: final 38.56 vs 34.48, best 43.97 vs 34.48. Treat SSAR as the strongest current modern baseline, but account for its expensive IQL-qv preselection.

## Mechanism Ablation

Same setting: `hopper-medium-replay-v2`, seed 0, 100k steps, eval every 10k.

| Variant | Final Normalized Score | Best Normalized Score | Best Step | Interpretation |
|---------|------------------------|-----------------------|-----------|----------------|
| SSAR cached IQL-qv | 92.44 | 100.98 | 90k | very strong seed0 run, but later seed1 eval20 shows this is a high-variance upper anchor rather than a stable final level |
| cheap SSAR without IQL action selection | 25.48 | 30.34 | 90k | removing IQL-qv trusted action selection collapses performance near TD3+BC/ReBRAC-lite range |
| ReBRAC-lite | 36.54 | 54.36 | 90k | simple regularization can spike, but remains far below cached full SSAR at 100k |

Mechanism takeaway: SSAR's expensive IQL-qv trusted action selection appears to be a central contributor, not a disposable preprocessing detail. A promising contribution should target a cheaper substitute for this selection step or a way to amortize/cache it safely.

## SSAR Anchor Reliability Check

This check was added because the seed0 cached SSAR curve had a very high tail under only 5 eval episodes. The seed1 run uses the same external SSAR logic, full IQL-qv preselection, 100k offline steps, but evaluates with 20 episodes.

| Variant | Seed | Eval Episodes | Final Normalized Score | Best Normalized Score | Best Step | Interpretation |
|---------|-----:|--------------:|-----------------------:|----------------------:|----------:|----------------|
| SSAR cached IQL-qv | 0 | 5 | 92.44 | 100.98 | 90k | strong seed0 upper anchor, but high-variance eval |
| SSAR full IQL-qv | 1 | 20 | 60.88 | 99.22 | 90k | still spikes near 100, but tail drops to 60.88 |

Reliability takeaway: SSAR is genuinely capable of high scores on this setting, and the IQL-qv selection signal remains important. However, `92.44/100.98` should not be reported as a stable baseline level. Use it as a high-variance upper anchor; use seed1 eval20 final `60.88` as evidence that the tail/final score is unstable.

## Cheap Trusted-Selector Probe

Same env/seed/eval setting. Local candidate: `trusted_td3_bc_top20`, which keeps TD3+BC critic training on the full dataset but applies the actor BC regularizer mainly to transitions from the top 20% trajectory-return slice.

| Variant | Steps | Final Normalized Score | Best Normalized Score | Best Step | Interpretation |
|---------|-------|------------------------|-----------------------|-----------|----------------|
| `trusted_td3_bc_top20` | 50k | 45.13 | 45.13 | 50k | beats 50k ReBRAC-lite, but the final eval has high variance |
| `trusted_td3_bc_top20` | 100k | 28.76 | 45.13 | 50k | the peak does not persist; return-ranked trust alone is not stable enough |

Selector takeaway: cheap trust weighting is worth pursuing, but naive trajectory-return filtering is not the final method. The next contribution candidate should use a cheap critic/Q-gap or behavior-consistency signal that is closer to SSAR's trusted action selection.

## Cheap Online Selector Follow-Up

Same env/seed/eval setting, 50k steps. These variants test whether the selector can be computed online from the local TD3+BC agent rather than from SSAR's IQL-qv preselection.

| Variant | Final Normalized Score | Best Normalized Score | Best Step | Selector Signal | Interpretation |
|---------|------------------------|-----------------------|-----------|-----------------|----------------|
| `trusted_td3_bc_qgap_soft` | 19.94 | 22.05 | 10k | Q-gap weight from detached `sigmoid((Q_data-Q_pi)/10)` after 5k warmup | failed the 50k gate; online TD3 critic is not reliable enough as a trusted-action selector |
| `trusted_td3_bc_consistency` | 19.77 | 28.00 | 40k | BC weight from policy/action consistency after 5k warmup | also below ReBRAC-lite; conservative behavior consistency alone is insufficient |

Follow-up takeaway: the cheap selector problem likely needs a better offline value signal than the online TD3 critic provides. Do not run 100k or seed expansion for these two exact variants. The next redesign should consider light critic pretraining, offline value labels, or amortizing SSAR/IQL-qv labels rather than relying on immediate online selector scores.

## Public Value Baseline Check

Same env/seed/eval setting. These compact shared-pipeline baselines were added to answer whether SSAR's gains are mostly an IQL/value-backbone effect.

| Variant | Steps | Final Normalized Score | Best Normalized Score | Best Step | Interpretation |
|---------|-------|------------------------|-----------------------|-----------|----------------|
| `iql` | 100k | 45.27 | 81.28 | 80k | strong value baseline; explains part of the SSAR signal, but does not match cached SSAR final/best |
| `cql` | 50k | 39.81 | 39.81 | 50k | useful conservative value anchor; comparable to SSAR full 50k final but far below IQL/SSAR high points |

Value-baseline takeaway: SSAR should not be described as a gain over weak TD3+BC alone. A fair story must include IQL as a strong public baseline. Cached SSAR can spike near 100, but seed1 eval20 drops to 60.88 final, so the remaining mechanism question is more precise: can we cheaply reproduce or amortize the IQL-derived trusted-action signal, while handling its high variance, not merely replace TD3+BC with IQL.

## ATLAS Infrastructure Smoke

This is not a method result yet. It validates the reusable path needed for the next method run.

| Component | Output | Status | Notes |
|-----------|--------|--------|-------|
| SSAR/IQL-qv label export | `iql_qv_hopper-medium-replay-v2_seed0.npz` | DONE | hard trust fraction 0.3696; advantage mean/std -1.4102 / 5.7651; trust score mean 0.3885 |
| ATLAS selector smoke train | `atlas_selector_smoke_hopper-medium-replay-v2_seed0.npz` | DONE | 20k transition subset, 1 epoch, validation accuracy 0.6375 |
| label-file TD3+BC smoke | `trusted_td3_bc_atlas_smoke` | DONE | 2-step AutoDL smoke passed; label loading, weighted BC update, eval write all work |

## ATLAS Teacher-Label Selector Result

Same env/seed/eval setting. ATLAS trains a lightweight `(state, action) -> trusted score` selector from cached SSAR/IQL-qv teacher labels, then uses the predicted label file as the BC regularization weight in `trusted_td3_bc.py`.

| Variant | Steps | Final Normalized Score | Best Normalized Score | Best Step | Interpretation |
|---------|-------|------------------------|-----------------------|-----------|----------------|
| `trusted_td3_bc_atlas` | 50k | 45.29 | 45.29 | 50k | passes first gate; beats ReBRAC-lite 34.48 and CQL 39.81, matches IQL final 45.27 |
| `trusted_td3_bc_atlas` | 100k | 69.97 | 69.97 | 100k | positive seed0 signal; does not collapse like return-ranked filtering; compare against SSAR as a high-variance upper anchor rather than a stable target |

## ATLAS Seed Stability Check

This run uses seed1 SSAR/IQL-qv teacher labels and evaluates with 20 episodes, matching the SSAR seed1 reliability check.

| Variant | Seed | Eval Episodes | Final Normalized Score | Best Normalized Score | Best Step | Interpretation |
|---------|-----:|--------------:|-----------------------:|----------------------:|----------:|----------------|
| `trusted_td3_bc_atlas` | 0 | 5 | 69.97 | 69.97 | 100k | first positive seed |
| `trusted_td3_bc_atlas` | 1 | 20 | 68.11 | 68.11 | 100k | similar final, no late collapse |
| SSAR full IQL-qv | 1 | 20 | 60.88 | 99.22 | 90k | higher spike, lower final/tail |

Seed-stability takeaway: ATLAS now has a stronger story than after seed0 alone. It does not reach SSAR's near-100 spike, but its seed0/seed1 final scores are both near 68-70, while SSAR's seed1 final drops to 60.88 after a 90k spike.

ATLAS takeaway: the amortized teacher-label direction is now the strongest local contribution candidate. The result supports the claim that SSAR/IQL trusted-action information can be partially distilled into a cheaper selector. It is not paper-ready yet: next run should be one narrow stability check, plus an ablation showing that teacher labels matter.

## ATLAS Label-Quality Ablation

Same env/seed/eval setting, 50k steps. The shuffled-label control preserves the `atlas_score` distribution but randomly permutes scores across transitions, breaking the `(state, action) -> teacher score` alignment.

| Variant | Final Normalized Score | Best Normalized Score | Best Step | Interpretation |
|---------|-----------------------:|----------------------:|----------:|----------------|
| `trusted_td3_bc_atlas` | 45.29 | 45.29 | 50k | aligned teacher labels |
| `trusted_td3_bc_atlas_shuffle` | 18.78 | 19.35 | 40k | same score distribution, broken alignment |

Label-quality takeaway: ATLAS is not merely benefiting from a generic weighted BC score distribution. The aligned SSAR/IQL teacher signal matters; shuffling the labels drops the method below TD3+BC/ReBRAC-lite and far below aligned ATLAS.

## ATLAS Second Replay Env Check

This run tests whether ATLAS is hopper-specific by moving to `walker2d-medium-replay-v2`, seed0, eval10. The full SSAR/IQL-qv anchor was run first to generate and preserve the walker teacher cache.

| Variant | Final Normalized Score | Best Normalized Score | Best Step | Interpretation |
|---------|-----------------------:|----------------------:|----------:|----------------|
| SSAR full IQL-qv | 94.28 | 94.60 | 40k | strong teacher anchor; walker is not teacher-limited |
| `trusted_td3_bc_atlas` | 71.26 | 77.86 | 90k | ATLAS transfers directionally to walker, but does not close the SSAR gap |

Teacher/cache diagnostics:

| Item | Value |
|------|-------|
| cache path | `/root/autodl-tmp/taoyao-rl-cache/SSAR/iql_qv/walker2d-medium-replay-v2/seed0/0.7_model.pth` |
| cache SHA256 | `ffe559a043a2f0ef5814d7cfeb18d6ce2960120ebddeb7fffde0eb31ef3aeb64` |
| hard trust fraction | `0.3678` |
| trust score mean | `0.3824` |
| selector val acc | `0.6833` |

Second-env takeaway: ATLAS is no longer just a hopper artifact. However, the walker gap to SSAR is large enough that the next step should be targeted ATLAS optimization rather than seed/env expansion. The most useful cheap ablations are soft-vs-hard labels, `label_min_weight`, and selector capacity.

## Minimal Offline-to-Online Fine-Tuning Check

This check fills the original project gap: current C-line evidence was mostly offline/mechanism evidence, while the project title is offline-to-online transfer. The new local runner performs 50k offline pretraining followed by 10k online fine-tuning with mixed offline/online replay. BC/ATLAS regularization is applied only to offline samples.

Setting: `hopper-medium-replay-v2`, seed0, 50k offline + 10k online, eval5.

| Variant | Offline Final | Online Final | Online Best | Best Online Step | Interpretation |
|---------|--------------:|-------------:|------------:|-----------------:|----------------|
| `td3_bc_o2o_decay` | 22.43 | 39.64 | 39.64 | 60k | online fine-tuning improves the weak TD3+BC anchor; decay has a small positive signal |
| `td3_bc_o2o_fixed` | 22.43 | 36.78 | 36.78 | 60k | online fine-tuning improves, but slightly below decay in this short slice |
| `atlas_o2o_decay` | 45.29 | 37.97 | 38.26 | 56k | ATLAS offline advantage does not persist under this first online setup |
| `atlas_o2o_fixed` | 45.29 | 29.22 | 32.27 | 58k | fixed ATLAS regularization is worse than decay, suggesting over-constraint during online adaptation |
| `atlas_o2o_decay_ofrac025` | 45.29 | 31.46 | 43.73 | 54k | lower online replay fraction gives a transient spike but worse final score |

O2O takeaway: the project now has a working offline-to-online curve. However, the first ATLAS O2O check is cautionary: ATLAS improves offline learning but can degrade once online mixed replay starts. Fixed ATLAS regularization is worse than decay, so the issue is not simply that decay relaxes the teacher signal too quickly. Reducing online replay fraction to `0.25` produces a transient spike but worse final score, so "online replay mixed too fast" is not the whole explanation either. The current teacher-label regularizer likely conflicts with online adaptation under this runner. Do not claim ATLAS improves O2O without a redesigned online objective.

## P0 Offline-to-Online Eval20 Panel

This is the required P0 panel for the C-line O2O claim boundary. It reruns the minimal offline-to-online setup with 20 evaluation episodes and adds the missing controls: random trusted subset and SSAR/IQL-qv teacher-derived labels.

Setting: `hopper-medium-replay-v2`, seed0, 50k offline + 10k online, eval20. Local raw copy: `refine-logs/remote-results/o2o_p0_eval20_20260514/`.

| Variant | Offline Final | Offline Best | Online Final | Online Best | Best Online Step | Final Delta | Interpretation |
|---------|--------------:|-------------:|-------------:|------------:|-----------------:|------------:|----------------|
| `td3_bc_o2o_eval20_decay` | 22.20 | 22.20 | 40.06 | 40.06 | 60k | +17.86 | weak offline anchor, best final online score in this P0 slice |
| `td3_bc_o2o_eval20_fixed` | 22.20 | 22.20 | 33.35 | 46.12 | 58k | +11.15 | improves online but tail is less stable than decay |
| `atlas_o2o_eval20_decay` | 46.70 | 46.70 | 37.50 | 38.79 | 56k | -9.20 | strong offline initialization, but loses final score online |
| `atlas_o2o_eval20_fixed` | 46.70 | 46.70 | 28.93 | 32.80 | 53k | -17.77 | fixed ATLAS teacher constraint is worse than decay |
| `random_subset_iqlqv_o2o_eval20_decay` | 12.14 | 19.95 | 35.53 | 44.13 | 57k | +23.39 | same trust fraction without alignment fails offline, then partially recovers online |
| `ssar_iqlqv_o2o_eval20_decay` | 50.71 | 50.71 | 28.87 | 31.85 | 58k | -21.84 | strongest offline endpoint among decay runs, but worst online final |
| `ssar_iqlqv_o2o_eval20_fixed` | 50.71 | 50.71 | 38.61 | 96.22 | 57k | -12.10 | huge transient spike, but final drops below TD3+BC decay |

P0 takeaway: the stronger teacher labels clearly help offline initialization, and the random-subset control supports that label alignment matters. But in this O2O runner, stronger teacher constraints do not translate into better final online performance. The defensible current claim is a mechanism diagnosis: trusted-action labels are useful offline supervision under low-quality replay, while online fine-tuning needs a better adaptation/release mechanism than simply carrying the same teacher regularizer forward.

## Q-Filtered Trust O2O Diagnostic

Gemini review identified the strongest technical weakness after P0: the linear release schedule diagnoses over-constraint but does not repair it. We therefore added a minimal online Q-filtered trust gate. During online fine-tuning, teacher BC weight is retained only when the current critic rates the dataset action above the current policy action.

Seed0 setting: `hopper-medium-replay-v2`, 50k offline + 10k online, eval20. Local raw copy: `refine-logs/remote-results/o2o_p1_qgate_eval20_20260514/`.

| Variant | Offline Final | Offline Best | Online Final | Online Best | Best Online Step | Interpretation |
|---------|--------------:|-------------:|-------------:|------------:|-----------------:|----------------|
| `atlas_o2o_eval20_qgate_fixed` | 46.70 | 46.70 | 48.41 | 48.41 | 60k | repairs ATLAS seed0 fixed-constraint failure and beats TD3+BC seed0 final |
| `ssar_iqlqv_o2o_eval20_qgate_fixed` | 50.71 | 50.71 | 38.88 | 65.73 | 57k | improves over SSAR release final, but remains unstable |

Seed1 check, same setting/eval. Local raw copies: `refine-logs/remote-results/o2o_p2_qgate_seed1_eval20_20260514/` and `refine-logs/remote-results/o2o_p3_atlas_seed1_eval20_20260514/`.

| Variant | Offline Final | Offline Best | Online Final | Online Best | Best Online Step | Interpretation |
|---------|--------------:|-------------:|-------------:|------------:|-----------------:|----------------|
| `td3_bc_o2o_eval20_seed1_decay` | 20.19 | 20.96 | 98.86 | 98.86 | 60k | seed1 TD3+BC release jumps to near-expert score; high O2O variance |
| `atlas_o2o_eval20_seed1_decay` | 31.21 | 31.21 | 84.35 | 84.35 | 60k | simple release is strong on seed1, but below TD3+BC |
| `atlas_o2o_eval20_seed1_fixed` | 31.21 | 31.21 | 23.88 | 49.92 | 57k | fixed ATLAS remains poor |
| `atlas_o2o_eval20_seed1_qgate_fixed` | 31.21 | 31.21 | 39.91 | 70.03 | 56k | q-gate improves over fixed, but is worse than release on seed1 |

Q-gate takeaway: Q-filtered trust is the first targeted online adaptation mechanism with a positive seed0 signal. It should be treated as the next hypothesis, not as a stable method claim. Seed1 shows that Hopper O2O is high-variance and that TD3+BC / ATLAS release can both become very strong. The paper should claim a constraint-transfer gap and a promising diagnostic gate, not robust superiority.

## W&B Runs

- `bc`: https://wandb.ai/tree06/taoyao-rl/runs/8fl2xyfh
- `td3_bc`: https://wandb.ai/tree06/taoyao-rl/runs/d7lna4n6
- `td3_bc_alpha5`: https://wandb.ai/tree06/taoyao-rl/runs/oshuqift
- `rebrac_lite`: https://wandb.ai/tree06/taoyao-rl/runs/pnttw327

## Caveats

- This is still exploration evidence, not a paper-ready claim; ATLAS now has two seeds on hopper and one seed on walker, but no broad multi-seed table.
- `rebrac_lite` is a compact PyTorch implementation for signal finding, not yet a strict official JAX ReBRAC reproduction.
- BC best@30k is high, so the next stage should compare full curves and final scores across seeds.
- Mechanism ablation uses only 5 evaluation episodes for seed0. Seed1 eval20 confirms that SSAR can still spike near 100, but final/tail score can drop sharply.
- PRDC and SSAR repos were used as external source checkouts because no license file was found locally; do not vendor their code into the public repo unless licensing is clarified.
- A cached/reduced SSAR run reached 22.22 at 50k, but it loaded a short preflight IQL-qv checkpoint; exclude it from official-source conclusions.
- Clean SSAR full-IQL log root on AutoDL: `/root/autodl-tmp/external_quick_logs/ssar_full_clean_20260512_101809`.
- Mechanism ablation log root on AutoDL: `/root/autodl-tmp/external_quick_logs/mech_ablation_20260512_163733`.
- SSAR seed1 eval20 check log root on AutoDL: `/root/autodl-tmp/external_quick_logs/ssar_seed1_eval20_20260512_230057`; local copy under `refine-logs/remote-results/ssar_seed1_eval20_20260512_230057/`.
- Trusted-selector log root on AutoDL: `/root/autodl-tmp/external_quick_logs/trusted_selector_top20_100k_20260512_183548.log`; local copy under `refine-logs/remote-results/trusted_selector_20260512_183548/`.
- Q-gap/consistency selector logs on AutoDL: `/root/autodl-tmp/external_quick_logs/qgap_soft_50k_retry_20260512_193926.log` and `/root/autodl-tmp/external_quick_logs/consistency_50k_20260512_194642.log`; local copy under `refine-logs/remote-results/qgap_consistency_20260512_194642/`.
- Value-baseline logs on AutoDL: `/root/autodl-tmp/external_quick_logs/iql_100k_20260512_203450.log` and `/root/autodl-tmp/external_quick_logs/cql_50k_20260512_204714.log`; local copy under `refine-logs/remote-results/value_baselines_20260512_2115/`.
- ATLAS labels/results on AutoDL: `/root/autodl-tmp/taoyao-rl/project/results/atlas_labels/`, `/root/autodl-tmp/taoyao-rl/project/results/atlas_50k/`, `/root/autodl-tmp/taoyao-rl/project/results/atlas_100k/`, and `/root/autodl-tmp/taoyao-rl/project/results/atlas_seed1_eval20/`; local copies under `refine-logs/remote-results/atlas_labels_20260512_2205/` and `refine-logs/remote-results/atlas_seed1_eval20_20260513_004052/`.
- ATLAS shuffled-label ablation local copy: `refine-logs/remote-results/atlas_label_ablation_shuffle_20260513_003101/`.
- ATLAS walker second-env check local copy: `refine-logs/remote-results/atlas_walker_seed0_eval10_20260513_023101/`.
- Minimal O2O fine-tuning local copy: `refine-logs/remote-results/o2o_minimal_20260513/`.
- P0 eval20 O2O panel local copy: `refine-logs/remote-results/o2o_p0_eval20_20260514/`.
- Q-filtered trust seed0 local copy: `refine-logs/remote-results/o2o_p1_qgate_eval20_20260514/`.
- Q-filtered trust seed1 local copy: `refine-logs/remote-results/o2o_p2_qgate_seed1_eval20_20260514/`.
- ATLAS seed1 O2O release/fixed local copy: `refine-logs/remote-results/o2o_p3_atlas_seed1_eval20_20260514/`.
