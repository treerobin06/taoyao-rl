# C-Line Results Table

Date: 2026-05-13  
Scope: C-line exploration results for low-quality D4RL replay settings.  
Status: working evidence table, not final paper table.

## How To Read This Table

These results are exploration-stage evidence. They are meant to guide the next experiments, not to serve as the final multi-seed benchmark.

Current interpretation:

- Broad baseline expansion is not the priority.
- `hopper-medium-replay-v2` is the main weak/diagnostic environment.
- SSAR is the strongest teacher anchor but has expensive IQL-qv preselection and unstable tail scores.
- ATLAS is the strongest local contribution candidate, but should be framed as a lightweight approximation of SSAR/IQL trusted-action information, not as a SOTA replacement.

## Main Hopper Replay Results

Environment: `hopper-medium-replay-v2`.

| Method | Seed | Steps | Eval eps | Final | Best | Best step | Current role | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|---|
| BC | 0 | 50k | 5 | 17.86 | 32.26 | 30k | cheap anchor | High variance; not a strong baseline |
| TD3+BC | 0 | 50k | 5 | 22.43 | 22.43 | 50k | weak anchor | Basic policy-regularization anchor |
| TD3+BC alpha5 | 0 | 50k | 5 | 21.95 | 21.95 | 50k | negative ablation | Stronger Q term did not help |
| ReBRAC-lite | 0 | 50k | 5 | 34.48 | 34.48 | 50k | simple strong baseline | Clear improvement over TD3+BC |
| PRDC official source | 0 | 50k | 5 | 23.54 | 23.54 | 50k | reference only | Runs cleanly, no clear signal |
| A2PR official source | 0 | 50k | 5 | 22.31 | 22.81 | 40k | reference only | Runs cleanly, no clear signal |
| SSAR full IQL-qv | 0 | 50k | 5 | 38.56 | 43.97 | 40k | modern teacher anchor | Strongest 50k source-screen result, but IQL-qv preselection is expensive |
| SSAR cached IQL-qv | 0 | 100k | 5 | 92.44 | 100.98 | 90k | high-variance upper anchor | Strong but should not be reported as stable final level |
| SSAR full IQL-qv | 1 | 100k | 20 | 60.88 | 99.22 | 90k | reliability check | Near-100 spike repeats, tail/final remains unstable |
| cheap SSAR no IQL selection | 0 | 100k | 5 | 25.48 | 30.34 | 90k | mechanism ablation | Removing trusted action selection collapses performance |
| ReBRAC-lite | 0 | 100k | 5 | 36.54 | 54.36 | 90k | same-budget simple baseline | Can spike, but remains below SSAR/ATLAS |
| IQL compact | 0 | 100k | 5 | 45.27 | 81.28 | 80k | public value baseline | Strong baseline; needed for fair SSAR/ATLAS interpretation |
| CQL compact | 0 | 50k | 5 | 39.81 | 39.81 | 50k | conservative value anchor | Useful cross-line reference |
| trusted TD3+BC top20 | 0 | 50k | 5 | 45.13 | 45.13 | 50k | negative selector probe | Return-ranked trust can help briefly |
| trusted TD3+BC top20 | 0 | 100k | 5 | 28.76 | 45.13 | 50k | negative selector probe | Peak does not persist; do not scale this selector |
| trusted TD3+BC qgap-soft | 0 | 50k | 5 | 19.94 | 22.05 | 10k | failed cheap selector | Online TD3 critic is too noisy |
| trusted TD3+BC consistency | 0 | 50k | 5 | 19.77 | 28.00 | 40k | failed cheap selector | Behavior-consistency alone is insufficient |
| ATLAS teacher-label selector | 0 | 50k | 5 | 45.29 | 45.29 | 50k | contribution candidate | Beats ReBRAC-lite/CQL final at 50k |
| ATLAS teacher-label selector | 0 | 100k | 5 | 69.97 | 69.97 | 100k | contribution candidate | Positive seed0 100k signal |
| ATLAS teacher-label selector | 1 | 100k | 20 | 68.11 | 68.11 | 100k | stability check | Close to seed0 final; not a seed0 artifact |
| ATLAS shuffled-label control | 0 | 50k | 5 | 18.78 | 19.35 | 40k | label-quality ablation | Label alignment matters; not generic weighted BC |

## Walker Replay Check

Environment: `walker2d-medium-replay-v2`.

| Method | Seed | Steps | Eval eps | Final | Best | Best step | Current role | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|---|
| SSAR full IQL-qv | 0 | 100k | 10 | 94.28 | 94.60 | 40k | second-env teacher anchor | Strong teacher; walker is not teacher-limited |
| ATLAS teacher-label selector | 0 | 100k | 10 | 71.26 | 77.86 | 90k | second-env ATLAS check | Signal survives beyond hopper, but gap to SSAR remains |

## Minimal Offline-to-Online Check

Environment: `hopper-medium-replay-v2`.  
Setting: seed0, 50k offline + 10k online fine-tuning, eval5.

| Method | Offline final | Online final | Online best | Best online step | Current role | Interpretation |
|---|---:|---:|---:|---:|---|---|
| TD3+BC O2O decay | 22.43 | 39.64 | 39.64 | 60k | O2O anchor | Online fine-tuning improves over offline; decay slightly beats fixed |
| TD3+BC O2O fixed | 22.43 | 36.78 | 36.78 | 60k | O2O control | Fixed regularization also improves, but slightly below decay |
| ATLAS O2O decay | 45.29 | 37.97 | 38.26 | 56k | ATLAS O2O probe | ATLAS offline advantage does not persist in this first online setup |
| ATLAS O2O fixed | 45.29 | 29.22 | 32.27 | 58k | ATLAS O2O control | Fixed ATLAS regularization is worse than decay, suggesting over-constraint |
| ATLAS O2O decay, online fraction 0.25 | 45.29 | 31.46 | 43.73 | 54k | ATLAS O2O ablation | Lower online replay fraction gives a transient spike but worse final |

## Current Claims Supported

Supported:

- SSAR/IQL-qv trusted action selection is a high-value mechanism on replay data.
- Naive return-ranked trust and online Q-gap/consistency selectors are not enough.
- ATLAS benefits from aligned SSAR/IQL teacher labels, as shown by the shuffled-label control.
- ATLAS has passed one seed check on hopper and one second replay environment check.

Not supported:

- ATLAS is a new SOTA offline RL algorithm.
- PRDC/A2PR are worth broad expansion in the current project phase.
- More broad baseline sweeps will create the main contribution by themselves.

## Next Evidence Needed

P0:

1. Do not claim ATLAS O2O improvement under the current runner; keep ATLAS as offline/mechanism contribution.
2. One targeted ATLAS offline ablation, preferably soft-vs-hard labels or `label_min_weight`.

P1:

1. SSAR IQL-qv preselection cost vs ATLAS selector cost.
2. Walker ATLAS ablation to explain the gap to SSAR.
