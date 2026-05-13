# Official-Source Reproduction Plan

**Date**: 2026-05-12
**Stage**: exploration, one dataset first
**Primary setting**: `hopper-medium-replay-v2`, seed 0 plus targeted seed1 reliability check

## Goal

Use official or author-provided source code first, because the current project is still in exploration. The immediate goal is not to publish a clean unified implementation. The goal is to find out whether recent C-track methods produce real signal on the replay bottleneck, then decide which methods deserve deeper migration.

## Source Baselines

| Method | Source | Role | License / Sharing Policy | Current Status |
|--------|--------|------|--------------------------|----------------|
| PRDC | `LAMDA-RL/PRDC` | ICML 2023 dataset constraint regularization baseline | no license found locally; run as external source, do not vendor into public repo | preflight passed; 50k source run in progress/done on AutoDL |
| A2PR | `ltlhuuu/A2PR` | ICML 2024 adaptive advantage-guided policy regularization | MIT license; can later vendor small patches if needed | preflight passed; 50k source run in progress |
| SSAR | `QinwenLuo/SSAR` | ICML 2025 state-adaptive regularization | no license found locally; run as external source, do not vendor into public repo | clean full-IQL source-localized run done; best current signal |

## Localized Patch Policy

Allowed local changes during exploration:

- environment compatibility: `LD_PRELOAD`, MuJoCo path, import cleanup, Python dependency shims
- CLI/output localization: unified env/seed/steps, result directory, result parser
- logging integration: Aim/W&B adapters, JSON summaries, timestamps
- compute controls: shorter `offline_timesteps`, fewer eval episodes, or cached pretraining for smoke tests, if clearly labeled
- reproducibility helpers: commit hash, source patch note, command log

Not allowed to call an official-source result "official" if we change:

- objective terms or loss formulas
- action/state selection logic
- model architecture
- replay buffer semantics
- reward normalization semantics

If any of those are changed, label the run as `localized-variant`, not `source-repro`.

## Run Blocks

### S0: Source Compatibility

- Dataset: `hopper-medium-replay-v2`, seed 0
- Runs: 100-step preflight for PRDC, A2PR, SSAR
- Success: code imports D4RL, loads the dataset, performs at least one evaluation, and writes logs
- Status: done
- Notes: SSAR required removing an unused `gym.envs.classic_control.acrobot.bound` import and adding an optional `SSAR_IQL_STEPS` smoke override.

### S1: Fast Source Screen

- Dataset: `hopper-medium-replay-v2`, seed 0
- Runs:
  - PRDC, 50k steps, eval every 10k
  - A2PR, 50k steps, eval every 10k
  - SSAR quick-localized, 50k offline steps with reduced IQL-qv preselection
- Purpose: get the first comparable source-code signal without waiting for full paper-scale settings
- Success gate: any method exceeds TD3+BC by 5-10 normalized score, or shows a clearly better curve than `rebrac_lite`
- Interpretation: directional only; SSAR quick-localized is not a strict reproduction if IQL-qv is shortened

### S2: Full-Logic SSAR Screen

- Dataset: `hopper-medium-replay-v2`, seed 0
- Runs:
  - SSAR TD3+BC backbone with full official IQL-qv preselection (`SSAR_IQL_STEPS=1000000`)
  - offline training at 50k first; extend to 100k if the curve is still rising
- Purpose: distinguish "SSAR logic is expensive but useful" from "SSAR only looks good under a shortcut"
- Expected cost: roughly 1-2 GPU-hours based on preflight throughput, still below the current small-budget threshold
- Success gate: final or best normalized score beats TD3+BC and approaches/exceeds the ReBRAC-lite smoke signal
- Status: done on 2026-05-12
- Result: final 38.56, best 43.97 @40k on `hopper-medium-replay-v2`, seed 0
- Cost note: clean IQL-qv preselection plus 50k offline training took 4003s on the retained AutoDL instance
- Seed0 cache to keep: `/root/autodl-tmp/external_repos/SSAR/model/iql_qv/hopper-medium-replay-v2/0/0.7_model.pth`
- Seed0 cache backup: `/root/autodl-tmp/taoyao-rl-cache/SSAR/iql_qv/hopper-medium-replay-v2/seed0/0.7_model.pth`
- Seed0 cache checksum: `dffa751dd22177b0161baa0bd5661517984644fbfe7afb27fb1065a3eb8c0579`
- Seed1 cache backup: `/root/autodl-tmp/taoyao-rl-cache/SSAR/iql_qv/hopper-medium-replay-v2/seed1/0.7_model.pth`
- Seed1 cache checksum: `53dd12638216579de50a2449ad7c598ffe9f97f85c7975ee71583fb1694a08fd`
- Walker seed0 cache backup: `/root/autodl-tmp/taoyao-rl-cache/SSAR/iql_qv/walker2d-medium-replay-v2/seed0/0.7_model.pth`
- Walker seed0 cache checksum: `ffe559a043a2f0ef5814d7cfeb18d6ce2960120ebddeb7fffde0eb31ef3aeb64`

### S3: Migration Candidate Selection

- Keep at most two methods for deeper local integration:
  - one strongest official-source method
  - one simplest strong method, currently `rebrac_lite` unless PRDC/A2PR/SSAR clearly dominate
- Expand only winners:
  - `hopper-medium-replay-v2`, seeds 1/2
  - one second replay dataset, likely `walker2d-medium-replay-v2` or `halfcheetah-medium-replay-v2`
- Stop rule: do not expand a method that fails on seed0 unless it gives a new mechanism-level clue.

## Immediate Execution Order

1. Parse all source logs into one JSON/Markdown summary. Done.
2. Decide whether the next real experiment is "SSAR seed expansion", "second replay env", or "local migration/caching of SSAR".
3. If continuing SSAR, keep and reuse the IQL-qv cache instead of deleting it each time.
4. Do not rerun clean full-IQL for the same env/seed unless intentionally testing IQL-qv variance.

## Current Decision Rule

- If PRDC/A2PR are weak but cheap, they remain reference baselines.
- If SSAR full-logic wins but is expensive, keep it as the strongest modern baseline and later optimize/cached-pretrain it. Current seed0 result supports this path.
- If no official-source method beats `rebrac_lite`, the project should pivot from "add more SOTA baselines" to "understand why ReBRAC-style regularization helps replay data".

## Mechanism Ablation Queue

Launched on 2026-05-12:

- `SSAR_cached_100k`: reuse the cached IQL-qv model and extend offline training to 100k. This tests whether the 40k SSAR peak was transient.
- `cheap_SSAR_no_iql_select_100k`: set `select_actions=false` while keeping SSAR's state-adaptive beta update. This tests how much of SSAR comes from expensive IQL-qv trusted action selection.
- `ReBRAC_lite_100k`: same env/seed/steps simple strong baseline, used to judge whether cheap regularization can stay competitive.

Remote log root: `/root/autodl-tmp/external_quick_logs/mech_ablation_20260512_163733`.

Results:

| Variant | Final | Best | Interpretation |
|---------|------:|-----:|----------------|
| `SSAR_cached_100k`, seed0, eval5 | 92.44 | 100.98 | strong upper anchor, but high-variance and not a stable final estimate |
| `SSAR_seed1_100k`, eval20 | 60.88 | 99.22 | confirms near-100 spike is possible, but final/tail is unstable |
| `cheap_SSAR_no_iql_select_100k` | 25.48 | 30.34 | without trusted action selection, SSAR loses most gains |
| `ReBRAC_lite_100k` | 36.54 | 54.36 | useful simple baseline, but does not match cached SSAR |

Updated decision: the most valuable contribution direction is not broad SOTA collection, but replacing or amortizing SSAR's expensive IQL-qv trusted action selection with a lighter and more stable mechanism. Do not cite seed0 `92.44/100.98` as a stable baseline level; cite it as a high-variance upper anchor.

## Second Replay Env Anchor

Completed on 2026-05-13:

| Env | Seed | Eval Episodes | Final | Best | Best Step | Cache |
|-----|-----:|--------------:|------:|-----:|----------:|-------|
| `walker2d-medium-replay-v2` | 0 | 10 | 94.28 | 94.60 | 40k | preserved under `/root/autodl-tmp/taoyao-rl-cache/SSAR/iql_qv/walker2d-medium-replay-v2/seed0/` |

Use this as the walker teacher anchor for ATLAS optimization. Do not rerun walker IQL-qv unless intentionally testing teacher variance.
