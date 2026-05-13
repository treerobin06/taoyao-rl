# Taoyao RL Agent Guide

给 Claude / Codex / Cursor 等 AI 助手用。这里记录本 RL 项目的当前状态、实验纪律和下一步方向。不要把全局个人规则、token、API key 写进这个文件。

## Project Identity

- Repo: `https://github.com/treerobin06/taoyao-rl`
- Local path: `/Users/robin/Desktop/taoyao/RL/project`
- Remote AutoDL path: `/root/autodl-tmp/taoyao-rl/project`
- External source checkouts on AutoDL: `/root/autodl-tmp/external_repos/`
- Project type: offline RL / offline-to-online RL group project
- Current stage: **C-line P0 offline-to-online panel is complete; current contribution is a constraint-transfer-gap mechanism study, not robust ATLAS superiority**

## Canonical Docs

Read these first before proposing or launching experiments:

1. `refine-logs/EXPERIMENT_RESULTS.md` - latest result summary and decisions
2. `refine-logs/EXPERIMENT_TRACKER.md` - run tracker, TODOs, deferred experiments
3. `refine-logs/SOURCE_REPRO_PLAN.md` - official-source reproduction policy, SSAR cache paths
4. `refine-logs/remote-results/` - parsed remote run summaries

If these docs conflict with older README text, trust the `refine-logs/` files.

## Current Experimental Conclusion

The exploration-stage baseline sweep is sufficient. Do **not** spend more mainline compute on broad baseline expansion unless Tree explicitly asks.

Key `hopper-medium-replay-v2`, seed 0 results:

| System | Final | Best | Current Role |
|--------|------:|-----:|--------------|
| TD3+BC | 22.43 | 22.43 | weak anchor |
| TD3+BC alpha5 | 21.95 | 21.95 | do not expand |
| ReBRAC-lite 50k | 34.48 | 34.48 | simple strong baseline |
| ReBRAC-lite 100k | 36.54 | 54.36 | simple baseline with spikes |
| PRDC official source 50k | 23.54 | 23.54 | reference only; do not expand now |
| A2PR official source 50k | 22.31 | 22.81 | reference only; do not expand now |
| SSAR full-IQL 50k | 38.56 | 43.97 | strongest modern baseline after full IQL-qv |
| SSAR cached-IQL 100k seed0 eval5 | 92.44 | 100.98 | high-variance upper anchor, not stable final estimate |
| SSAR full-IQL 100k seed1 eval20 | 60.88 | 99.22 | near-100 spike repeats, but final/tail is unstable |
| cheap SSAR without IQL selection 100k | 25.48 | 30.34 | ablation showing IQL-qv selection is critical |
| IQL compact 100k | 45.27 | 81.28 | strong value baseline; include in any fair SSAR comparison |
| CQL compact 50k | 39.81 | 39.81 | conservative value anchor; not a C-line expansion target |
| ATLAS teacher-label selector 100k | 69.97 | 69.97 | strongest local contribution candidate so far |
| ATLAS teacher-label selector seed1 eval20 100k | 68.11 | 68.11 | passed first narrow stability check |
| ATLAS shuffled-label control 50k | 18.78 | 19.35 | label-quality ablation; teacher alignment matters |
| SSAR walker2d full-IQL 100k seed0 eval10 | 94.28 | 94.60 | second-env teacher anchor |
| ATLAS walker2d teacher-label selector 100k seed0 eval10 | 71.26 | 77.86 | second-env ATLAS signal survives, but gap to SSAR remains |

Main offline takeaway:

> The likely contribution is not another baseline table. IQL is already a strong value baseline, and SSAR/IQL-qv can spike very high but has unstable tail scores. The sharper question is whether we can cheaply reproduce or amortize the trusted-action signal in a lighter, more stable way. ATLAS passed the first offline stability and label-quality checks, while the O2O panel reveals the harder problem: offline trusted-action gains do not automatically transfer online.

Offline-to-online update, `hopper-medium-replay-v2`, 50k offline + 10k online, eval20:

| System | Seed | Offline Final | Online Final | Online Best | Current Role |
|--------|-----:|--------------:|-------------:|------------:|--------------|
| TD3+BC release | 0 | 22.20 | 40.06 | 40.06 | weak offline anchor, improves online |
| ATLAS release | 0 | 46.70 | 37.50 | 38.79 | strong offline, loses final online |
| SSAR/IQL-qv release | 0 | 50.71 | 28.87 | 31.85 | strongest offline decay endpoint, worst online final |
| SSAR/IQL-qv fixed | 0 | 50.71 | 38.61 | 96.22 | transient spike, unstable tail |
| ATLAS Q-gate fixed | 0 | 46.70 | 48.41 | 48.41 | repairs seed0 fixed-constraint failure |
| TD3+BC release | 1 | 20.19 | 98.86 | 98.86 | exposes high Hopper O2O variance |
| ATLAS release | 1 | 31.21 | 84.35 | 84.35 | strong seed1, but below TD3+BC |
| ATLAS Q-gate fixed | 1 | 31.21 | 39.91 | 70.03 | better than fixed, worse than release |

O2O takeaway:

> Teacher labels help offline initialization, but carrying fixed teacher regularization into online fine-tuning can trap adaptation. Q-filtered trust is a useful diagnostic hypothesis on seed0, but seed1 prevents claiming robust method dominance.

## Current TODO

The next agent should work on paper integration and A/B-line alignment, not broad C-line baseline collection.

1. ATLAS seed1 stability check is done.
   - Seed1 eval20 final/best is `68.11`, close to seed0 final/best `69.97`.
   - Next highest-value check is one second replay env, not more seeds on hopper yet.
2. Label-quality ablation is done for shuffled labels.
   - Shuffled labels preserve the score distribution but drop to 18.78 final at 50k.
   - This supports that the gain comes from IQL/SSAR teacher alignment, not generic weighted BC.
3. `walker2d-medium-replay-v2` seed0 second-env check is done.
   - SSAR full IQL-qv final/best: `94.28 / 94.60`.
   - ATLAS final/best: `71.26 / 77.86`.
   - This supports cross-env ATLAS signal, but leaves a large optimization gap.
4. P0 O2O and Q-filtered trust diagnostics are done.
   - The paper should claim a constraint-transfer gap and a promising diagnostic gate.
   - It should not claim ATLAS or Q-gate is a robust SOTA online fine-tuning method.

Do not unblock broad multi-seed/multi-env until ATLAS survives at least one stability check.
ATLAS has survived one seed check and one second-env check; broad sweeps are still deferred until the walker gap is better explained.

Optimization priority after the O2O panel: only run more C-line compute if it directly diagnoses online constraint transfer. The cheapest meaningful follow-ups are an A-line O2O anchor, a B-line non-conservative online contrast, or a narrow Q-gate/reset ablation; do not rerun PRDC/A2PR or broad seed sweeps.

## Explicitly Deferred

- PRDC multi-seed / multi-env expansion
- A2PR multi-seed / multi-env expansion
- 6 D4RL env x 3 seed table
- more TD3+BC alpha sweeps
- full paper-ready seed sweep before there is a contribution mechanism

Reason: these are expensive and currently unlikely to create the paper contribution.

## AutoDL / Cache Discipline

AutoDL is used as a retained project instance. If continuing this project, **power off but do not release** unless Tree explicitly asks to release.

Known instance:

- AutoDL instance id: `pro-7785f027d673`
- Remote project: `/root/autodl-tmp/taoyao-rl/project`
- Remote external sources: `/root/autodl-tmp/external_repos/`

SSAR IQL-qv cache to preserve:

- Seed0 primary: `/root/autodl-tmp/external_repos/SSAR/model/iql_qv/hopper-medium-replay-v2/0/0.7_model.pth`
- Seed0 backup: `/root/autodl-tmp/taoyao-rl-cache/SSAR/iql_qv/hopper-medium-replay-v2/seed0/0.7_model.pth`
- Seed0 SHA256: `dffa751dd22177b0161baa0bd5661517984644fbfe7afb27fb1065a3eb8c0579`
- Seed1 backup: `/root/autodl-tmp/taoyao-rl-cache/SSAR/iql_qv/hopper-medium-replay-v2/seed1/0.7_model.pth`
- Seed1 SHA256: `53dd12638216579de50a2449ad7c598ffe9f97f85c7975ee71583fb1694a08fd`
- Walker seed0 backup: `/root/autodl-tmp/taoyao-rl-cache/SSAR/iql_qv/walker2d-medium-replay-v2/seed0/0.7_model.pth`
- Walker seed0 SHA256: `ffe559a043a2f0ef5814d7cfeb18d6ce2960120ebddeb7fffde0eb31ef3aeb64`

Do not delete or overwrite this cache unless intentionally testing IQL-qv variance. A clean full IQL-qv run took about 62 minutes before the 50k offline training.

Before launching remote runs:

```bash
/Users/robin/.agent-context/skills/autodl-pro/scripts/autodl-dev.sh list
/Users/robin/.agent-context/skills/autodl-pro/scripts/autodl-dev.sh ssh pro-7785f027d673 -- 'nvidia-smi'
```

## Source Code Policy

Use official source during exploration, but keep licensing clean.

- A2PR has MIT license; small patches can be considered later.
- PRDC and SSAR had no license file in the checked source; run them as external source checkouts, but do not vendor their code into the public GitHub repo unless licensing is clarified.
- Compatibility-only local patches are allowed for reproduction:
  - MuJoCo / `LD_PRELOAD`
  - unused import cleanup
  - logging / parser scripts
  - eval step controls clearly labeled as smoke/localized
- Do not call a modified result "official" if objective, architecture, replay semantics, reward normalization, or selection logic changes. Label it as `localized-variant`.

## Codebase Rules

For local algorithms in this repo:

- Use `common.data.D4RLDataset` for D4RL loading.
- Use `common.eval.eval_episodes` and `common.eval.write_result` for evaluation and JSONL output.
- Use `common.seed.set_seed`.
- Do not add new D4RL env names outside `envs.txt` without discussion.
- Do not commit D4RL data, checkpoints, `.aim/`, `wandb/`, or raw experiment JSONL under `results/`.
- Do not hardcode W&B keys, AutoDL tokens, personal SSH info, or proxy credentials.

This project targets Linux + CUDA + Python 3.10. macOS is for editing and analysis, not main training.

## Useful Commands

Baseline smoke:

```bash
ENV=hopper-medium-replay-v2 SEED=0 STEPS=50000 bash scripts/run_c_track_smoke.sh
```

Mechanism ablation runner on retained AutoDL:

```bash
bash scripts/run_mech_ablation_autodl.sh
```

This script runs:

- `SSAR_cached_100k`
- `cheap_SSAR_no_iql_select_100k`
- `ReBRAC_lite_100k`

It is a record of the mechanism ablation, not the default next experiment. For next contribution work, implement a new cheap trusted selector instead of rerunning this script unchanged.

## Experiment Logging

- Aim is the default local tracker.
- W&B is optional and user-level; credentials should be configured with `wandb login`, never committed.
- For external-source runs, copy parsed JSON summaries into `refine-logs/remote-results/` instead of committing large raw logs or checkpoints.

## Git / Sharing

The public repo is used for sharing scripts, docs, and lightweight parsed results. Push useful docs and reusable scripts, but do not push private credentials, raw datasets, checkpoints, or no-license third-party source trees.

When making a meaningful experiment decision, update:

1. `refine-logs/EXPERIMENT_RESULTS.md`
2. `refine-logs/EXPERIMENT_TRACKER.md`
3. `refine-logs/SOURCE_REPRO_PLAN.md` if source/caching policy changes

## Current Stop/Go Rule

Stop doing baseline reproduction when it does not create a new mechanism insight. Go only when a run directly answers:

- Can ATLAS survive a second seed or second replay env?
- Does teacher-label quality matter versus random/return-only labels?
- Can we quantify the cost/performance tradeoff against full SSAR IQL-qv preselection?
- Can label weighting or selector capacity close the walker gap without rerunning expensive IQL-qv?
- Does an online adaptation rule release trusted-action constraints when the online critic disagrees?

That is now the C-line project question.
