# Taoyao RL Agent Guide

给 Claude / Codex / Cursor 等 AI 助手用。这里记录本 RL 项目的当前状态、实验纪律和下一步方向。不要把全局个人规则、token、API key 写进这个文件。

## Project Identity

- Repo: `https://github.com/treerobin06/taoyao-rl`
- Local path: `/Users/robin/Desktop/taoyao/RL/project`
- Remote AutoDL path: `/root/autodl-tmp/taoyao-rl/project`
- External source checkouts on AutoDL: `/root/autodl-tmp/external_repos/`
- Project type: offline RL / offline-to-online RL group project
- Current stage: **baseline/reproduction enough; contribution exploration should start**

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
| SSAR cached-IQL 100k | 92.44 | 100.98 | strongest current signal, but 5-episode eval and one seed |
| cheap SSAR without IQL selection 100k | 25.48 | 30.34 | ablation showing IQL-qv selection is critical |

Main takeaway:

> The likely contribution is not another baseline table. It is a cheaper, stable, reproducible replacement or amortization strategy for SSAR's expensive IQL-qv trusted-action selection.

## Current TODO

The next agent should work on contribution exploration, not baseline collection.

1. Design 2-3 cheap trusted-action selector candidates.
   - Possible directions: short critic warmup, return-ranked trajectory filter, behavior-consistency filter, Q-gap proxy.
2. Implement one minimal selector in local editable code.
   - Prefer compatibility with current `common/` data and eval APIs.
   - Keep output as a trusted mask, beta weight, or selector artifact that can be compared against SSAR.
3. Run only `hopper-medium-replay-v2`, seed 0, 50k/100k first.
   - Compare against TD3+BC, ReBRAC-lite, cheap SSAR no-IQL, and SSAR cached.
4. Only if this local selector has signal, run seed1 or a second replay env.

Do not unblock multi-seed/multi-env until a local selector beats ReBRAC-lite or clearly narrows the gap to SSAR.

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

- Primary: `/root/autodl-tmp/external_repos/SSAR/model/iql_qv/hopper-medium-replay-v2/0/0.7_model.pth`
- Backup: `/root/autodl-tmp/taoyao-rl-cache/SSAR/iql_qv/hopper-medium-replay-v2/seed0/0.7_model.pth`
- SHA256: `dffa751dd22177b0161baa0bd5661517984644fbfe7afb27fb1065a3eb8c0579`

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

- Can we replace IQL-qv trusted selection cheaply?
- Can we amortize/cached-pretrain it safely?
- Can a simple local selector approach SSAR while staying much cheaper?

That is now the C-line project question.
