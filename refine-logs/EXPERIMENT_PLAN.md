# Experiment Plan

**Problem**: SSAR 在 `hopper-medium-replay-v2` 上很强，但它依赖约 1 小时的 IQL-qv trusted action preselection；更重要的是，P0 offline-to-online eval20 结果显示：trusted-action labels 帮助 offline initialization，但固定带入 online fine-tuning 会出现 constraint-transfer gap。
**Method Thesis**: 用 ATLAS 把 cached SSAR/IQL-qv trusted-action label 蒸馏成轻量 selector，用 P0/P1/P2/P3 O2O panel 研究这些 trusted constraints 什么时候该保留、什么时候该释放。当前论文主线是机制诊断，不是 ATLAS 稳定 SOTA claim，也不是继续扩大 PRDC/A2PR 或多 seed baseline sweep。
**Date**: 2026-05-12

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|-----------------------------|---------------|
| C1: SSAR 的 trusted action selection 是当前最值得替代的关键机制 | 这决定项目贡献是否从“复现 baseline”转向“更便宜的机制” | cached SSAR 明显强于 no-IQL ablation 和 ReBRAC-lite；已有结果支持机制诊断 | B1 |
| C2: online cheap Q-aware selector by itself is insufficient | return-ranked selector 100k 不稳定，online Q-gap/behavior consistency 也失败；这说明 selector 需要更强 teacher signal | Q-gap/behavior selector failed 50k; use as negative evidence, not next mainline | B2, B3 |
| C3: cached SSAR/IQL-qv teacher labels can be amortized into a lightweight selector | 这是当前最像贡献的 offline 方向：省掉重复 IQL-qv preselection，同时保留部分 trusted-action gain | ATLAS seed0/seed1 finals are both near 68-70; shuffled-label control collapses to 18.78 | B5 |
| C4: offline trusted-action gains do not automatically transfer online | 这是当前最像 workshop contribution 的机制发现：teacher constraints 是 offline catalyst，但 online adaptation 需要 gating/release | P0 eval20: TD3+BC release final 40.06, ATLAS/SSAR stronger offline but lower final; Q-gate seed0 positive, seed1 high variance | B6 |

## Paper Storyline

- Main paper must prove: trusted-action labels 是低质量 replay 下有效的 offline supervision，但存在 constraint-transfer gap；在线阶段需要 adaptive release/gating，而不是固定继承 teacher regularization。
- Appendix can support: seed 扩展、第二 replay 环境、eval episode 增加、cache/checksum 细节。
- Experiments intentionally cut: PRDC/A2PR 多 seed、多环境展开；6 env x 3 seed baseline 表；更多 TD3+BC alpha sweep；return-ranked selector 的 seed 扩展。

## Experiment Blocks

### Block 1: Mechanism Diagnosis Baseline

- Claim tested: C1。
- Why this block exists: 固定当前事实基础，证明“trusted action selection 是主要机制”，不是普通 regularization 或更多 baseline sweep。
- Dataset / split / task: D4RL `hopper-medium-replay-v2`，seed0，offline training。
- Compared systems: `td3_bc`, `rebrac_lite`, SSAR cached IQL-qv, cheap SSAR no-IQL, `trusted_td3_bc_top20`。
- Metrics: normalized score final/best；wall-clock and preselection cost；curve stability。
- Setup details: 已完成；100k、eval every 10k、5 eval episodes。
- Success criterion: SSAR cached clearly beats no-IQL and ReBRAC-lite；return-ranked selector not stable enough to scale。
- Failure interpretation: 如果 SSAR 优势不来自 IQL-qv selection，则转向其他 SSAR 组件；当前结果不支持这个失败解释。
- Table / figure target: internal diagnosis table; later paper motivation/ablation table。
- Priority: MUST-RUN, DONE。

### Block 2: Cheap Q-Gap Selector

- Claim tested: C2。
- Why this block exists: Q-gap 是最接近 SSAR trusted action selection 的便宜替代：它直接比较 dataset action 和 policy action 的 critic value，而不是只看 trajectory return。
- Dataset / split / task: `hopper-medium-replay-v2`，seed0。
- Compared systems: `trusted_td3_bc_qgap_soft` vs `trusted_td3_bc_top20`, `rebrac_lite`, `td3_bc`, cheap SSAR no-IQL。
- Metrics: normalized score final/best；curve stability after 50k；BC weight statistics；extra runtime。
- Setup details: TD3+BC critic full-data training不变；actor loss 中 BC 权重用 detached `sigmoid((Q(s,a_data)-Q(s,pi(s)))/temperature)`；先 50k，pass 后 100k。
- Success criterion: 50k 不低于 ReBRAC-lite，100k final/best 明显优于 return-ranked selector；运行时间接近 TD3+BC/ReBRAC，不引入 IQL-qv 预训练。
- Failure interpretation: 如果 Q-gap 不稳，说明当前 critic 质量不足以在线选择 trusted action，需要 warmup 或更保守的 selector。
- Table / figure target: next internal go/no-go table；若成功，进入 paper main candidate。
- Priority: MUST-RUN。

### Block 3: Behavior-Consistency Selector

- Claim tested: C2。
- Why this block exists: 如果 Q-gap 对 critic 噪声敏感，behavior consistency 是更保守的低成本替代。
- Dataset / split / task: `hopper-medium-replay-v2`，seed0。
- Compared systems: `trusted_td3_bc_consistency` vs Q-gap selector and ReBRAC-lite。
- Metrics: normalized score；selected/trusted fraction；policy-action distance distribution。
- Setup details: warmup 后按 `||pi(s)-a_data||` 或 percentile 生成 BC weight；先 50k，不直接 100k。
- Success criterion: 比 return-ranked selector 更稳定，且不显著落后 ReBRAC-lite。
- Failure interpretation: 如果仍不稳，cheap selector 需要更强的 learned value target，转向 light-IQL/amortized cache。
- Table / figure target: appendix or negative-results note。
- Priority: NICE-TO-HAVE, only if Q-gap fails or is ambiguous。

### Block 4: Minimal Stability Gate

- Claim tested: C2。
- Why this block exists: 单 seed 只能筛方向，不能写 paper claim。
- Dataset / split / task: first `hopper-medium-replay-v2` seed1, then one second replay env if seed1 is positive。
- Compared systems: only the winning cheap selector, `rebrac_lite`, and SSAR cached/no-IQL anchor if cache is available。
- Metrics: mean/std normalized score；best and final score；runtime。
- Setup details: 100k, eval every 10k, initially 5 eval episodes; increase eval episodes only for paper-facing confirmation。
- Success criterion: positive direction survives seed1 or second env。
- Failure interpretation: seed0 effect is likely overfit/noisy; return to selector design, not broader sweeps。
- Table / figure target: paper main candidate only after positive result。
- Priority: READY after Block 5; keep as one narrow validation run, not a broad sweep。

### Block 5: ATLAS Teacher-Label Selector

- Claim tested: C3。
- Why this block exists: online Q-gap/consistency selectors failed, but SSAR/IQL-qv trusted labels are already known to contain useful action-quality information. ATLAS tests whether that signal can be distilled into a small reusable selector.
- Dataset / split / task: `hopper-medium-replay-v2`，seed0。
- Compared systems: `trusted_td3_bc_atlas` vs ReBRAC-lite, IQL, CQL, cached SSAR, cheap SSAR no-IQL, return-ranked selector。
- Metrics: normalized score final/best；selector validation accuracy；label weight stats；wall-clock cost。
- Setup details: export SSAR/IQL-qv `Q-V` labels from cache; train MLP `(s,a)->trust score`; feed predicted label file into `trusted_td3_bc.py --selector_mode label_file`。
- Result: 50k final/best `45.29`; 100k final/best `69.97`。
- Success criterion: beats ReBRAC-lite/CQL and does not collapse by 100k。
- Failure interpretation: if later env fails, ATLAS may be hopper-specific and needs second-env redesign or light-IQL stabilizers.
- Priority: MUST-RUN, DONE for seed0/seed1 plus shuffled-label ablation; next is one second replay env。

### Block 6: Offline-to-Online Constraint Transfer

- Claim tested: C4。
- Why this block exists: 原项目标题和汇报需要 offline-to-online；只看 offline final 不足以回答迁移问题。
- Dataset / split / task: `hopper-medium-replay-v2`，50k offline + 10k online，eval20。
- Compared systems: TD3+BC release/fixed, ATLAS release/fixed, random matched trust subset, SSAR/IQL-qv release/fixed, ATLAS/SSAR Q-filtered trust diagnostics。
- Metrics: offline final/best, online final/best, best online step, final delta。
- Result: TD3+BC seed0 release online final 40.06；ATLAS seed0 release 37.50；SSAR/IQL-qv seed0 release 28.87；ATLAS Q-gate seed0 fixed 48.41；seed1 TD3+BC release 98.86 and ATLAS release 84.35。
- Success criterion: not method superiority; clear diagnosis of whether offline teacher constraints transfer online。
- Current interpretation: teacher labels help offline, but fixed teacher regularization can trap online adaptation. Q-gate is promising on seed0 but not robust under seed1。
- Priority: MUST-RUN, DONE。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 | Keep current state reproducible | local py_compile; confirm AutoDL instance is shutdown; preserve SSAR cache | code imports and prior results are readable | no GPU | remote/local divergence if not later committed |
| M1 | Implement Q-gap selector | add `trusted_td3_bc_qgap_soft` in local TD3+BC-style code | 2-step remote smoke passes | local + minutes on GPU | critic-value weighting may be noisy early |
| M2 | Q-gap 50k go/no-go | `hopper-medium-replay-v2`, seed0, 50k | final/best beats TD3+BC/no-IQL and is competitive with ReBRAC-lite | about 3-5 min GPU, <¥1 including overhead | one eval spike may mislead |
| M3 | Q-gap 100k stability | same variant, 100k | final remains meaningfully above 30s or best clearly exceeds ReBRAC-lite with stable tail | about 5-8 min GPU, <¥1-2 | instability after 50k |
| M4 | Minimal stability | seed1 or second replay env | only run if M3 positive | about 10-20 min GPU | expanding too early wastes budget |
| M5 | ATLAS seed0 50k/100k | teacher-label selector on `hopper-medium-replay-v2`, seed0 | final/best beats ReBRAC-lite/CQL and does not collapse by 100k | about 10 min GPU after labels | DONE; final/best 69.97 @100k |
| M6 | ATLAS narrow stability | seed1 eval20 | positive direction survives outside seed0 | done | final/best 68.11 @100k |
| M7 | ATLAS second replay env | `walker2d-medium-replay-v2`, seed0, SSAR cache + ATLAS 100k | positive direction is not hopper-only | about 1 GPU-hour if teacher cache must be created | DONE; ATLAS final 71.26 / best 77.86 vs SSAR final 94.28 / best 94.60 |
| M8 | ATLAS optimization ablations | tune how teacher labels are used, not more baselines | improve or explain M5/M7 behavior | cheap after cache exists | READY; run only 1-2 ablations first |
| M9 | P0 O2O eval20 panel | test whether offline trusted constraints transfer online | TD3+BC improves online while ATLAS/SSAR lose final score | completed | DONE |
| M10 | Q-filtered trust diagnostic | test adaptive release/gating after external review | seed0 positive, seed1 high variance | completed | DONE |

## Compute and Data Budget

- Total estimated GPU-hours for must-run M1-M3: under 0.5 GPU-hour on retained AutoDL 4090.
- Expected cost: likely under ¥2 for the next Q-gap gate, assuming no environment reinstall.
- Data preparation needs: none; reuse D4RL and existing AutoDL environment.
- Human evaluation needs: none.
- Biggest bottleneck: algorithm design, not compute.

## Risks and Mitigations

- Risk: Q-gap uses the same critic being trained and may be unreliable early.
- Mitigation: detach weights, log Q-gap distribution, optionally add warmup or EMA threshold.
- Risk: 5 eval episodes cause noisy normalized scores.
- Mitigation: use them only for go/no-go; increase eval episodes after a stable positive signal.
- Risk: remote GitHub is behind local bug fix/results.
- Mitigation: keep local as source of truth for now; push a small fix commit only when sharing is needed.

## Final Checklist

- [x] Main paper claim is not yet asserted
- [x] Novelty target is isolated: cheap trusted-action selection
- [x] Simplicity is defended by cutting baseline sweeps
- [x] Frontier contribution is not claimed
- [x] Nice-to-have runs are separated from must-run runs

## Execution Update: 2026-05-12 Evening

The first cheap online selectors did not pass the 50k gate:

| Variant | Final | Best | Decision |
|---------|-------|------|----------|
| `trusted_td3_bc_qgap_soft` | 19.94 | 22.05 @10k | stop; do not run 100k |
| `trusted_td3_bc_consistency` | 19.77 | 28.00 @40k | stop; do not run 100k |

Revised next direction: do not expand these online selector variants. If continuing this line, redesign toward a stronger offline value source: light critic pretraining, cached SSAR/IQL-qv labels, or a cheaper amortized label model.

## Execution Update: 2026-05-12 Late Evening

ATLAS passed the seed0 gate:

| Variant | Steps | Final | Best | Decision |
|---------|-------|-------|------|----------|
| `trusted_td3_bc_atlas` | 50k | 45.29 | 45.29 @50k | pass first gate |
| `trusted_td3_bc_atlas` | 100k | 69.97 | 69.97 @100k | pass stability gate |

Revised next direction: ATLAS is now the strongest local contribution candidate. The shuffled-label control and seed1 stability check are done; next do one second replay env before any broad seed/env sweep.

## Execution Update: 2026-05-13 After Midnight

Two key checks completed:

| Check | Result | Decision |
|-------|--------|----------|
| Shuffled-label control | final 18.78, best 19.35 | teacher-label alignment matters |
| ATLAS seed1 eval20 | final/best 68.11 @100k | first narrow stability check passed |

Next direction: do not add more hopper seeds immediately. The next high-value run is a second replay environment to test whether ATLAS is hopper-specific.

## Execution Update: 2026-05-13 Walker Second Env

The second replay environment check completed on `walker2d-medium-replay-v2`, seed0, eval10.

| Variant | Final | Best | Best Step | Decision |
|---------|------:|-----:|----------:|----------|
| SSAR full IQL-qv anchor | 94.28 | 94.60 | 40k | strong teacher anchor; walker is not teacher-limited |
| `trusted_td3_bc_atlas` | 71.26 | 77.86 | 90k | cross-env ATLAS signal survives, but gap to SSAR remains large |

Decision: ATLAS is no longer only a hopper result. It has one seed-stability check and one second-env check. It is still not paper-ready as a final method because walker shows a 16-23 point gap to full SSAR. The next work should be targeted ATLAS optimization/ablation, not broad baseline expansion.

## Optimization Space: ATLAS

If the second replay env is positive or ambiguous, optimize the teacher-label usage before spending on broader sweeps:

| Knob | Current Setting | Cheap Test | Why It Could Help |
|------|-----------------|------------|-------------------|
| label target | selector trains on `hard_trust` | train on continuous `trust_score` or normalized `Q-V` advantage | hard threshold may discard useful confidence information |
| BC floor | `label_min_weight=0.05` | compare `0.0`, `0.02`, `0.10` for 50k | too much floor dilutes trusted labels; too little may hurt coverage |
| binarization | use soft `atlas_score` weights | compare `LABEL_BINARIZE=1` at threshold 0.5 | if teacher labels are truly binary, soft scores may add calibration noise |
| selector capacity | 2-layer MLP hidden 256, 5 epochs | hidden 512 or 10 epochs, same cache | current val acc around 0.72 suggests remaining approximation error |
| cross-seed reuse | train per seed | train seed0 selector, apply to seed1 or second env only if dimensions match | tests whether ATLAS is a reusable selector or just cache compression |

Do not run all of these immediately. First priority after M7 is the smallest explanatory pair: soft-vs-hard label usage and BC floor. These are cheap because the expensive IQL-qv cache already exists.

## Execution Update: 2026-05-14 O2O P0/P1/P2/P3

The required offline-to-online panel is complete.

| Check | Result | Decision |
|-------|--------|----------|
| P0 eval20 seed0 | TD3+BC release final 40.06; ATLAS release final 37.50; SSAR/IQL-qv release final 28.87 | offline teacher gains do not directly transfer online |
| SSAR fixed seed0 | best 96.22, final 38.61 | near-expert spike exists but tail is unstable |
| Q-gate seed0 | ATLAS qgate fixed final/best 48.41 | first positive online adaptation signal |
| Seed1 check | TD3+BC release final 98.86; ATLAS release final 84.35; ATLAS qgate fixed final 39.91 | Hopper O2O variance is high; no robust superiority claim |

Revised next direction: the C-line baseline work is sufficient for a workshop mechanism draft. Next compute should be A-line/B-line comparable O2O anchors or a very targeted online constraint-transfer diagnostic, not more C-line baseline expansion.
