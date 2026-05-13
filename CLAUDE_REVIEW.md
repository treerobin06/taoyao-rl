# Claude Review Report

## Round 1 - 2026-05-13 22:31 CST

**Skill**: `/claude-review`  
**Mode**: Claude CLI fallback (`claude -p ... --output-format json --permission-mode plan --tools ''`)  
**Claude session**: `b1fd72aa-c0ac-47ea-b26f-19cfe5597916`  
**Scope**:

- `idea-stage/IDEA_REPORT.md`
- `idea-stage/IDEA_REPORT_20260513_POSTHOC.md`
- `refine-logs/EXPERIMENT_RESULTS.md`

## Verdict

**Score: 5/10.** The idea is defensible for a course paper after reframing, but weak for a workshop-style claim without more evidence.

The reviewer's central judgment is:

> Current evidence contains a real signal, but the proposed title/framing is larger than the evidence. The strongest contribution is not yet "mechanistic offline-to-online transfer"; it is an offline label-quality and cheap-selector story.

## Main Findings

### S1 Critical: single-seed headline numbers are not reliable

Most headline results are still single-seed. The reviewer highlighted that cached SSAR varies heavily across seeds:

- SSAR cached 100k seed0: final 92.44, best 100.98
- SSAR cached 100k seed1: final 60.88, best 99.22

This variance means ATLAS-vs-baseline comparisons cannot be treated as robust method superiority yet.

### S2 Critical: "mechanistic study" is currently too strong

The existing evidence shows a phenomenon: trusted labels help offline, but do not trivially transfer online. It does not yet isolate the cause. To support a mechanism claim, the project needs controlled ablations such as random-subset and online shuffled-label controls.

### S3 High: novelty risk is mainly in the online-release framing

General online constraint release overlaps with Adaptive BC, PROTO, SUF, ENOTO, Cal-QL, and related offline-to-online RL work. The potentially defensible novelty is narrower:

> Distilling IQL-qv / SSAR teacher trust labels into a compute-cheap action-trust selector, then studying label quality.

### S4 High: online evidence does not support an online-improvement claim

Current offline-to-online results do not show ATLAS outperforming TD3+BC:

- TD3+BC decay: offline 22.43 -> online final/best 39.64
- ATLAS decay: offline 45.29 -> online final 37.97, best 38.26
- ATLAS fixed: offline 45.29 -> online final 29.22, best 32.27

The safe claim is only that offline trusted constraints may fail to transfer online without release/redesign.

### S5 Medium: evaluation protocol needs unification

Some results use eval5 and some use eval20. Cross-run comparison should not be used as a strong claim until paper-facing cells are rerun with the same evaluation protocol.

### S6 Medium: task coverage is narrow

Most evidence is on `hopper-medium-replay-v2`; walker has only one seed. This is enough for exploration, but not enough for broad D4RL generalization claims.

## Defensible Claims Now

1. **Action-level label quality matters.** Aligned ATLAS labels beat shuffled labels on hopper 50k: 45.29 vs 18.78/19.35.
2. **Cheap heuristics are not enough.** Return-ranked, Q-gap, and behavior-consistency selectors are unstable or fail.
3. **ATLAS is a cheap surrogate candidate for IQL-qv trust selection.** This is a practical, narrower contribution.
4. **Offline gain does not trivially transfer online.** This can be reported as an observation or limitation, not as a solved mechanism.

## Claims to Avoid

1. Do not claim ATLAS solves offline-to-online transfer.
2. Do not claim ATLAS robustly outperforms baselines without multi-seed evidence.
3. Do not claim a general mechanism without random-subset / shuffled-label / release controls.
4. Do not claim broad D4RL generalization from hopper plus one walker seed.

## Recommended Reframe

### Primary route

**ATLAS: A Cheap Action-Trust Selector Distilled from IQL-qv Labels**

This route focuses on offline label quality and compute-quality tradeoff:

- Full IQL-qv/SSAR trust signal is useful but expensive.
- A learned cheap selector can recover part of that benefit.
- Naive cheap selectors and shuffled labels fail, so the trust signal is not trivial.
- Online transfer remains a limitation/future direction.

### Secondary route

**When Teacher-Trust Signals Fail to Transfer Online**

This can work only as a controlled negative-result paper if the next experiments isolate why fixed trusted constraints hurt online adaptation.

## Minimum Next Experiments

The reviewer recommended, in order:

1. ATLAS hopper 3 seeds with eval20 under the same 50k offline + 10k online protocol.
2. One second D4RL replay task, preferably walker or halfcheetah, with 3 seeds eval20.
3. Online shuffled-label control.
4. Random-subset control using the same trust fraction as ATLAS.
5. Eval protocol cleanup for paper-facing table cells.

For a course paper, the practical minimum is:

1. Random-subset control.
2. Eval20 unification for key cells.
3. One more ATLAS seed or one second-env sanity check.

## Decision

The idea is not dead. The safe path is to stop presenting it as a broad online-transfer mechanism paper and instead write it as a focused empirical/method paper:

> IQL-qv/SSAR action-trust labels are valuable but expensive; ATLAS distills them into a cheaper selector; label quality is the actual mechanism; online adaptation exposes a failure mode rather than a solved contribution.

