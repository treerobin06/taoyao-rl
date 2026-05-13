# Post-hoc Idea Discovery Report

**Direction**: 从现有实验反推可贡献问题，不继承早期项目设想  
**Date**: 2026-05-13  
**Mode**: literature check + existing-pilot synthesis + reviewer-style triage  
**New GPU pilots launched**: none

## Executive Summary

如果完全忽略早期设想，只看现有实验，最有贡献潜力的方向不是“继续堆 baseline”或“宣称 ATLAS 是 SOTA”，而是：

> **Action-level trusted constraints are useful offline, but can become harmful during online adaptation unless we understand how the trust signal should be released, transformed, or extracted.**

这个方向的优势是：它直接由已有结果支撑。ATLAS offline seed0/seed1 接近 68-70，shuffled labels 掉到 18.78，说明 teacher-label alignment 真的重要；但 O2O 中 TD3+BC 从 22.43 升到 39.64，而 ATLAS 从 45.29 掉到 37.97，fixed ATLAS 掉到 29.22，说明 offline-useful trust 并不会自动转成 online-useful constraint。

这不是顶会级“全新问题”，因为 PROTO、Adaptive BC、SUF、ENOTO 已经研究过 offline-to-online 中约束/保守性如何调整。但我们的较窄贡献可以是：**针对 SSAR/IQL-qv 这种 action-level teacher trust signal，做 label quality、policy extraction、online interference 的机制分解。**

## Ranked Ideas

| Rank | Idea | Novelty | Course value | Status |
|---:|---|---:|---:|---|
| 1 | Trusted-action transfer failure and release | 5.5/10 | 9/10 | recommended main contribution |
| 2 | Label-quality decomposition of behavior regularization | 6/10 | 8.5/10 | best low-cost support |
| 3 | Policy extraction under a fixed teacher signal | 6.5/10 | 7.5/10 | best academic extension |
| 4 | Cost-quality frontier for teacher trust | 5/10 | 7/10 | secondary / appendix |
| 5 | Cross-algorithm reuse of trusted labels | 5.5/10 | 6.5/10 | P1 only |

## Recommended Contribution

The strongest framing is:

> **When Trusted Actions Stop Helping: A Mechanistic Study of Offline-to-Online Constraint Transfer**

Claim stack:

1. **Offline gain**: SSAR/IQL-qv action-level trusted labels contain useful supervision under low-quality replay.
2. **Label-quality mechanism**: aligned labels work; shuffled labels collapse; cheap heuristics fail.
3. **Transfer failure**: the same teacher constraint does not automatically improve online fine-tuning.
4. **Next mechanism**: online improvement requires release, reweighting, or a different extraction operator.

## First Three Next Actions

1. Write the paper around mechanism, not performance.
2. Add label-quality analysis plots.
3. Run one decisive O2O mechanism test: fixed vs decay vs online-gated trust.

See the timestamped full report:

`idea-stage/IDEA_REPORT_20260513_POSTHOC.md`
