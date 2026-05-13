# Review Summary

**Problem**: offline-to-online RL under low-quality D4RL replay data.  
**Initial approach**: keep the original A/B/C family comparison, while using C-line ATLAS/SSAR evidence as the mechanism core.  
**Date**: 2026-05-13  
**Rounds**: 1 / 5  
**Final external score**: 6.45 / 10  
**Final external verdict**: REVISE

## Problem Anchor

- Bottom-line problem: The course project needs a defensible answer to how different constraint families affect offline-to-online RL under low-quality D4RL replay data, without turning the whole project into an overclaimed ATLAS-only algorithm paper.
- Must-solve bottleneck: Low-quality replay contains many suboptimal or misleading actions. Weak behavior regularization underuses useful data, but strong teacher/action constraints can over-constrain online adaptation. SSAR exposes a strong IQL-qv trusted-action signal, but that signal is expensive and not automatically safe to carry into online fine-tuning.
- Non-goals: Do not claim ATLAS is SOTA; do not run a 6-env x 3-seed benchmark before the mechanism is clear; do not vendor no-license third-party code; do not make broad PRDC/A2PR expansion the main contribution.
- Constraints: Course-project timeline; AutoDL budget should stay exploration-first; one or two replay environments and selective seeds are acceptable; A/B-line teammate results are still pending; C-line evidence is mostly smoke-to-mechanism validation, not final benchmark proof.
- Success condition: A coherent workshop-style project where A/B/C lines compare value conservatism, non-conservative contrast, and policy/trusted-action regularization under one protocol; C-line contributes a clear mechanism finding about trusted-action selection and online constraint release; remaining TODOs are narrow enough for teammates to fill.

## Round-by-Round Resolution Log

| Round | Main Reviewer Concerns | What This Round Simplified / Modernized | Solved? | Remaining Risk |
|---|---|---|---|---|
| 1 | ATLAS method underspecified; release schedule optional despite thesis; related work not current enough; C-line may swallow group paper | Pinned selector/loss/BC weight; promoted linear release to P0; reduced online test to 2x2; added related-work targets | Partial | A/B-line rows still missing; release schedule still needs final run/decision; citations need verification |

## Overall Evolution

- The proposal moved from "ATLAS as a promising local method" to "trusted-action constraints help offline but need release online."
- The dominant contribution is now clearer: label-quality decomposition plus online constraint-transfer diagnosis.
- The plan avoids broad sweeps and keeps PRDC/A2PR as reference-only.
- ATLAS is framed as a controlled probe/post-cache distillation component, not a SOTA replacement.

## Final Status

- Anchor status: preserved, with an explicit warning that A/B rows are still needed if the paper keeps the family-comparison title.
- Focus status: tighter than the initial proposal, but still dependent on group merge.
- Modernity status: improved; needs citation verification for PROTO, ENOTO, SUF, Adaptive BC, Cal-QL, IQL/AWR, ReBRAC, PRDC, A2PR, SSAR.
- Strongest parts of final method: aligned-vs-shuffled ATLAS label control, no-IQL SSAR ablation, minimal O2O slice showing online over-constraint.
- Remaining weaknesses: no external re-score after refinement; no final A/B rows; release schedule is specified but not newly executed in this refinement.
