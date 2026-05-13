# ATLAS Walker2d Second-Env Check

Date: 2026-05-13

Purpose: test whether ATLAS is hopper-specific by running one second replay environment, `walker2d-medium-replay-v2`, seed0, eval10.

## Results

| Run | Final | Best | Best Step | Notes |
|-----|------:|-----:|----------:|-------|
| SSAR full IQL-qv anchor | 94.28 | 94.60 | 40k | full 1M-step IQL-qv preselection plus 100k offline training |
| ATLAS teacher-label selector | 71.26 | 77.86 | 90k | distilled selector from the walker SSAR/IQL-qv cache |

## Teacher / Selector Diagnostics

- Cache: `/root/autodl-tmp/taoyao-rl-cache/SSAR/iql_qv/walker2d-medium-replay-v2/seed0/0.7_model.pth`
- Cache SHA256: `ffe559a043a2f0ef5814d7cfeb18d6ce2960120ebddeb7fffde0eb31ef3aeb64`
- Teacher hard trust fraction: `0.3678`
- Teacher trust score mean: `0.3824`
- Selector validation accuracy after 5 epochs: `0.6833`
- ATLAS score mean/std: `0.3706 / 0.1742`

## Interpretation

ATLAS survives a second replay env but does not close the gap to the full SSAR teacher on walker2d. This supports ATLAS as a real cross-env signal, not just a hopper-only artifact, while also showing clear optimization room in teacher-label usage or selector quality.

The next optimization should target label usage before broader sweeps: soft-vs-hard labels, `label_min_weight`, and selector capacity are the cheapest useful ablations because the expensive walker cache now exists.
