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

## Decision

- `rebrac_lite` passes the exploration-stage go gate on this one-seed smoke: it improves final normalized score by more than 10 points over `td3_bc`.
- `td3_bc_alpha5` should not be expanded.
- Next run should not be a broad sweep. Run `rebrac_lite` vs `td3_bc` on `hopper-medium-replay-v2` seeds 1/2 at 50k or 100k, then decide whether to add a second replay env.

## W&B Runs

- `bc`: https://wandb.ai/tree06/taoyao-rl/runs/8fl2xyfh
- `td3_bc`: https://wandb.ai/tree06/taoyao-rl/runs/d7lna4n6
- `td3_bc_alpha5`: https://wandb.ai/tree06/taoyao-rl/runs/oshuqift
- `rebrac_lite`: https://wandb.ai/tree06/taoyao-rl/runs/pnttw327

## Caveats

- This is one seed on one env, not a paper-ready claim.
- `rebrac_lite` is a compact PyTorch implementation for signal finding, not yet a strict official JAX ReBRAC reproduction.
- BC best@30k is high, so the next stage should compare full curves and final scores across seeds.
