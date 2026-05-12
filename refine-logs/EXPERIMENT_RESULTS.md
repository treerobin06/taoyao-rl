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

## Official-Source Screen

Same setting unless noted: `hopper-medium-replay-v2`, seed 0, 50k steps, eval every 10k.

| Source Method | Final Normalized Score | Best Normalized Score | Best Step | Status | Interpretation |
|---------------|------------------------|-----------------------|-----------|--------|----------------|
| PRDC official source | 23.54 | 23.54 | 50k | DONE | runs cleanly, but only matches/slightly exceeds TD3+BC at 50k |
| A2PR official source | 22.31 | 22.81 | 40k | DONE | runs cleanly, unstable early curve, no clear uplift over TD3+BC |
| SSAR official source-localized | 38.56 | 43.97 | 40k | DONE | strongest current signal; clean full IQL-qv preselection took about 62 min, total run 4003s |

## Decision

- `rebrac_lite` passes the exploration-stage go gate on this one-seed smoke: it improves final normalized score by more than 10 points over `td3_bc`.
- `td3_bc_alpha5` should not be expanded.
- PRDC/A2PR official-source 50k runs do not yet beat the ReBRAC-lite signal.
- SSAR full-IQL beats ReBRAC-lite on this one-seed screen: final 38.56 vs 34.48, best 43.97 vs 34.48. Treat SSAR as the strongest current modern baseline, but account for its expensive IQL-qv preselection.

## Mechanism Ablation

Same setting: `hopper-medium-replay-v2`, seed 0, 100k steps, eval every 10k.

| Variant | Final Normalized Score | Best Normalized Score | Best Step | Interpretation |
|---------|------------------------|-----------------------|-----------|----------------|
| SSAR cached IQL-qv | 92.44 | 100.98 | 90k | SSAR remains very strong when the full IQL-qv cache is reused; the 50k high score was not merely a one-off peak |
| cheap SSAR without IQL action selection | 25.48 | 30.34 | 90k | removing IQL-qv trusted action selection collapses performance near TD3+BC/ReBRAC-lite range |
| ReBRAC-lite | 36.54 | 54.36 | 90k | simple regularization can spike, but remains far below cached full SSAR at 100k |

Mechanism takeaway: SSAR's expensive IQL-qv trusted action selection appears to be a central contributor, not a disposable preprocessing detail. A promising contribution should target a cheaper substitute for this selection step or a way to amortize/cache it safely.

## W&B Runs

- `bc`: https://wandb.ai/tree06/taoyao-rl/runs/8fl2xyfh
- `td3_bc`: https://wandb.ai/tree06/taoyao-rl/runs/d7lna4n6
- `td3_bc_alpha5`: https://wandb.ai/tree06/taoyao-rl/runs/oshuqift
- `rebrac_lite`: https://wandb.ai/tree06/taoyao-rl/runs/pnttw327

## Caveats

- This is one seed on one env, not a paper-ready claim.
- `rebrac_lite` is a compact PyTorch implementation for signal finding, not yet a strict official JAX ReBRAC reproduction.
- BC best@30k is high, so the next stage should compare full curves and final scores across seeds.
- Mechanism ablation uses only 5 evaluation episodes, so 90k/100k spikes should be treated as exploration signal rather than paper-ready scores.
- PRDC and SSAR repos were used as external source checkouts because no license file was found locally; do not vendor their code into the public repo unless licensing is clarified.
- A cached/reduced SSAR run reached 22.22 at 50k, but it loaded a short preflight IQL-qv checkpoint; exclude it from official-source conclusions.
- Clean SSAR full-IQL log root on AutoDL: `/root/autodl-tmp/external_quick_logs/ssar_full_clean_20260512_101809`.
- Mechanism ablation log root on AutoDL: `/root/autodl-tmp/external_quick_logs/mech_ablation_20260512_163733`.
