# Result-to-Claim Verdict

**Date**: 2026-05-12
**Scope**: D4RL `hopper-medium-replay-v2`, seed0 exploration results through `trusted_td3_bc_top20`.
**Integrity status**: provisional; no `EXPERIMENT_AUDIT.json` exists.

## Intended Claim

SSAR's expensive IQL-qv trusted action selection is the dominant useful mechanism on the current replay setting, and the project should focus on a cheaper local substitute rather than expanding baseline sweeps.

## Verdict

- **claim_supported**: partial
- **confidence**: medium

## What The Results Support

The current evidence supports a mechanism-level diagnosis:

- SSAR cached IQL-qv is much stronger than the simple baselines at 100k: final `92.44`, best `100.98`.
- Removing SSAR's IQL-qv trusted action selection collapses performance: final `25.48`, best `30.34`.
- ReBRAC-lite is a useful cheap reference but remains far below cached SSAR at 100k: final `36.54`, best `54.36`.
- PRDC and A2PR official-source 50k screens do not show a useful signal on this setting: PRDC final `23.54`, A2PR final `22.31`.
- The local return-ranked selector `trusted_td3_bc_top20` can move the curve once, with best `45.13 @50k`, but does not stay stable to 100k: final `28.76`.

Supported working claim:

> On the current `hopper-medium-replay-v2` exploration setting, trusted action selection appears to be the main SSAR mechanism worth studying; naive trajectory-return filtering is insufficient, so the next contribution candidate should approximate trusted action selection with a cheap critic/Q-gap or behavior-consistency signal.

## What The Results Do Not Support

The results do **not** support these stronger claims:

- that `trusted_td3_bc_top20` is a final method;
- that a return-ranked trajectory mask solves the replay problem;
- that we have a paper-ready improvement across seeds or environments;
- that PRDC/A2PR are worth expanding in this project phase;
- that more TD3+BC baseline sweeps would create contribution by themselves.

## Missing Evidence

Required before claiming a method contribution:

1. A stronger cheap selector than return-ranked trajectories, preferably Q-gap or behavior consistency.
2. A 50k then 100k single-seed go/no-go on `hopper-medium-replay-v2`.
3. If positive, one stability check: seed1 on the same env or one second replay env.
4. If still positive, 3-seed confirmation and higher eval episode count.
5. Optional later: compare compute/time cost against full SSAR IQL-qv preselection.

## Suggested Claim Revision

Use the narrower claim now:

> Current exploration identifies SSAR-style trusted action selection as the useful mechanism and rules out naive return-ranked filtering; the next research target is a cheaper Q-aware trusted-action selector.

Do not claim:

> We already have a cheap replacement for SSAR.

## Next Experiments Needed

Priority order:

1. Implement `trusted_td3_bc_qgap_soft`: actor BC weight uses detached `sigmoid((Q(s,a_data)-Q(s,pi(s)))/temperature)`.
2. Run `trusted_td3_bc_qgap_soft`, 50k, `hopper-medium-replay-v2`, seed0.
3. If 50k beats ReBRAC-lite or clearly improves over return-ranked selector stability, extend to 100k.
4. If Q-gap fails, try behavior-consistency selector; do not expand seeds first.
5. Only after a positive 100k signal, run seed1 or second replay env.

## Route

Verdict is `partial`, so the correct route is supplementary experiments, not paper writing and not broad baseline expansion.
