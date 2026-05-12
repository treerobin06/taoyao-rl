# Cheap Trusted-Selector Plan

Date: 2026-05-12

## Why This Is The Next Mainline

The current exploration suggests that SSAR's main practical advantage comes from its IQL-qv trusted action selection. On `hopper-medium-replay-v2` seed0, cached SSAR reached `92.44` final / `100.98` best at 100k, while removing IQL selection collapsed to `25.48` final / `30.34` best. ReBRAC-lite remains a useful cheap reference at `36.54` final / `54.36` best.

So the next contribution target is not another baseline table. It is a cheaper, easier-to-reproduce selector that captures part of the trusted-action effect without paying the one-hour IQL-qv preselection cost for every env/seed.

## First Local Candidate

`trusted_td3_bc_top20`:

- keep TD3+BC critic training on the full offline dataset;
- infer trajectories from D4RL transition continuity and terminal flags;
- rank trajectories by return;
- apply the actor BC regularizer mainly to transitions from the top 20% of transitions by trajectory return;
- keep a small BC weight (`0.05`) on the rest to avoid fully discarding coverage.

This is intentionally a cheap mechanism probe. It asks whether a simple return-ranked trust mask can narrow the gap between ReBRAC-lite and SSAR.

## First Run

Run only:

- env: `hopper-medium-replay-v2`
- seed: `0`
- steps: `50k`
- eval episodes: `5`

Decision rule:

- if it is clearly below ReBRAC-lite, stop and redesign selector;
- if it beats ReBRAC-lite or trends toward SSAR, extend to 100k;
- only after a positive 100k signal, consider seed1 or a second replay env.

## Deferred Selectors

- short critic-warmup selector: rank actions by a cheap early critic instead of IQL-qv;
- behavior-consistency selector: trust states where learned policy and dataset action agree after warmup;
- Q-gap proxy: trust dataset action when its Q is close to or above the policy action.

These are deferred until the return-ranked selector gives us a concrete first reference point.
