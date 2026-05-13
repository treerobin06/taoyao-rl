# Refinement Report

**Problem**: offline-to-online RL under low-quality replay data.  
**Initial approach**: use C-line ATLAS/SSAR experiments to support the broader A/B/C family-comparison course project.  
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

## Output Files

- Initial proposal: `refine-logs/round-0-initial-proposal.md`
- Round 1 review: `refine-logs/round-1-review.md`
- Round 1 refinement: `refine-logs/round-1-refinement.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Final proposal: `refine-logs/FINAL_PROPOSAL.md`
- Score history: `refine-logs/score-history.md`

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 | 6 | 6 | 5 | 8 | 7 | 7 | 6.45 | REVISE |

## Round-by-Round Review Record

| Round | Main Reviewer Concerns | What Was Changed | Result |
|---|---|---|---|
| 1 | Method underspecified; release schedule optional; related work under-positioned; C-line/group-level drift | Pinned selector/loss/objective/protocol; made linear release P0; added verified related-work targets; kept A/B rows as critical final merge | Partial |

## Final Proposal Snapshot

- The group paper should stay framed as a constraint-family comparison unless A/B results fail to arrive.
- C-line's defensible contribution is not "ATLAS beats SSAR"; it is "aligned trusted-action labels help offline, but fixed teacher regularization can block online adaptation."
- ATLAS is a post-cache distillation/probe mechanism.
- One more C-line experiment, if any, should test release under a clean 2x2 protocol rather than expanding baselines.

## Method Evolution Highlights

1. ATLAS changed from a loose selector idea into a pinned supervised trust model with a specific weighted TD3+BC objective.
2. Online release moved from optional to P0 because it directly tests the main observed failure mode.
3. The paper claim narrowed from generic conservatism to label-quality plus constraint-transfer diagnosis.

## Pushback / Drift Log

| Round | Reviewer Said | Author Response | Outcome |
|---|---|---|---|
| 1 | Add more walker2d label-control evidence if cheap | Accept as P1, not P0 | Avoids pulling C-line back into sweep mode before A/B merge |
| 1 | Add state/action confidence threshold gate | Reject for now | Linear release is the smaller adequate mechanism |
| 1 | Decide whether A/B are critical or supporting | Accept | Final title depends on A/B arrival |

## Remaining Weaknesses

- The revised proposal has not been externally re-scored.
- A/B rows are still missing.
- Citation verification is still required before final paper writing.
- Release schedule is specified as the next decisive test but not executed in this refinement pass.

## Raw Reviewer Responses

See `refine-logs/round-1-review.md`.

## Next Steps

1. Update the paper draft around the refined thesis.
2. Ask teammates for A/B rows in the shared table schema.
3. If running more C-line compute, run the release-schedule test before any broad sweep.
