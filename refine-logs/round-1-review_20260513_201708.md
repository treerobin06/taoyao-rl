# Round 1 Review

Reviewer: Claude CLI fallback because `claude-review` MCP tools were not visible in this Codex session.  
Subject: `refine-logs/round-0-initial-proposal_20260513_201708.md`  
Verdict: **REVISE**  
Overall score: **6.45 / 10**

## Parsed Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 8 |
| Method Specificity | 6 |
| Contribution Quality | 6 |
| Frontier Leverage | 5 |
| Feasibility | 8 |
| Validation Focus | 7 |
| Venue Readiness | 7 |
| Overall | 6.45 |

## Main Reviewer Concerns

1. Method specificity is still too loose.
   - Pin selector architecture, loss, trust-label derivation, and BC weighting form.
   - Pin online fine-tuning protocol, including buffer mix, exploration noise, eval cadence, and release schedule.
2. Contribution quality is promising but currently diagnostic-only.
   - ATLAS as a probe is honest, but the paper becomes stronger if constraint release is promoted from optional to P0.
   - If release helps, the story becomes diagnosis plus partial fix; if it fails, the negative finding is still sharper.
3. Frontier positioning needs direct neighbors.
   - Add related-work positioning against offline-to-online methods about policy regularization, unconstrained fine-tuning, Q-ensembles, and adaptive BC regularization.
   - Explicitly connect or differentiate ATLAS from AWR/IQL-style advantage-weighted learning.
4. Drift risk remains.
   - The anchor is group-level A/B/C family comparison, but the actual contribution is drifting toward C-line mechanism.
   - Either keep A/B as critical path, or explicitly make C-line mechanism the main paper and A/B supporting.

## Raw Reviewer Response

<details>
<summary>Full Claude CLI reviewer response</summary>

# Independent Review: Round-0 Refine Proposal

**Subject**: `refine-logs/round-0-initial-proposal_20260513_201708.md`  
**Lens**: Anchor-fidelity over module-count; mechanism clarity over benchmark breadth.

## Dimension Scores

| Dim | Score | Weight | Weighted |
|---|---:|---:|---:|
| 1. Problem Fidelity | 8 | 15% | 1.20 |
| 2. Method Specificity | 6 | 25% | 1.50 |
| 3. Contribution Quality | 6 | 25% | 1.50 |
| 4. Frontier Leverage | 5 | 15% | 0.75 |
| 5. Feasibility | 8 | 10% | 0.80 |
| 6. Validation Focus | 7 | 5% | 0.35 |
| 7. Venue Readiness | 7 | 5% | 0.35 |
| **Overall** |  |  | **6.45 / 10** |

## 1. Problem Fidelity: 8/10

The anchor is clear and largely preserved: low-quality D4RL replay; weak BC underuses data; strong trust constraints may block online adaptation; SSAR/IQL-qv is strong but expensive; non-goals are explicit. The main deduction is that the success condition still depends on pending A/B-line teammate results, creating a project-level contract risk.

## 2. Method Specificity: 6/10

Strong points: the selector signature `g_phi(s,a) -> [0,1]` is fixed, aligned-vs-shuffled labels are a clean control, and failure-mode diagnostics are unusually concrete.

Weak points:

- selector implementation is abstract;
- loss, label binarization, thresholding, and optimizer settings are not pinned;
- the way `g_phi` enters TD3+BC is underspecified;
- release schedule is treated as optional despite being central to the online-transfer thesis;
- online protocol details are missing.

Concrete fixes:

1. Lock selector architecture, e.g. `[256, 256, ReLU] -> sigmoid` over concatenated state/action.
2. Use BCE on binarized trust labels derived from IQL-qv / SSAR labels; leave continuous advantage regression as stretch.
3. Pin actor BC form: `L_BC = sum_i w_i || pi(s_i)-a_i ||^2`, with `w_i = clip(g_phi(s_i,a_i), 0,1)`.
4. Promote linear release schedule to P0: `lambda(t)=lambda_0 * max(0, 1 - t/K)`.
5. Define online protocol: buffer mix, exploration noise, eval cadence, and eval episodes.

Priority: IMPORTANT.

## 3. Contribution Quality: 6/10

Strong points: reframing ATLAS as a controlled instrument rather than SOTA is correct. Shuffled-label collapse plus no-IQL-SSAR collapse form a strong decomposition.

Weak points:

- the dominant contribution is still mostly diagnostic;
- transfer failure is known in offline-to-online RL, so the label-quality decomposition needs emphasis;
- without a positive or negative release intervention, C-line risks becoming "we observed a failure mode".

Concrete fixes:

1. Add a second-environment shuffled-label control if cheap.
2. Promote release schedule to P0.
3. Rewrite the claim around action-level trust labels and online release.

Priority: IMPORTANT.

## 4. Frontier Leverage: 5/10

Strong points: the right method family is named: CQL/Cal-QL, ReBRAC, PRDC, A2PR, SSAR, IQL.

Weak points:

- related work needs direct offline-to-online constraint-release neighbors;
- the proposal should connect to AWR/IQL advantage-weighted policy learning;
- constraint annealing or adaptive BC regularization should be discussed.

Concrete fixes:

1. Add a related-work paragraph around PROTO, ENOTO, Cal-QL O2O, AWR/IQL, and adaptive behavior cloning regularization after citation verification.
2. State that ATLAS is trained from cached teacher labels rather than on-policy advantage, which is the cleaner differentiation.

Priority: IMPORTANT.

## 5. Feasibility: 8/10

The proposal is feasible because it reuses existing infrastructure, avoids broad sweeps, uses one canonical env plus one sanity env, and keeps A/B-line dependency explicit. The main operational risk is that promoting release schedule to P0 adds extra runs, but this is still plausible within a retained AutoDL session.

## 6. Validation Focus: 7/10

The claims are reasonably claim-driven. Suggested improvements:

- add numeric thresholds for label-alignment evidence;
- define online-transfer failure as both below ATLAS offline endpoint and below TD3+BC online final;
- add a release-schedule claim once it is promoted.

Priority: MINOR.

## 7. Venue Readiness: 7/10

For a course/workshop paper, the scope is acceptable and honest. A top-tier version would need more environments, a positive intervention, deeper related-work positioning, and stronger theory or mechanism evidence.

## Simplification Opportunities

1. Remove state/action confidence threshold from the headline plan; linear decay is enough.
2. Express the online experiment as a clean 2x2: `{ATLAS, TD3+BC} x {fixed, linear-decay}`.
3. Treat walker2d shuffled-label control as a cheap strengthening run, not a full protocol expansion.
4. Fold label-control gap into the main numeric result rather than listing it as a separate metric.

## Modernization Opportunities

1. Use continuous IQL-qv advantage regression as a stretch target and binarized labels as the main controlled version.
2. Add a no-pretrain warm-start baseline if cheap.
3. State that 50k/10k is compute-bounded and intentionally used for controlled ablation.
4. Add direct offline-to-online related work after verification.

## Drift Warning

Mechanism-vs-comparison drift: the anchor is an A/B/C constraint-family comparison, while the proposal's contribution is moving toward C-line mechanism. The group should decide whether A/B are critical-path results or supporting context.

Diagnostic-vs-intervention drift: the thesis implies online constraint release, but the plan currently treats release as optional. If release is not tested, the paper becomes mainly observational.

## Verdict

**REVISE.**

The proposal is workshop-acceptable after two important revisions and one minor revision:

- pin selector and loss details;
- promote release schedule to P0;
- add frontier related-work positioning.

</details>
