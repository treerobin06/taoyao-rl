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

## Literature Landscape

### What is already known

- Offline-to-online 中“约束需要调整/放松”不是新结论。Adaptive BC 指出 BC-style constraints 能稳定 fine-tuning，但会把 policy 拉回 behavior policy、拖慢 online improvement。
- PROTO 直接把 policy regularization 做成随 online fine-tuning 演化、逐步放松的 regularization term。
- SUF 更进一步，主张 retained offline constraints 会伤害效率，应该稳定 unconstrained fine-tuning。
- ENOTO 从 Q-ensemble 角度处理 offline-to-online 中 degradation、slow improvement，并显式 loosen pessimism。
- SSAR 在 offline 阶段已经说明 uniform regularization 不合适，需要 selective state-adaptive regularization 和高质量 action selection。
- NeurIPS 2024 的 offline RL bottleneck 工作说明，value learning 不是唯一瓶颈，policy extraction 和 test-time state generalization 也可能更关键。

### What remains useful for us

泛泛说“release constraint”不够新；但我们现在的数据不是泛泛 BC constraint，而是更具体的：

1. **SSAR/IQL-qv action-level trust labels**；
2. 它们被蒸馏成 ATLAS selector 后 offline 有效；
3. label alignment ablation 很强；
4. online fine-tuning 反而暴露了 constraint conflict。

因此可以做的贡献不是“发明 release”，而是：

> **A mechanism study of how action-level trusted labels transfer, fail, and should be converted across offline pretraining and online fine-tuning.**

## Evidence We Already Have

| Evidence | Result | Supports |
|---|---:|---|
| TD3+BC 50k | 22.43 | weak behavior-regularized anchor |
| ReBRAC-lite 50k | 34.48 | simple regularization helps but limited |
| SSAR full IQL-qv 50k | 38.56 / 43.97 | trusted-action teacher has signal |
| cheap SSAR without IQL selection 100k | 25.48 / 30.34 | IQL-qv trusted selection is not disposable |
| IQL compact 100k | 45.27 / 81.28 | value backbone is strong; SSAR not only better than weak TD3+BC |
| ATLAS seed0 / seed1 100k | 69.97 / 68.11 | teacher trust can be distilled into a stable offline selector |
| ATLAS shuffled 50k | 18.78 vs aligned 45.29 | label alignment matters, not generic weighting |
| walker ATLAS | 71.26 vs SSAR 94.28 | signal survives second env but does not close teacher gap |
| TD3+BC O2O | 22.43 -> 39.64 | online fine-tuning works |
| ATLAS O2O decay | 45.29 -> 37.97 | offline advantage does not persist |
| ATLAS O2O fixed | 45.29 -> 29.22 | fixed teacher constraint over-constrains adaptation |

## Ranked Ideas

### Idea 1: Trusted-Action Transfer Failure and Release

**One-line thesis**: Action-level trusted labels are useful as offline supervision, but become an online adaptation bottleneck if kept as fixed constraints.

**Why this is top-ranked**:

- It uses the strongest existing evidence: aligned vs shuffled labels and ATLAS offline vs online failure.
- It connects directly to offline-to-online RL, not just offline RL.
- It explains a negative result rather than hiding it.
- It gives the project a real mechanism claim even if ATLAS does not beat SSAR.

**Novelty**: 5.5/10  
The general release/adaptive constraint problem is known. The more specific novelty is the action-level teacher trust transfer failure.

**Course-project value**: 9/10  
This is the cleanest story from our data.

**Closest prior work**:

- Adaptive BC: adaptive BC loss during online fine-tuning.
- PROTO: evolving policy regularization term.
- SUF: stable unconstrained fine-tuning without retained constraints.
- SSAR: selective state/action regularization in offline RL.

**Differentiation**:

Prior work mostly studies behavior/policy/value constraints as method-level objects. We can study whether **teacher-derived per-transition trusted labels** remain valid once online data arrives.

**Minimum next experiment**:

```text
hopper-medium-replay-v2, seed0
offline checkpoint: 50k
online horizon: 10k first, optionally 50k if needed

Compare:
- TD3+BC fixed / decay
- ATLAS fixed / decay
- ATLAS trust-release variants:
  1. offline-only trust, zero on online samples
  2. decayed offline trust
  3. confidence-gated trust only when Q_online agrees with teacher
```

**Success condition**:

- Positive: release/gating recovers ATLAS online final above TD3+BC online final.
- Negative but still useful: every ATLAS trust variant loses online, showing teacher trust is an offline stabilizer but not an O2O mechanism.

**Reviewer risk**:

- If written as “constraints should release online,” novelty is weak.
- It must be written as “action-level teacher trust does not transfer naively; here is the mechanism decomposition.”

**Verdict**: **Recommended main contribution**.

### Idea 2: Label-Quality Decomposition of Behavior Regularization

**One-line thesis**: Behavior regularization benefits depend on label/action alignment, not just on adding a weighted BC term.

**Why this is strong**:

- We already have a very strong ablation: aligned 45.29 vs shuffled 18.78.
- We also have failed cheap selectors: return-ranked 100k collapse, online Q-gap/consistency fail.
- This can be turned into a mechanism figure/table with little extra compute.

**Novelty**: 6/10  
Weighted BC and advantage weighting are old; but a controlled decomposition of SSAR/IQL-qv trust label alignment is more specific.

**Course-project value**: 8.5/10

**Closest prior work**:

- IQL / AWR: advantage-weighted policy extraction.
- SSAR: selective state-adaptive regularization and high-quality action selection.
- A2PR / PRDC: policy regularization toward selected or constrained dataset actions.

**Differentiation**:

We can show a hierarchy:

```text
generic BC weight distribution       fails
trajectory return ranking            unstable
online TD3 Q-gap / consistency        fails
SSAR/IQL-qv aligned action labels     works offline
```

This makes the claim more precise than “regularization helps.”

**Minimum next work**:

- No heavy run required.
- Add diagnostic plots:
  - trust score histogram;
  - aligned vs shuffled selected-action advantage distribution;
  - policy-action distance by trust bucket;
  - selected fraction and return distribution.
- Optional cheap run: random labels or constant labels if not already enough.

**Success condition**:

- The report can state: action-level teacher alignment is a necessary component of the offline gain.

**Reviewer risk**:

- Without O2O, it remains an offline mechanism result.
- It should be paired with Idea 1 for the final story.

**Verdict**: **Best low-cost support contribution**.

### Idea 3: Policy Extraction Bottleneck Under a Fixed Teacher Signal

**One-line thesis**: Given the same teacher/value/trust signal, the extraction operator determines whether the signal becomes a useful policy.

**Why this matters**:

- We have IQL high best but unstable final.
- SSAR can spike near 100 but final/tail varies.
- ATLAS has stable offline final but loses online.
- This suggests the issue may not only be teacher quality; it may be how the actor extracts and carries that signal.

**Novelty**: 6.5/10  
The broad bottleneck is known from NeurIPS 2024, but applying it to SSAR/IQL-qv trust transfer is a plausible contribution.

**Course-project value**: 7.5/10

**Closest prior work**:

- “Is Value Learning Really the Main Bottleneck in Offline RL?”: policy extraction can matter more than value learning.
- IQL/AWR: supervised advantage-weighted extraction.
- TD3+BC/ReBRAC: behavior-constrained policy gradient extraction.

**Minimum next experiment**:

Fix the same teacher labels/value and compare:

```text
1. weighted BC / ATLAS current
2. TD3+BC actor regularization with teacher weights
3. candidate action reranking using teacher Q or trust score
```

First version can be eval-only:

- sample N candidate actions from current policy or BC policy;
- select by teacher Q/trust;
- evaluate without long retraining if possible.

**Success condition**:

- If extraction choice changes score materially under the same teacher, we can claim policy extraction is the bottleneck in trusted-label transfer.

**Reviewer risk**:

- More implementation work.
- It may drift away from the original course deadline.

**Verdict**: **Best academic extension if there is time**.

### Idea 4: Cost-Quality Frontier for Teacher Trust

**One-line thesis**: Full SSAR/IQL-qv labels are expensive; partial teachers or selector distillation may preserve enough signal at lower cost.

**Why this is still useful**:

- We already paid the cost and saw strong SSAR/ATLAS signals.
- The practical contribution could be “how much teacher computation is necessary.”

**Novelty**: 5/10

**Minimum next experiment**:

```text
teacher budget: 10k / 20k / 50k / full IQL-qv
measure:
- label agreement with full teacher
- ATLAS 50k score
- wall-clock cost
```

**Reviewer risk**:

- More of an engineering/cost paper.
- Needs careful cost logging.

**Verdict**: **Good appendix/secondary idea, not mainline unless results are very clean**.

### Idea 5: Cross-Algorithm Reuse of Trusted Labels

**One-line thesis**: If trusted labels capture dataset quality, they should help more than one actor-update algorithm.

**Minimum next experiment**:

```text
same teacher labels
apply to:
- TD3+BC
- ReBRAC-lite
- maybe A2PR/PRDC if easy
```

**Novelty**: 5.5/10

**Reviewer risk**:

- If only one algorithm benefits, it becomes a negative finding.
- Integration cost can be annoying.

**Verdict**: **P1, only after Idea 1/2 are clean**.

## Eliminated or Low-Priority Directions

| Direction | Decision | Reason |
|---|---|---|
| “ATLAS is SOTA” | eliminate | not supported; SSAR walker and spikes remain stronger |
| Broad PRDC/A2PR expansion | eliminate | current 50k gate has no signal; low contribution |
| Multi-seed / 6-env baseline sweep | defer | expensive and does not answer mechanism |
| Generic regularization decay | weak alone | PROTO/Adaptive BC already cover broad release |
| Online TD3 Q-gap selector | eliminate | already failed |
| Behavior-consistency selector | eliminate | already failed |
| Return-ranked trust as method | eliminate | 100k collapse |
| Uncertainty selector | low priority | plausible but looks like generic ensemble trick unless tied to teacher-label transfer |

## Recommended Contribution Framing

### Strongest title direction

> **When Trusted Actions Stop Helping: A Mechanistic Study of Offline-to-Online Constraint Transfer**

### Claim stack

1. **Offline gain**: SSAR/IQL-qv action-level trusted labels contain useful supervision under low-quality replay.
2. **Label-quality mechanism**: aligned labels work; shuffled labels collapse; cheap heuristics fail.
3. **Transfer failure**: the same teacher constraint does not automatically improve online fine-tuning.
4. **Next mechanism**: online improvement requires release, reweighting, or a different extraction operator.

### What this contributes

This is a mechanism contribution, not a SOTA algorithm contribution:

> We decompose when an offline trusted-action signal helps, when it fails, and what must be changed before it becomes useful for online adaptation.

## First Three Next Actions

### P0. Write the paper around mechanism, not performance

Current evidence is already enough for a workshop/course mechanism paper if written carefully.

Required changes:

- rename ATLAS from “main method” to “diagnostic selector”;
- make O2O negative result central, not embarrassing;
- put release/extraction as future or next experiment, depending on time.

### P0. Add label-quality analysis plots

This is cheap and likely improves the paper more than another baseline.

Plot:

- aligned vs shuffled trust distribution;
- selected action advantage distribution;
- return-ranked / Q-gap / consistency vs teacher-label agreement;
- online degradation table.

### P1. Run one decisive O2O mechanism test

Do not run broad sweeps. Run one controlled test:

```text
ATLAS trust policy:
1. fixed offline trust
2. linear decay
3. online-gated trust: keep trust only if online critic agrees
4. optional: trust only for offline samples, zero trust for online samples
```

If this improves online final, we have a method. If not, we have a stronger negative mechanism result.

## Bottom Line

如果忽略历史，只看现有实验，最值得做的贡献是：

> **trusted-action signal 的机制分解：它为什么 offline 有用，为什么 online 直接迁移会失败，以及 release / extraction 可能怎么修。**

这比“我们提出 ATLAS 并提高指标”更稳，也比“复现一堆 baseline”更有内容。

## Source Anchors

- Adaptive BC Regularization: https://arxiv.org/abs/2210.13846
- PROTO: https://arxiv.org/abs/2305.15669
- SUF: https://ojs.aaai.org/index.php/AAAI/article/view/29083
- ENOTO: https://www.ijcai.org/proceedings/2024/615
- SSAR: https://proceedings.mlr.press/v267/luo25p.html
- Offline RL bottlenecks / policy extraction: https://proceedings.neurips.cc/paper_files/paper/2024/hash/8ffb4e3118280a66b192b6f06e0e2596-Abstract-Conference.html
- ReBRAC: https://proceedings.neurips.cc/paper_files/paper/2023/hash/26cce1e512793f2072fd27c391e04652-Abstract-Conference.html
