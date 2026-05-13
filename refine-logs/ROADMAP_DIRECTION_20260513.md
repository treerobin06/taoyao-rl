# RL Project Roadmap Direction

Date: 2026-05-13
Status: working memo, not paper draft

## Why This Document Exists

现在还不适合直接写完整论文稿件。其他组员的 A 线、B 线还没有合并结果，整体 claim 不能提前定死。

当前更合适的产物是一个 Markdown 路线方向文档，用来固定：

- 我们这条 C 线到底在回答什么问题；
- 哪些实验已经支持当前判断；
- 哪些 claim 可以说，哪些不能说；
- 下一步优先跑什么，避免继续乱烧 baseline。

等三条线结果合并后，再把这里的内容迁移到 LaTeX 报告。

## Current Project-Level Framing

推荐保留原始项目主线：

> 低质量离线数据下，不同保守性 / 正则化设计如何影响 offline-to-online reinforcement learning 的迁移效率。

这条主线是课程项目最稳的骨架。它不是单纯追一个新算法，也不是只复现 baseline，而是比较不同 conservatism design 在低质量数据和在线微调阶段的表现。

## C-Line Positioning

C 线不应该抢 A 线的 CQL/value-conservatism 叙事，也不应该把 IQL 当作自己的核心创新。

C 线建议定位为：

> policy / behavior regularization family under low-quality replay data.

具体包括：

- TD3+BC：anchor baseline；
- ReBRAC-lite：强 regularization baseline；
- PRDC / A2PR：近年 policy regularization 参考，但当前 smoke 信号不强；
- SSAR：强但昂贵的 state-adaptive / trusted-action selection anchor；
- ATLAS：我们从 SSAR 机制诊断中提出的轻量 trusted-action selector 尝试。

## ATLAS Positioning

ATLAS 不能现在 claim 成新的 SOTA offline RL 算法。

更合理的表述是：

> SSAR 的强性能很大程度依赖 IQL-qv trusted action selection，但这个预筛过程成本高。ATLAS 尝试把这个 trusted-action signal 蒸馏成轻量 `(state, action) -> trust score` selector，再用于 TD3+BC-style weighted behavior regularization。

当前支持 ATLAS 的证据：

- hopper seed0/seed1 100k final 都在约 68-70；
- shuffled-label ablation 从 aligned 45.29 掉到 18.78，说明不是普通 weighted BC 分布起作用；
- walker2d second-env 上 ATLAS final 71.26，说明不是 hopper-only artifact；
- 但 walker 上 SSAR final 94.28，ATLAS 仍有明显差距。

所以 ATLAS 是“机制扩展 / 轻量化近似”，不是最终主线替代品。

## P0 Todo

1. 补一个 online fine-tuning 最小闭环。
   - 原始项目是 offline-to-online，现在我们 offline 结果多，online fine-tuning 证据少。
   - 先选 `hopper-medium-replay-v2`。
   - 方法优先级：TD3+BC / IQL or SSAR / ATLAS。
   - 目标是拿 online curve 和 sample efficiency，不是跑大 sweep。

2. 固定一张 C 线现有结果表。
   - 包含 TD3+BC、ReBRAC-lite、IQL、CQL、SSAR、ATLAS。
   - 列：env、seed、steps、eval episodes、final score、best score、interpretation。
   - 这张表先放 Markdown，后续再进 LaTeX。

3. 只补一个 ATLAS 高价值 ablation。
   - 优先：soft vs hard labels；
   - 或：`label_min_weight`；
   - 或：selector capacity。
   - 目标是解释为什么 ATLAS 与 SSAR 仍有差距。

4. 和组员同步边界。
   - A 线：CQL/value conservatism；
   - B 线：normal / non-conservative contrast，例如 PPO/SAC-style online baseline 或 vanilla TD3-style online fine-tuning；
   - C 线：policy regularization + SSAR/ATLAS mechanism；
   - 公共：统一 evaluator、统一图表、统一 offline-to-online 记录格式。

## P1 Todo

1. 做 cost/time 对比。
   - SSAR full IQL-qv preselection time；
   - ATLAS selector train/export time；
   - 同样 100k offline training 下的收益与成本。

2. 补 walker 上的 ATLAS ablation。
   - walker 上 ATLAS 与 SSAR 差距大，适合定位 selector/label usage 损失。

3. 等正信号明确后再补 seed。
   - 不要现在就回到 6 env x 3 seed。
   - seed 应该服务于关键 claim，而不是替代思考。

4. 准备两张图。
   - 一张结果曲线或柱状图；
   - 一张机制图：SSAR/IQL teacher -> trust labels -> ATLAS selector -> weighted TD3+BC.

## Markdown vs LaTeX Decision

当前阶段用 Markdown。

原因：

- 组员结果未合并，LaTeX 结构容易反复改；
- Markdown 更适合维护实验表、todo、claim 边界；
- 后续可以把 Markdown 中稳定的段落迁移到 LaTeX；
- 现在写 LaTeX 容易把未验证 claim 过早固化。

建议文件分层：

- `ROADMAP_DIRECTION_20260513.md`：当前路线方向；
- `EXPERIMENT_RESULTS.md`：事实表；
- `NOVELTY_CHECK_20260513.md`：novelty / claim 边界；
- later `paper/main.tex`：等三条线合并后再写正式稿。

## One-Sentence Message To Teammates

我们这边建议先不急着写完整论文稿，而是先固定路线和证据表：项目主线仍然是低质量离线数据下 conservatism / regularization 对 offline-to-online 迁移效率的影响；C 线主要负责 policy regularization，并把 SSAR 机制诊断和 ATLAS 轻量 trusted-action selector 作为扩展贡献。下一步优先补最小 online fine-tuning 闭环和少量关键 ablation，不做大规模多 seed / 多环境 sweep。
