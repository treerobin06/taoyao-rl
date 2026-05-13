# Team Next Actions

Date: 2026-05-13  
Purpose: make the three-person RL project executable again.

## Current Problem

现在三个人都不清楚自己要做什么，原因不是实验太少，而是有三件事混在了一起：

1. 原始项目主线：低质量离线数据下，不同 conservatism / regularization 设计如何影响 offline-to-online 迁移效率。
2. C 线探索中出现的新机制：SSAR / ATLAS / trusted-action selection。
3. LJY 提到的新想法：Gate-aware release rule。

课程项目角度，先不要让 2 和 3 替代 1。最稳的做法是：主线保持 1，2 和 3 作为 mechanism / discussion / bonus contribution。

## One-Sentence Project Question

低质量 offline replay 数据下，保守 value 方法、普通非保守在线方法、policy regularization 方法，在 offline-to-online fine-tuning 中分别有什么优劣？

## Three-Person Split

| Person | Track | Role | What to run first | What not to do now |
|---|---|---|---|---|
| KX | A: value conservatism | CQL / Cal-QL style conservative value methods | `hopper-medium-replay-v2`, seed0, CQL-family smoke | 不要先做新算法或多 seed sweep |
| LJY | B: normal / non-conservative contrast | PPO / SAC / vanilla TD3-style online fine-tuning | `hopper-medium-replay-v2`, seed0, PPO/SAC/vanilla online curve | 不要把 PPO 50k 低分直接解释成失败 |
| Tree / JBW | C: policy regularization | TD3+BC / ReBRAC / SSAR / ATLAS | 整理已完成结果，补必要图表和机制解释 | 不要继续把 C 线扩成独立论文主线 |

## Minimum Comparable Experiment

每个人先只交这个最小格式：

- method;
- method family: value conservatism / non-conservative contrast / policy regularization;
- env: `hopper-medium-replay-v2`;
- seed: 0;
- offline steps;
- online steps;
- eval episodes;
- final normalized score;
- best normalized score;
- log / curve path;
- one-sentence interpretation.

## Stop Rule

现在不要做：

- 6 environments x 3 seeds;
- 大规模调参；
- 每个人各自提出一个新主线；
- 没有统一 evaluator 的结果比较；
- 只发截图、不发 method/env/seed/steps/eval 信息。

只有当 `hopper-medium-replay-v2` seed0 的最小表能说明问题后，再决定是否补：

- seed1 / seed2;
- `walker2d-medium-replay-v2`;
- longer online fine-tuning;
- Gate-aware / ATLAS 机制扩展。

## Message To Send

我感觉我们现在的问题不是实验不够，而是主线和分工还没完全锁住。建议先把课程项目主线收回到原来的问题：低质量 offline replay 数据下，不同 conservatism / regularization 设计如何影响 offline-to-online fine-tuning。

先按这个分工推进会比较清楚：KX 这边做 A 线，主要看 CQL/Cal-QL 这类 value conservatism；LJY 这边做 B 线，主要看 PPO/SAC/vanilla TD3-style online fine-tuning 这类 normal / non-conservative contrast；我这边做 C 线，整理 TD3+BC/ReBRAC/SSAR/ATLAS 这类 policy regularization 和 trusted-action selection。

大家先不要做多 seed、多环境的大 sweep，也先不要各自扩成新题目。最小目标就是统一在 `hopper-medium-replay-v2`、seed0 下，把 method、steps、eval episodes、final/best normalized score、log/curve 路径和一句话解释交出来。等这个最小表能合起来，再决定要不要补 seed、walker2d、longer online 或 Gate-aware/ATLAS 机制扩展。
