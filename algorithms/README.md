# Algorithms

本目录放各算法实现。**只放修改过的版本**，未改的算法直接从官方仓库 clone 引用，避免重复维护。

## 已就位

| 文件 | 算法 | 用途 |
|---|---|---|
| `bc.py` | Behavior Cloning | smoke test 用，验证 pipeline |

## 待引入（按小组分工）

### Policy Regularization Family（C 线）

| 算法 | 原仓库 | License | 引入方式 |
|---|---|---|---|
| **TD3+BC** | tinkoff-ai/CORL `algorithms/offline/td3_bc.py` | Apache-2.0 | 拷贝 + 加 `common.eval` 替换原 eval |
| **ReBRAC** | tinkoff-ai/CORL `algorithms/offline/rebrac.py` | Apache-2.0 | 同上 |
| **PRDC** | LAMDA-RL/PRDC | 无 license | 改写成单文件 CORL-style |
| **A2PR** | ltlhuuu/A2PR | MIT | 改写成单文件 CORL-style |

### Doubly Mild Generalization 方向

| 算法 | 原仓库 | License | 引入方式 |
|---|---|---|---|
| **DMG** | maoyixiu/DMG | check repo | 改写时用 `common.data` 和 `common.eval` |

### Strategically Conservative Value 方向

| 算法 | 原仓库 | License | 引入方式 |
|---|---|---|---|
| **SCQ** | purewater0901/SCQ | check repo | 改写时用 `common.data` 和 `common.eval` |

## 引入新算法的步骤

1. `git clone <原仓库> /tmp/<algo>` 看下原实现
2. 在本目录新建 `<algo>.py`（单文件 CORL-style）
3. **必须**用 `common.data.D4RLDataset` 而不是自己写 loader
4. **必须**用 `common.eval.eval_episodes` 和 `common.eval.write_result`，保证 JSON schema 一致
5. **必须**调用 `common.seed.set_seed(seed)`
6. CLI 接口对齐 `bc.py`：`--env --seed --steps --batch_size --eval_freq --result_dir --wandb`
7. 跑 100k step 在 `hopper-medium-v2` 上看分数是否进入 `configs/shared.yaml > expected_scores` 的 ±15% 区间
8. 通过 → 提交；不过 → 调试，不要先提交跑歪的实现

## 不在本目录的

- **在线 RL 基线（PPO / SAC / TD3）**：直接用 stable-baselines3 / CleanRL，不自己写
- **CQL** / **IQL**：组内有人做，跨线协调，不在 C 线 repo 中实现

## 反模式（不要这样做）

- ❌ 在算法实现里 import 自己写的 dataset class
- ❌ 在算法实现里自己实现 eval（一定走 `common.eval`）
- ❌ 在算法实现里硬编码 wandb entity 或 result_dir
- ❌ 拷 CORL 整个 framework（只拷单个算法文件）
- ❌ 给每个算法搞独立 conda env（全组共用一个 venv）
