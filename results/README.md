# results/

所有实验结果 JSON 写到这里（gitignored，不进 commit）。

## 命名规范

```
<algo>_<env>_seed<seed>.jsonl
```

例：
- `bc_hopper-medium-v2_seed0.jsonl`
- `td3_bc_halfcheetah-medium-replay-v2_seed2.jsonl`
- `dmg_hopper-medium-v2_seed1.jsonl`

## JSON Schema（每行一条 eval record）

```json
{
  "algo": "td3_bc",
  "env": "hopper-medium-v2",
  "seed": 0,
  "step": 1000000,
  "phase": "offline",
  "wall_time": 1715432100,
  "raw_return": 2840.5,
  "normalized_score": 78.3,
  "episode_length": 999.5,
  "return_std": 65.2
}
```

`phase` ∈ {`offline`, `online_finetune`}。

## 聚合分析

```python
from common import load_results
records = load_results("results/")
# → list[dict]，喂给 pandas / rliable 做 IQM
```

## 同步

各人本地跑完后**用 wandb 同步**（不要把 jsonl push 到 git）。
最终统计用 wandb export + rliable 画图，或者 `scp` 互相拉本地 jsonl 做交叉验证。
