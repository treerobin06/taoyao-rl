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

## 同步和可视化

默认脚本会同时写：

- `results/*.jsonl`：最终统计和复现备份
- `.aim/`：本地 Aim 曲线面板

打开 Aim：

```bash
bash scripts/aim_ui.sh
```

如果要云端同步，再启用 W&B：

```bash
USE_WANDB=1 bash scripts/run_td3_bc_pilot.sh
```

最终统计可以用 wandb export + rliable，也可以直接读 `results/*.jsonl` 做交叉验证。

本仓库提供一个轻量导出脚本：

```bash
python scripts/export_wandb.py --entity <your-team-or-username> --project taoyao-rl
```

默认导出到 `results/wandb_export/`，包括：

- `runs_summary.csv`
- 每个 run 的 `*_config.json`
- 每个 run 的 `*_summary.json`
- 每个 run 的 `*_history.jsonl` / `*_history.csv`
