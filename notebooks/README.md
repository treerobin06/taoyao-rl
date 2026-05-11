# notebooks/

分析画图用。所有 `.ipynb` 读 `results/*.jsonl` 聚合，**不直接重跑训练**。

推荐工具：
- `rliable`：IQM + bootstrap CI（不用 mean ± std）
- `seaborn`：曲线 / boxplot
- `matplotlib`：自定义

模板见 `analysis_template.ipynb`（待加）。

## 出图规范

- 横轴 = step（offline / online_finetune 分两段）
- 纵轴 = normalized_score（0-100）
- 阴影 = IQM 95% CI（rliable.plot_sample_efficiency_curve）
- 各算法不同颜色，同 family 用深浅
- 字号 ≥ 12，标题字号 ≥ 14
- 图导出 `results/figures/<algo>_<env>.pdf` 或 .png

## 不要做

- ❌ 在 notebook 里跑训练
- ❌ 在 notebook 里硬编码本地路径
- ❌ commit 带输出的 notebook（先 `jupyter nbconvert --clear-output`）
