"""Unified evaluation + result JSON schema — 全组共用。

JSON schema (one line per eval, append-only)：
{
  "algo": "td3_bc",
  "env": "hopper-medium-v2",
  "seed": 0,
  "step": 1000000,
  "phase": "offline",            # offline | online_finetune
  "raw_return": 2840.5,
  "normalized_score": 78.3,      # 0-100，按 d4rl.get_normalized_score
  "episode_length": 999.5,
  "wall_time": 1715432100        # unix timestamp
}

每条实验输出到 results/<algo>_<env>_seed<seed>.jsonl，append 模式。
分析阶段读所有 .jsonl 聚合即可。
"""
import json
import os
import time
import numpy as np


def eval_episodes(agent, env, n_episodes: int = 10) -> dict:
    """Run agent in env for N episodes, return mean metrics.

    Agent 必须实现 agent.act(obs: np.ndarray) -> np.ndarray.
    env 必须实现 d4rl 的 get_normalized_score。
    """
    returns, lengths = [], []
    for _ in range(n_episodes):
        obs = env.reset()
        ep_return, ep_len = 0.0, 0
        done = False
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            ep_return += float(reward)
            ep_len += 1
        returns.append(ep_return)
        lengths.append(ep_len)

    raw_return = float(np.mean(returns))
    normalized = float(env.get_normalized_score(raw_return) * 100.0)
    return {
        "raw_return": raw_return,
        "normalized_score": normalized,
        "episode_length": float(np.mean(lengths)),
        "return_std": float(np.std(returns)),
    }


def write_result(result_dir: str, algo: str, env_name: str, seed: int,
                 step: int, phase: str, eval_metrics: dict) -> dict:
    """Append one eval record to results/<algo>_<env>_seed<seed>.jsonl."""
    os.makedirs(result_dir, exist_ok=True)
    fname = f"{algo}_{env_name}_seed{seed}.jsonl"
    path = os.path.join(result_dir, fname)

    record = {
        "algo": algo,
        "env": env_name,
        "seed": seed,
        "step": int(step),
        "phase": phase,
        "wall_time": int(time.time()),
        **eval_metrics,
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def load_results(result_dir: str) -> list:
    """读 results/ 下所有 .jsonl，合并成 list[dict]。"""
    out = []
    if not os.path.isdir(result_dir):
        return out
    for fname in sorted(os.listdir(result_dir)):
        if not fname.endswith(".jsonl"):
            continue
        with open(os.path.join(result_dir, fname)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out
