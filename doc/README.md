# Orbit Wars Agent

搜索增强策略（Expert Iteration with Beam Search）。

## 快速开始

```bash
conda activate kaggle
pip install numpy pyyaml torch kaggle-environments
```

### 训练

```bash
python -m src.train_search --config default_cfg.yaml
```

训练参数在 `default_cfg.yaml` 的 `train_search` 段落中配置。

### 评估

```bash
python eval_vs_sniper.py --config default_cfg.yaml --checkpoint artifacts/orbit_wars_ppo/ckpt_001000.pt --games 10
```

### 回放

```bash
python play_vs_sniper.py --config default_cfg.yaml --checkpoint artifacts/orbit_wars_ppo/ckpt_001000.pt --output replay.html
```

## 项目结构

| 模块 | 文件 | 作用 |
|---|---|---|
| 策略网络 | `src/policy.py` | 候选动作生成 + 局面价值评估 |
| 搜索引擎 | `src/search_agent.py` | 候选推演 + 最优动作选择 |
| 轻量模拟器 | `src/simulator.py` | 纯 numpy 游戏模拟 |
| 世界模型 | `src/world_model.py` | 分析型预测（ETA、战斗结果） |
| 训练循环 | `src/train_search.py` | Expert Iteration |
| 特征编码 | `src/features.py` | 局面→特征向量 |
| 配置文件 | `default_cfg.yaml` | 全部训练/搜索参数 |

详细方案见 [doc/SEARCH_TRAINING.md](doc/SEARCH_TRAINING.md)。
