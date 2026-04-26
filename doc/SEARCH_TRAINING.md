# Neural-guided Beam Search for Orbit Wars

## 方案思路

Orbit Wars 是**完美信息、确定性模拟、500回合**的游戏。纯 PPO 靠随机探索+稀疏奖励学策略，本质是让神经网络"盲猜未来"。实践表明 1000 轮 PPO 训练仍打不过简单的 nearest_planet_sniper 规则 agent。

核心思路：**神经网络给搜索方向，搜索补足战术精度，迭代训练让搜索越来越强。**

### 架构总览

```
每一步决策流程：

当前局面 observation
     │
     ▼
特征编码 features.encode_turn()
     │
     ▼
策略网络 PlanetPolicy → 每个源星球输出 top-K 候选 (目标, 船数)
     │
     ▼
对每个候选:
 ├─ WorldModel 计算 ETA / 速度 / 方向 / 预估战斗结果
 ├─ 模拟未来 H 步 (轻量 OrbitSimulator)
 └─ 价值网络评估最终状态
     │
     ▼
选取评分最高的动作 → 执行 → 进入下一回合
```

```
训练流程（Expert Iteration）:

搜索增强策略 → 自我对局 → 收集 (state, action, value) 数据
     │
     ▼
监督学习训练策略网络模仿搜索决策（Behavior Cloning）
   + 训练价值网络预测胜负（MSE）
     │
     ▼
网络变强 → 搜索更强 → 数据更好 → 网络更强
```

### 核心组件

| 模块 | 文件 | 作用 |
|---|---|---|
| 策略网络 | `src/policy.py` | 从局面特征生成候选动作 + 评估局面价值 |
| 搜索引擎 | `src/search_agent.py` | 对每个候选动作做前向推演，选最优 |
| 轻量模拟器 | `src/simulator.py` | 纯 numpy 游戏模拟，用于搜索中的未来推演 |
| 世界模型 | `src/world_model.py` | 分析型预测（ETA、战斗结果、舰队速度） |
| 训练循环 | `src/train_search.py` | Expert Iteration 主循环 |
| 特征编码 | `src/features.py` | 局面→特征向量 |

### 动作评估公式

对每个候选动作 `(src → tgt, N_ships)` 评分：

```
score = analytical_score + heuristic_weight × value_network_score

analytical_score =
  2.0 × ships_net / max_ships      # 舰船效率
  + 3.0 × prod_efficiency           # 产能收益
  - 0.5 × dist_ratio                # 距离惩罚
  + 0.5 reinforcement_bonus         # 支援友方奖励
  + 0.3 capture_own_bonus           # 留守奖励
```

### 损失函数

训练时同时优化两个目标：

```
loss = CrossEntropy(policy_output, search_selected_action)
     + 0.5 × MSE(value_output, game_outcome)
```

- Policy loss：监督模仿搜索的决策（Behavior Cloning）
- Value loss：预测最终胜负（+1 赢 / -1 输 / 0 平）


## 如何训练

### 安装依赖

```bash
pip install numpy pyyaml torch kaggle-environments
```

### 运行训练

```bash
python -m src.train_search --config default_cfg.yaml
```

训练参数全部在 YAML 中配置（`train_search` 段落），无需命令行参数。

### 配置文件说明

`default_cfg.yaml` 包含完整的训练配置：

```yaml
# ----- 训练参数 (expert iteration with search) -----
train_search:
  iterations: 20                # 专家迭代总次数
  games_per_iter: 16            # 每次迭代收集数据的对局数
  epochs: 5                     # 每次迭代中监督训练轮数
  batch_size: 128               # 训练 batch size
  replay_buffer_size: 10000     # 经验回放缓冲区上限（FIFO）
  lr: 0.0001                    # 学习率
  checkpoint_every: 5           # 每N次迭代保存一次模型

# ----- 搜索参数 (SearchAgent 决策时使用) -----
search:
  top_k_targets: 3              # 每源星球选取 top-K 候选目标
  ship_options: 2               # 每候选目标考虑几个舰船数量选项
  simulation_horizon: 15        # 动作评估时的未来推演步数
  heuristic_weight: 0.5         # 启发式评分相对价值网络评分的权重
```

**注意：** `main()` 会自动将 `simulation_horizon` 设为 3 加速数据收集。比赛提交时可设回 10-15。

### 训练流程

每一轮迭代：

1. **收集数据**：用当前策略+搜索打 n 局，记录每步的 (state, action)
2. **训练**：在 replay buffer 上做监督学习，模仿搜索决策 + 预测胜负
3. **评估**：与 baseline (nearest_planet_sniper) 对战，看胜率
4. **保存**：每 checkpoint_every 轮保存 checkpoint

训练过程输出示例：

```
========== Iteration 5/20 ==========
  [Game 1/16] 对手=self 开始
    step    #1  search=  432ms  sources= 8  actions= 5  game=3/16
    step    #2  search=  387ms  sources= 7  actions= 4  game=3/16
    ...
  >>> 数据收集完成
      对局: 16 局
      总步数: 225
      样本: 1520 (缓冲区共 8450)
      胜率: 62.5%  负率: 37.5%
      搜索平均耗时: 452ms/步
  >>> 训练完成
      loss=0.4231  policy=0.3812  value=0.0419
      缓冲区样本: 8450  batches: 66  epochs: 5
  >>> 评估 vs baseline
      胜: 60%  平: 10%  负: 30%
  >>> 模型已保存 (iteration 5)
```

- **sources**：当前步己方可以发兵的星球数量
- **actions**：搜索最终决定执行的动作数（<= sources，部分星球可能不出兵）
- **search**：搜索单步耗时
- **loss / policy / value**：监督训练的损失值


## 如何与 Baseline 测试

### 方式一：训练循环自动评估

训练时每轮会自动跑 10 局 vs baseline，直接看输出中的 `评估 vs baseline`。

### 方式二：使用独立对战脚本

```bash
python eval_vs_sniper.py --config default_cfg.yaml --checkpoint artifacts/orbit_wars_ppo/search_iter_0020.pt --games 20
```

### 方式三：手动回放

```bash
python play_vs_sniper.py --config default_cfg.yaml --checkpoint artifacts/orbit_wars_ppo/search_iter_0020.pt
```

会渲染 HTML 回放，可以直观观察决策质量。

### 从已有 checkpoint 继续训练

```bash
python -m src.train_search --config default_cfg.yaml --checkpoint artifacts/orbit_wars_ppo/search_last.pt
```


## 训练与推理差异

| | 训练时 | 比赛提交时 |
|---|---|---|
| 使用搜索 | ✓ 搜索产生训练数据 | ✓ 搜索做决策 |
| 速度要求 | 宽松（离线） | 每步 < 1 秒 |
| 搜索深度 | 3 步（数据收集） | 10-15 步 |
| 并行环境 | 串行 | 单线程 |
| 模型用途 | 被搜索训练 | 给搜索提供候选+评估 |

比赛提交时，搜索跑多深取决于时间预算。Kaggle 给每步 1 秒，15 步的前向搜索是可行的。
