# 干净版 Agent 训练计划

这份计划描述一个基于大量回放训练的新 Orbit Wars agent。它是
clean-room 方案：不复用 `submission.py` 里的规划器、profile、任务打分、
开局规则或长期积累下来的常量。

## 目标

构建一个能泛化到以下场景的学习型 agent：

- 成千上万局公开回放；
- 2 人局和 4 人局；
- 每局都不同的地图；
- 可变数量的星球、舰队和彗星。

规则只允许用于表达环境机制和动作合法性。战略偏好应该从数据中学习，而
不是写成一大张阈值表。

## 非目标

- 不移植旧 `submission.py` 的 planner。
- 不添加按地图、开局、产能或玩家数量特化的启发式分支。
- 不硬编码星球 ID、队名、seed 行为或特定回放模式。
- 当目标星球和确定性物理已经能算出角度时，不让模型直接自由输出连续角度。

## 设计原则

使用一个很小的干净执行层，加上一个学习型策略：

```text
observation
-> entity encoder
-> learned action policy
-> clean execution layer
-> Kaggle action list
```

执行层只回答机械问题：

- 哪些源星球合法？
- 有多少船可用？
- 选中的目标预计在哪里？
- 用什么角度可以命中该目标？
- 路径是否穿过太阳？
- 最终动作格式是否合法？

除此之外的判断都交给模型。

## 规则预算

运行时规则应该少到能一次审完。

允许的规则：

1. 解析 planets、fleets、comets、player id、step 和配置。
2. 枚举合法源星球：当前玩家拥有、且至少有 1 艘船的星球。
3. 枚举目标实体：除了源星球以外的可见星球。
4. 将 `source + target + send_fraction` 转换为 `[source_id, angle, ships]`。
5. 拒绝违反环境机制的动作，例如发送 0 艘船、发送超过可用船数、或路径穿过太阳半径。
6. 使用环境形状决定解码上限，例如第一版最多从每个己方源星球发出一个主要动作。

避免写这样的战略阈值：

```text
if production >= 4 and ships < 23 and step < 80:
```

这些条件应该作为模型特征出现，而不是手写决策。

## Agent 架构

使用实体模型，而不是固定尺寸的地图特征。

输入：

- 星球 token：相对当前玩家的 owner、坐标、半径、船数、产能、是否彗星、已知时的彗星剩余寿命；
- 舰队 token：相对 owner、坐标、角度、源星球 ID embedding、船数、可恢复时的估计目标；
- 全局 token：step、剩余回合数、玩家数量、angular_velocity、归一化总分、归一化总产能；
- 候选动作上下文：选中的源星球 token、目标星球 token，以及该源星球当前剩余船数。

推荐模型：

```text
planet/fleet/global tokens
-> shared Entity Transformer or GNN encoder
-> 2-player policy head
-> 4-player policy head
-> optional value head
```

共享 encoder，因为物理机制和局部战术在 2P/4P 之间可以迁移。策略 head 分开，
因为 2P 和 4P 的战略激励差异足够大，单个无条件 head 容易把两种玩法混在一起。

## 动作空间

将动作表示成一段短的自回归序列：

```text
repeat:
  choose source planet or STOP
  choose target planet
  predict send_fraction in [0, 1]
  execution layer converts to ships and angle
```

第一版可以限制为每个己方源星球最多发出一个动作。这会让推理更确定，也能避免
重复抽干同一源星球；它是一个由环境结构导出的限制，而不是战略阈值。

不要把原始角度作为主要训练目标。对于回放中的专家动作，应该通过模拟舰队路径，
找出它最可能撞上的第一个星球，以此推断目标。训练策略学习
`source -> target -> send_fraction`，角度由干净执行层计算。

## 干净执行层

执行层应该是一个新模块，不从旧 bot 导入任何内容。

职责：

- 按环境公式计算舰队速度；
- 根据 `initial_planets` 和 `angular_velocity` 预测普通星球运动；
- 直接从 observation 读取彗星路径；
- 用少量确定性 fixed-point 迭代求解目标拦截；
- 使用真实太阳半径拒绝精确穿太阳路径，不引入调过参的安全 margin；
- 将发船数 clamp 到 `1..available`。

如果某条规则需要可调 margin 才表现好，优先把这个量做成模型输出或模型特征，
而不是再加一个常量。

## 回放数据集

创建回放抽取流水线，每个 agent-turn 写出一个训练样本。

推荐输出形状：

```json
{
  "episode_id": 75758322,
  "step": 42,
  "player": 3,
  "player_count": 4,
  "rank_or_result": 1,
  "observation": "...compact encoded state...",
  "expert_actions": [
    {"source": 12, "target": 5, "send_fraction": 0.63}
  ]
}
```

抽取细节：

- 按 episode 切分 train/validation/test，不要按 turn 切分；
- 保留 2P 和 4P 标签；
- 尽量包含多个顶尖队伍，不只使用第一名；
- 对完全相同的已下载回放去重；
- 如果某个专家动作无法映射到合理目标，可以丢弃该 turn，但必须记录丢弃率；
- 保留无动作 turn，因为等待和时机本身也是策略的一部分。

## 地图泛化

每张地图都不同，因此模型必须学习关系结构。

使用：

- 相对 owner 编码：self、neutral、enemy；如果未来环境存在队友，再加 teammate；
- 坐标按棋盘大小归一化；
- 可选的当前玩家视角变换，例如将对称地图反射或旋转到规范 home 象限；
- 对星球和舰队使用 permutation-invariant encoder；
- 当变换后的动作仍合法时，使用棋盘对称性做数据增强。

不要把固定 planet id slot 当作语义特征。

## 训练阶段

### 阶段 1：监督行为克隆

训练策略模仿回放动作：

- source 或 STOP 分类；
- 在可见星球上的 target 分类；
- send fraction 回归；
- 可选的 no-action 校准 loss。

这个阶段产出第一个可运行的干净 agent。

### 阶段 2：候选动作 Reranking 变体

如果直接自回归解码不稳定，切换到学习型打分器：

```text
score(state, source, target)
send_fraction(state, source, target)
```

推理时给所有合法 `source -> target` 对打分，然后发出最好的非冲突动作。这个
方案仍然不使用旧 planner，并且保持规则最小化。

### 阶段 3：Value Head

增加一个用回放结果训练的 value head：

- 2P：胜率和最终分差；
- 4P：最终排名期望和归一化分数占比。

先把 value head 用作诊断。只有当它能提升本地锦标赛结果后，再让它影响动作选择。

### 阶段 4：Self-Play 微调

监督学习稳定后，用 self-play 继续微调：

- 从行为克隆模型初始化；
- 对战冻结的历史版本、baseline 和自己；
- 保持同一个干净动作空间；
- 优先使用终局排名/胜负奖励；
- 只有在稀疏奖励训练失败时才添加 dense reward，并且数量要少、文档要清楚。

Self-play 可以突破回放分布，但不应该作为第一个里程碑。

## 评估

同时评估模仿质量和实战强度。

离线指标：

- source accuracy；
- target accuracy；
- send fraction error；
- valid action rate；
- no-action precision 和 recall；
- 2P 和 4P 分开统计。

实战指标：

- 本地 2P 对固定 baseline 的胜率；
- 本地 4P 对固定 baseline 的排名分布；
- score share；
- elimination rate；
- action invalid/rejected count；
- 每回合平均推理时间。

始终单独报告 2P 和 4P。一个改动如果帮助其中一种模式、伤害另一种模式，
不应该被合并平均数掩盖。

## 打包

比赛入口应该由干净 agent 包生成：

```text
clean_agent/
  features.py
  physics.py
  policy.py
  runtime.py
tools/
  extract_replay_dataset.py
  train_clean_policy.py
  export_clean_submission.py
```

如果 Kaggle 提交必须是单文件，`export_clean_submission.py` 应该生成自包含的
`submission.py`，其中包括：

- 干净 runtime 代码；
- 紧凑模型权重，必要时可以量化；
- 不依赖旧 planner。

## 第一个里程碑

交付一个最小可运行模型：

1. 从 leaderboard 顶尖提交下载一个均衡回放集。
2. 抽取 2P 和 4P 的 agent-turn 样本。
3. 实现干净的 `physics.py` 和动作转换。
4. 训练阶段 1 的行为克隆模型。
5. 导出自包含的干净 submission。
6. 在本地 2P 和 4P 锦标赛中对当前 baseline 评测。

如果 agent 能产生合法动作、在时间限制内跑完整局，并击败 no-op 或朴素扩张
baseline，就算第一个里程碑成功。之后再提升数据量、模型规模和 self-play。
