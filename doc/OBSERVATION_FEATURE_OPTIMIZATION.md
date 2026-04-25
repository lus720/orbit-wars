# 优化建议

## 世界空间

- 星球
  - 静止星球
    - 笛卡尔坐标
    - 极坐标
    - 生产力
    - 兵力
    - 半径
    - 我方/敌方/中立
    - 是受到超级舰队攻击（能够击败目标星球+其最近同阵营星球）
    - Incoming舰队 v(i)
      - 兵力
      - 速度
      - 到达回合 i
      - 阵营
      - 拦截（舰队带有目标，主要判断是否会被新出的彗星拦截）

  - 旋转星球
    - 笛卡尔坐标
    - 极坐标
    - 生产力
    - 兵力
    - 半径
    - 角速度
    - 笛卡尔路径 vector path(500), path(i)=(x,y) （能否保存为一个环形状轨迹，更容易判断是否相交）
    - 是受到超级舰队攻击（能够击败目标星球+其最近同阵营星球）
  - 太阳（杀死所有穿过它的舰队）
    - 笛卡尔坐标
    - 极坐标
    - 半径
  - 彗星
    - 笛卡尔坐标
    - 极坐标
    - 生产力
    - 兵力
    - 半径
    - 笛卡尔路径 vector path(n), path(i)=(x,y), i表示当前回合

- 舰队
  - 我方/敌方
  - 兵力
  - 速度
  - 轨迹 vector path(n), path(i)=(x,y), i表示当前回合
  - 目标星球（第一个轨迹相交的星球）
  - 是否为超级舰队（能够击败目标星球+其最近同阵营星球）





## 1. 当前实现概览

当前 observation 编码入口在 `src/features.py`，主要流程是：

1. 从原始观测解析出 `GameState`
2. 以每颗己方星球 `src` 为一行决策样本
3. 为每行样本构造：
   - `self_features`
   - `candidate_features`
   - `global_features`
4. 将这些输入送入 `PlanetPolicy`，输出每个 candidate 的打分 `target_logits`

当前 feature 维度：

- `self_features = 11`
- `candidate_features = 14`
- `global_features = 8`

对应代码位置：

- `src/features.py`
- `src/policy.py`
- `src/game_types.py`

## observation

observation 需要补足未来演化相关的信息。

## 2.1 坐标表达约定

本文档后续默认采用混合坐标系，而不是纯笛卡尔坐标或纯极坐标。

- 星球和彗星的坐标和相对位置使用极坐标。旋转星球的位置预测
  - 以太阳中心为原点$(0,0)$
  - 坐标点 $(r,\theta)$
- 舰队路径计算，
  - $(x,y)$

###  `self_features`

#### 当前已有内容

- 全局极坐标位置
- 半径
- 当前船数
- 生产力
- 己方星球数
- 敌方星球数
- 己方总船数
- 敌方总船数

- 旋转速度，非旋转星球取 `0`
- `src_inbound_enemy_ships`
  含义：正在朝 `src` 方向推进的敌方舰船总数
- `src_inbound_friendly_ships`
  含义：正在朝 `src` 方向推进的己方舰船总数
- `src_has_overwhelming_enemy_fleet`
  含义：是否存在一支指向 `src` 的敌方进攻舰队，其舰船数大于等于 `src` 当前驻军与最近 `k` 个己方星球驻军之和；可用于标记“单次来袭就可能压穿本地与局部支援”的极大威胁
- `src_nearest_enemy_eta`
  含义：最近一支敌方舰队到达 `src` 的估计时间
- `src_nearest_friendly_eta`
  含义：最近一支己方增援舰队到达 `src` 的估计时间
- `src_frontline_distance`
  含义：`src` 到最近敌方星球的距离
- `src_support_distance`
  含义：`src` 到最近己方支援星球的距离
- `src_local_enemy_planets`
  含义：`src` 周围一定半径内敌方星球数量
- `src_local_friendly_planets`
  含义：`src` 周围一定半径内己方星球数量
- `src_local_enemy_ships`
  含义：`src` 周围一定半径内敌方总船数
- `src_local_friendly_ships`
  含义：`src` 周围一定半径内己方总船数
- `src_orbit_radius`
  含义：`src` 相对太阳中心的轨道半径
- `src_orbit_sin_theta`
- `src_orbit_cos_theta`
  含义：`src` 当前轨道角位置的正余弦表示
- `src_rotation_speed`
  含义：源星球 `src` 的旋转速度；如果不是旋转星球则取 `0`
- `src_comet_speed`
  含义：源星球 `src` 的彗星轨道速度；如果不是彗星则取 `0`
- `src_comet_radius`
  含义：若 `src_comet_speed != 0`，则显式记录其半径
- `src_comet_production`
  含义：若 `src_comet_speed != 0`，则显式记录其产量
- `src_comet_path_index`
  含义：若 `src_comet_speed != 0`，则记录其当前走到轨迹上的哪一段
- `src_comet_next_radius`
- `src_comet_next_sin_theta`
- `src_comet_next_cos_theta`
  含义：若 `src_comet_speed != 0`，则记录其轨迹上的下一位置或下一关键点，使用极坐标表达
- `src_comet_remaining_life`
  含义：若 `src_comet_speed != 0`，则记录它还会在几回合后离场；否则取 `0`
- `src_post_send_ratio`
  含义：如果对当前目标发兵，`src` 剩余驻军占原始驻军的比例

#### 这一组的价值

这组 feature 的目标是让模型知道：

- 哪些星球可以作为出兵源
- 哪些星球更像前线据点
- 哪些星球应该先保命而不是扩张

### 3.2 `candidate_features`

#### 当前已有内容

当前 `candidate_features` 已经包含一些有价值的信息：

- 中立 / 己方 / 敌方标记
- 目标全局极坐标位置
- 相对笛卡尔位移
- 距离
- 目标船数
- 目标产量
- 旋转速度，非旋转星球取 `0`
- 当前直线是否穿太阳
- `src` 当前船数

#### 当前不足

这一组是当前 observation 最明显的短板。

目前大部分信息仍然是“目标现在长什么样”，但 Orbit Wars 决策真正关心的是：

- 我飞过去要多久
- 我到的时候它还有多少船
- 它会不会被别人先拿下
- 我打过去之后是赚还是亏
- 它是不是彗星，是否即将离场

#### 推荐新增项

优先级最高：

- `eta`
  含义：从 `src` 到 `tgt` 的预计飞行时间
- `target_expected_ships_at_eta`
  含义：按产兵与已知入射舰队粗略估计，`eta` 时刻目标的驻军
- `target_expected_owner_at_eta`
  含义：`eta` 时刻目标更可能属于谁
- `target_inbound_enemy_ships`
  含义：预计在 `eta` 时间窗内，敌方打向 `tgt` 的舰船总数
- `target_inbound_friendly_ships`
  含义：预计在 `eta` 时间窗内，己方打向 `tgt` 的舰船总数
- `capture_margin`
  含义：若本回合从 `src` 出兵，预计最终净剩多少船
- `production_per_required_ship`
  含义：目标产量与所需投入船数的比值
- `production_per_eta`
  含义：目标产量与飞行时间的比值

建议作为第二批补充：

- `target_orbit_radius`
  含义：目标星球相对太阳中心的轨道半径
- `target_orbit_sin_theta`
- `target_orbit_cos_theta`
  含义：目标星球当前轨道角位置的正余弦表示
- `rel_dx`
- `rel_dy`
  含义：目标相对 `src` 的局部笛卡尔位移
- `rel_sin_bearing`
- `rel_cos_bearing`
  含义：`src -> target` 指向角的正余弦表示
- `target_rotation_speed`
  含义：目标星球的旋转速度；如果不是旋转星球则取 `0`
- `target_comet_speed`
  含义：目标星球的彗星轨道速度；如果不是彗星则取 `0`
- `target_comet_radius`
  含义：若 `target_comet_speed != 0`，则显式记录其半径
- `target_comet_production`
  含义：若 `target_comet_speed != 0`，则显式记录其产量
- `target_comet_path_index`
  含义：若 `target_comet_speed != 0`，则记录其当前轨迹进度
- `target_comet_next_radius`
- `target_comet_next_sin_theta`
- `target_comet_next_cos_theta`
  含义：若 `target_comet_speed != 0`，则记录其轨迹上的下一位置或下一关键点，使用极坐标表达
- `comet_remaining_life`
  含义：若 `target_comet_speed != 0`，距离离场还有多少回合；否则取 `0`
- `target_future_radius`
- `target_future_sin_theta`
- `target_future_cos_theta`
  含义：根据旋转规则或彗星路径预测的未来目标位置，使用极坐标表达
- `future_crosses_sun`
  含义：按未来位置计算，航线是否更容易穿太阳
- `target_local_enemy_support`
  含义：目标周边敌方支援强度
- `target_local_friendly_support`
  含义：目标周边己方支援强度
- `strategic_value`
  含义：目标的桥头堡价值、前线价值或高产价值综合分

#### 这一组的价值

这组的目标是把模型从：

- “看当前最近的目标”

推进到：

- “看未来到达结算，选择时机更好的目标”

如果只允许优先强化一组 feature，应该优先强化 `candidate_features`。

#### 关于“彗星轨迹”如何表达

彗星轨迹在 observation 中通常以 `paths + path_index` 提供。对于 feature 设计，不建议把整条轨迹直接完整展开成高维向量，而更建议拆成少量更稳定的标量：

- 当前彗星速度，非彗星取 `0`
- 当前轨迹进度 `path_index`
- 下一位置 `next_radius / next_sin_theta / next_cos_theta`
- 若干回合后的预测位置，优先使用极坐标表达
- 剩余寿命 `remaining_life`

这样既保留了轨迹信息，又不会让 feature 维度和构造成本暴涨。

#### 关于“未来外来舰队信息向量”如何表达

如果希望在 observation 中加入“这颗星球未来会受到哪些外来舰队影响”的信息，建议优先用固定长度向量来表达，而不是直接展开所有舰队对象。

推荐形式：

- `arrival_flow_t1 ... arrival_flow_tH`
  含义：未来第 `1..H` 回合内，预计到达该星球的外来舰队净流量
- 友军记正，敌军记负

如果需要更细粒度，也可以拆成两组：

- `friendly_arrival_t1 ... friendly_arrival_tH`
- `enemy_arrival_t1 ... enemy_arrival_tH`

但第一版更推荐只保留净流量向量，因为：

- 维度更小
- 更容易学习
- 更适合先验证是否有效

同时，这类信息不必每回合都从零重新计算。更合理的工程方式是：

- 当前先保留“所有外来舰队的单回合信息”
- 将其作为可缓存状态
- 后续每个回合只做增量更新

也就是说，这部分信息应被视为“可保留的中间状态”：

- 已存在舰队继续前进时，只更新它们的剩余 ETA、位置或目标关系
- 已结束的舰队从缓存中移除
- 新发射的舰队追加进去

这样可以避免每步都重新扫描和重建完整未来时间线。

### 3.3 `global_features`

#### 当前已有内容

当前 `global_features` 已包含：

- 回合进度
- 己方 / 敌方 / 中立星球数
- 己方 / 敌方总星球驻军
- 己方 / 敌方飞行中舰船总数

#### 当前不足

它可以描述大势，但缺少以下关键信息：

- 星球旋转速度
- 彗星状态
- 局部压力是否集中在某一路
- 双方总产能差
- 当前是否已经进入收官阶段

#### 推荐新增项

优先级较高：

- `has_comets`
  含义：当前场上是否存在活跃彗星
- `comet_count`
  含义：当前场上的彗星数量
- `comet_group_count`
  含义：当前活跃彗星组数量
- `my_comet_count`
  含义：己方控制中的彗星数量
- `enemy_comet_count`
  含义：敌方控制中的彗星数量
- `neutral_comet_count`
  含义：中立彗星数量
- `comet_ships`
  含义：彗星上的舰船数
- `comet_production`
  含义：彗星的产量
- `angular_velocity`
  含义：当前对局的统一旋转角速度
- `my_total_production`
  含义：己方总产能
- `enemy_total_production`
  含义：敌方总产能
- `production_diff`
  含义：双方总产能差
- `next_comet_spawn_eta`
  含义：距离下一次彗星刷新还有多少回合

建议作为第二批补充：

- `frontline_pressure`
  含义：综合局部入射舰队与敌近邻密度的压力指标
- `my_high_prod_planet_count`
  含义：己方控制高产星数量
- `enemy_high_prod_planet_count`
  含义：敌方控制高产星数量
- `estimated_score_diff`
  含义：若当前对局立刻结束，双方总分差估计
- `endgame_phase_flag`
  含义：是否进入偏收官的阶段

#### 这一组的价值

这组的目标是让模型知道：

- 当前更适合扩张、转守还是抢收官
- 大图资源控制上谁占优
- 彗星和旋转规则会如何影响整体节奏

## 4. 哪些 Observation 信息当前还没用上

从 Kaggle 提供的 observation 字段看，当前项目还没有利用以下信息：

- `angular_velocity`
- `initial_planets`
- `comets`
- `comet_planet_ids`

这几个字段都在 `doc/STRATEGY.md` 中被明确点名，其中：

- `angular_velocity` 和 `initial_planets` 可用于预测旋转星球未来位置
- `comets` 和 `comet_planet_ids` 可用于判断彗星类型、未来路径和离场时机

这说明当前 observation 仍然有一块明显的动态信息空白。

## 4.1 Reward 设计建议

推加入轻度 shaping reward。

```text
Phi(s) =
  w1 * (我的总飞船数 - 最强敌人的总飞船数)
  + w2 * (我的总产能 - 最强敌人的总产能)
```

其中：

- `w1`：总飞船数差的权重
- `w2`：总产能差的权重



```text
reward_t =
  alpha * (gamma * Phi(s_{t+1}) - Phi(s_t))
  + terminal_bonus
```

其中：

- `alpha`：shaping 奖励缩放系数
- `gamma`：PPO 训练时使用的折扣因子
- `terminal_bonus`：终局胜负奖励，建议继续保留

### 权重

w1 前期略小，后期略大
w2=1

### 额外注意事项

- 如果直接使用原始船数和原始产能，reward 量级可能偏大
- 实际实现时可以考虑先做归一化，再乘以 `w1 / w2`
- shaping reward 的累计量级不应明显压过终局胜负奖励
- 如果后面发现模型开始“保船不进攻”，说明 `w1` 可能过强，需要回调

## 4.2 Aiming Logic

```text
[from_planet_id, direction_angle, num_ships]
```

- `target selection` 负责决定打谁
- `aiming logic` 负责决定朝哪个方向打，也就是如何求 `direction_angle`
- `ship-count logic` 负责决定发多少船

限制

- 保留 `future_crosses_sun` 一类安全性检查
- 如果航线被太阳遮挡，当前 aiming logic 直接视为无效，`direction_angle = 0`
- 如果被彗星遮挡，标记为无效


- 对目标建立短期未来位置预测
- 结合预计飞行时间 `eta`，瞄准目标在 `eta` 时刻的位置
- 对旋转星球使用 `angular_velocity`
- 对彗星使用 `paths + path_index`

这一阶段 aiming logic 的核心公式可以理解为：

```text
aim_point = predicted_target_position(t + eta)
direction_angle = angle(src -> aim_point)
```

### 4.2.1 Ship-Count Logic

 `ship-count logic`
从 4 个离散选项中选择一个：

```text
target.ships + 1
总兵力的 50%
总兵力的 75%
总兵力的 100%
```

