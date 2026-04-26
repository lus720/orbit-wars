# submission.py 函数说明文档

本文档按代码结构整理 `submission.py` 中的类型、顶层函数、`WorldModel` 方法和 `plan_moves()` 内部辅助函数。整体上，这个 bot 是一个启发式规划器：先建立世界模型，再预测舰队到达与星球未来状态，最后生成、打分、筛选并提交发兵动作。

## 数据类型

### `Planet`

星球元组：

```python
Planet(id, owner, x, y, radius, ships, production)
```

- `id`: 星球 ID。
- `owner`: 所属玩家，`-1` 表示中立。
- `x`, `y`: 当前坐标。
- `radius`: 星球半径。
- `ships`: 当前驻军。
- `production`: 每回合产能。

### `Fleet`

舰队元组：

```python
Fleet(id, owner, x, y, angle, from_planet_id, ships)
```

- `id`: 舰队 ID。
- `owner`: 所属玩家。
- `x`, `y`: 当前坐标。
- `angle`: 飞行方向，单位弧度。
- `from_planet_id`: 来源星球 ID。
- `ships`: 舰队船数。

### `ShotOption`

一次候选发射方案。

- `score`: 该方案分数。
- `src_id`: 来源星球 ID。
- `target_id`: 目标星球 ID。
- `angle`: 发射角度。
- `turns`: 预计到达回合数。
- `needed`: 最少需要发送的船数。
- `send_cap`: 推荐或允许发送的船数上限。
- `mission`: 发射所属任务类型，如 `capture`、`snipe`、`rescue`。
- `anchor_turn`: 可选时间锚点，用于截胡、救援、碰撞利用等需要卡时间的任务。

### `Mission`

战略任务。

- `kind`: 任务类型，如 `single`、`swarm`、`snipe`、`rescue`、`recapture`、`reinforce`、`crash_exploit`。
- `score`: 任务总分。
- `target_id`: 目标星球 ID。
- `turns`: 任务关键回合。
- `options`: 执行该任务的一组 `ShotOption`。单源任务通常只有一个，多源合击会有多个。

## 物理与几何函数

### `dist(ax, ay, bx, by)`

返回两点之间的欧氏距离。

### `orbital_radius(planet)`

返回星球中心到地图中心太阳点 `(50, 50)` 的距离。

### `is_static_planet(planet)`

判断星球是否为静态星球。规则是：

```python
orbital_radius(planet) + planet.radius >= ROTATION_LIMIT
```

满足该条件的外圈星球不绕太阳旋转。

### `fleet_speed(ships)`

根据舰队船数估算速度。船越多速度越快，最高接近 `MAX_SPEED`。

### `point_to_segment_distance(px, py, x1, y1, x2, y2)`

计算点 `(px, py)` 到线段 `(x1, y1) -> (x2, y2)` 的最短距离。用于判断舰队路径是否靠近太阳或星球。

### `segment_hits_sun(x1, y1, x2, y2, safety=SUN_SAFETY)`

判断路径线段是否进入 bot 认为的太阳危险区。它使用：

```python
distance_to_sun < SUN_R + safety
```

注意环境真实销毁阈值是 `< SUN_R`，而这里额外加了 `SUN_SAFETY`，属于保守过滤。

### `launch_point(sx, sy, sr, angle)`

计算舰队刚生成时的位置：

```python
source_center + (source_radius + LAUNCH_CLEARANCE) * direction
```

也就是从源星球边缘外一点开始，而不是从星球中心开始。

### `actual_path_geometry(sx, sy, sr, tx, ty, tr)`

计算从源星球打到目标星球的直线路径几何信息。

返回：

```python
(angle, start_x, start_y, end_x, end_y, hit_distance)
```

- `angle`: 从源中心指向目标中心的发射角。
- `start_x`, `start_y`: 舰队生成点。
- `end_x`, `end_y`: 预计碰到目标边缘的位置。
- `hit_distance`: 从生成点飞到目标边缘的距离。

### `safe_angle_and_distance(sx, sy, sr, tx, ty, tr)`

在 `actual_path_geometry()` 基础上增加太阳安全检查。若路径进入太阳危险区，返回 `None`；否则返回：

```python
(angle, hit_distance)
```

### `estimate_arrival(sx, sy, sr, tx, ty, tr, ships)`

估算一支 `ships` 数量的舰队从源星球到目标星球的发射角和到达时间。

返回：

```python
(angle, turns)
```

如果路线不安全，返回 `None`。

### `travel_time(sx, sy, sr, tx, ty, tr, ships)`

只返回预计飞行回合数。若路线不可行，返回一个极大值 `10**9`。

## 目标位置预测

### `predict_planet_position(planet, initial_by_id, angular_velocity, turns)`

预测普通星球 `turns` 回合后的坐标。

- 静态星球返回当前坐标。
- 旋转星球按当前角度和 `angular_velocity * turns` 预测未来坐标。

返回：

```python
(future_x, future_y)
```

### `predict_comet_position(planet_id, comets, turns)`

根据 `obs["comets"]` 中的路径表，预测彗星星球 `turns` 回合后的坐标。

返回：

```python
(future_x, future_y)
```

如果找不到彗星或未来下标越界，返回 `None`。

### `comet_remaining_life(planet_id, comets)`

返回某个彗星星球剩余还能存在多少个路径点。找不到时返回 `0`。

### `predict_target_position(target, turns, initial_by_id, ang_vel, comets, comet_ids)`

统一预测目标未来位置。

- 如果目标是彗星，调用 `predict_comet_position()`。
- 否则调用 `predict_planet_position()`。

### `target_can_move(target, initial_by_id, comet_ids)`

判断目标是否可能移动。彗星一定视为可移动；普通星球根据初始轨道半径判断是否旋转。

### `search_safe_intercept(src, target, ships, initial_by_id, ang_vel, comets, comet_ids)`

当直接瞄准当前目标位置不安全时，搜索未来若干回合是否存在可安全直线命中的拦截窗口。

返回：

```python
(angle, turns, target_future_x, target_future_y)
```

找不到则返回 `None`。

### `aim_with_prediction(src, target, ships, initial_by_id, ang_vel, comets, comet_ids)`

预测性瞄准函数。它会：

1. 先尝试打目标当前位置。
2. 如果目标会移动，则迭代预测目标到达时的位置。
3. 如果当前路径不安全，则调用 `search_safe_intercept()` 寻找未来安全窗口。

返回：

```python
(angle, turns, target_future_x, target_future_y)
```

失败则返回 `None`。

## 到达账本与战斗模拟

### `fleet_target_planet(fleet, planets)`

根据舰队当前位置、角度、速度，用射线与星球圆形碰撞的方式估算该舰队最可能撞到哪个星球，以及几回合后到达。

返回：

```python
(target_planet, eta)
```

找不到目标则返回 `(None, None)`。

### `build_arrival_ledger(fleets, planets)`

建立每个星球的未来舰队到达账本。

返回格式：

```python
{
    planet_id: [(eta, owner, ships), ...],
    ...
}
```

### `resolve_arrival_event(owner, garrison, arrivals)`

模拟同一回合到达同一星球的舰队战斗。

处理逻辑：

1. 按 owner 汇总到达舰队。
2. 最大攻击方和第二大攻击方先互相抵消。
3. 胜出的攻击方再和星球驻军结算。
4. 返回战斗后的 `(owner, garrison)`。

### `normalize_arrivals(arrivals, horizon)`

整理到达事件：

- 船数小于等于 `0` 的事件会被丢弃。
- 到达时间向上取整为整数回合，且至少为 `1`。
- 超过 `horizon` 的事件会被丢弃。
- 输出按到达回合排序。

### `simulate_planet_timeline(planet, arrivals, player, horizon)`

单星球未来模拟器。它逐回合模拟：

- 星球生产。
- 舰队到达。
- 战斗结算。
- 星球归属和驻军变化。

返回字典包含：

- `owner_at`: 每回合 owner。
- `ships_at`: 每回合驻军。
- `keep_needed`: 如果当前是我方星球，至少需要保留多少船才能在 horizon 内守住。
- `min_owned`: 我方持有期间最低驻军。
- `first_enemy`: 第一次敌方舰队到达的回合。
- `fall_turn`: 我方星球首次失守回合。
- `holds_full`: 当前全部兵力是否能守住。
- `horizon`: 模拟长度。

### `state_at_timeline(timeline, arrival_turn)`

从 `simulate_planet_timeline()` 的结果中读取某一回合的 `(owner, ships)`。

### `count_players(planets, fleets)`

根据当前星球和舰队中的 owner 推断局内玩家数量，至少返回 `2`。

### `nearest_distance_to_set(px, py, planets)`

返回点 `(px, py)` 到一组星球的最近距离。若列表为空，返回 `10**9`。

### `indirect_features(planet, planets, player)`

计算目标附近的间接战略价值特征，返回：

```python
(friendly, neutral, enemy)
```

它会按附近星球产能和距离加权，分别统计友方、中立、敌方影响。

## `WorldModel`

`WorldModel` 是每回合构造的局面缓存。它把原始 `obs` 转成方便查询和模拟的结构，并缓存路线、时间线、兵力需求等结果。

### `WorldModel.__init__(...)`

初始化世界模型。主要工作：

- 分类我方、敌方、中立、静态中立星球。
- 统计当前兵力、产能、玩家数量、游戏阶段。
- 生成 `arrivals_by_planet`。
- 为每个星球构建基础未来时间线。
- 缓存防守需求、失守时间、间接价值等。

### `WorldModel.is_static(planet_id)`

判断某个星球是否静态。

### `WorldModel.comet_life(planet_id)`

返回彗星剩余寿命。

### `WorldModel.source_inventory_left(source_id, spent_total)`

计算某个我方星球扣除本回合已计划发兵后，还剩多少船可用。

### `WorldModel.plan_shot(src_id, target_id, ships)`

对一次发射进行预测性瞄准，内部调用 `aim_with_prediction()`，并用 `shot_cache` 缓存结果。

返回：

```python
(angle, turns, future_x, future_y)
```

或 `None`。

### `WorldModel.probe_ship_candidates(src_id, target_id, source_cap, hints=())`

生成一组要测试的候选发兵数量。它会结合：

- 小船数。
- 源星球可用兵力比例。
- 目标当前驻军附近的数量。
- 外部传入的 `hints`。

### `WorldModel.best_probe_aim(...)`

在一组候选发兵数量中，寻找满足时间限制的最优瞄准方案。

可选限制包括：

- `min_turn`
- `max_turn`
- `anchor_turn`
- `max_anchor_diff`

返回：

```python
(ships, (angle, turns, future_x, future_y))
```

### `WorldModel.reaction_times(target_id)`

估算我方和敌方最快能到达某个目标的时间，返回：

```python
(my_t, enemy_t)
```

### `WorldModel.projected_state(target_id, arrival_turn, planned_commitments=None, extra_arrivals=())`

查询某个目标在指定回合的预测状态。会把当前真实在飞舰队、本回合已计划的发兵、额外假设到达事件一起纳入模拟。

返回：

```python
(owner, ships)
```

### `WorldModel.projected_timeline(target_id, horizon, planned_commitments=None, extra_arrivals=())`

返回加入计划发兵和额外事件后的动态时间线。

### `WorldModel.hold_status(target_id, planned_commitments=None, horizon=HORIZON)`

返回某个星球的防守状态摘要：

```python
{
    "keep_needed": ...,
    "min_owned": ...,
    "first_enemy": ...,
    "fall_turn": ...,
    "holds_full": ...,
}
```

### `WorldModel._ownership_search_cap(eval_turn)`

给二分搜索兵力需求时使用的上界估计。它基于当前可见总船数、总产能和评估回合生成一个保守上限。

### `WorldModel.min_ships_to_own_by(...)`

计算为了在 `eval_turn` 回合拥有目标星球，攻击方最少需要在 `arrival_turn` 投入多少船。

它会通过二分搜索和动态时间线模拟求最小可行船数。

### `WorldModel.min_ships_to_own_at(...)`

`min_ships_to_own_by()` 的简化版本，要求到达回合和评估回合相同。

### `WorldModel.reinforcement_needed_to_hold_until(...)`

计算为了让目标星球一直守到 `hold_until`，在 `arrival_turn` 到达的支援最少需要多少船。

### `WorldModel.ships_needed_to_capture(...)`

计算我方在某个回合占领目标所需最少船数。是 `min_ships_to_own_at()` 的我方封装。

## 策略辅助函数

### `planet_distance(first, second)`

返回两个星球中心之间的距离。

### `nearest_sources_to_target(target, sources, top_k)`

从候选来源星球中选出距离目标最近的前 `top_k` 个。距离相同时，船更多、ID 更小的优先。

### `min_legal_reaction_time(target, sources, world)`

估算一组来源星球中，最快多久能合法打到目标。

### `policy_reaction_times(target_id, policy)`

从策略状态中读取目标的 `(my_t, enemy_t)`，不存在则返回极大值。

### `candidate_time_valid(target, turns, world, remaining_buffer)`

判断一个候选到达时间是否仍有意义。

- 太接近游戏结束则无效。
- 彗星目标如果会过期或追击时间太长则无效。

### `stacked_enemy_proactive_keep(planet, world)`

估计多个敌方星球在短窗口内同时威胁我方星球时，需要额外主动保留多少防守兵力。

### `swarm_eta_tolerance(options, target, world)`

返回多源合击允许的到达时间差。三源合击、敌方目标、普通目标使用不同容忍度。

### `detect_enemy_crashes(world)`

在到达账本中寻找敌方不同 owner 的舰队在同一目标、接近回合内可能互相抵消的事件，用于四人局中的 `crash_exploit`。

### `build_policy_state(world, deadline=None)`

构建当前回合的策略状态：

- `indirect_wealth_map`: 各目标间接战略价值。
- `reserve`: 每个我方星球需要保留的防守船数。
- `attack_budget`: 每个我方星球可用于进攻的船数。
- `reaction_time_map`: 我方和敌方对目标的最快反应时间。

会在接近时间预算时提前停止部分计算。

### `build_modes(world)`

根据兵力、产能和回合判断局势模式：

- 是否落后。
- 是否领先。
- 是否支配局面。
- 是否进入收官。
- 进攻冗余倍率。

### `is_safe_neutral(target, policy)`

判断中立目标是否是安全中立星。条件是我方最快到达时间明显早于敌方。

### `is_contested_neutral(target, policy)`

判断中立目标是否为争夺中立星。条件是我方和敌方最快到达时间差距较小。

### `opening_filter(target, arrival_turns, needed, src_available, world, policy)`

开局阶段过滤不值得抢的旋转中立星。它会放行高产、安全、短时间可达的目标；四人局有更严格的旋转中立星判断。

### `target_value(target, arrival_turns, mission, world, modes, policy)`

计算目标在某个到达时间和任务类型下的战略价值。

主要考虑：

- 产能和剩余回合。
- 间接战略价值。
- 静态/旋转星球。
- 中立/敌方目标。
- 安全中立或争夺中立。
- 彗星寿命。
- 任务类型倍率。
- 后期即时船数价值和消灭弱敌奖励。
- 当前领先、落后、收官状态。

### `reinforce_value(target, hold_until, world, policy)`

计算支援或防守某个我方星球的价值，重点考虑保住产能的收益和前线位置。

### `preferred_send(target, base_needed, arrival_turns, src_available, world, modes, policy)`

根据最低需求 `base_needed` 计算推荐发送船数。会加入目标类型、产能、静态、争夺程度、四人局、长距离、收官等冗余。

### `apply_score_modifiers(base_score, target, mission, world)`

对基础分数做额外修正，比如静态星球加权、开局静态中立加权、四人局旋转中立降权、snipe/swarm/crash_exploit 加权。

## 方案收敛与任务构建

### `settle_plan(...)`

把一个初始发兵猜测收敛成可执行方案。它会反复：

1. 用指定船数重新瞄准。
2. 根据实际到达回合计算最小占领需求。
3. 根据任务类型计算推荐发送量。
4. 若推荐量变化，则再次评估。

返回：

```python
(angle, turns, eval_turn, need, actual_send)
```

失败则返回 `None`。

### `settle_reinforce_plan(...)`

支援任务版本的 `settle_plan()`。目标不是占领，而是在指定时间前到达并让目标守到 `hold_until`。

返回：

```python
(angle, turns, hold_until, need, actual_send)
```

### `build_snipe_mission(src, target, src_available, world, planned_commitments, modes, policy)`

为中立目标构建截胡任务。它寻找敌方已经在飞向该中立星的舰队，然后尝试在敌方到达附近卡时间，让敌方先消耗守军，我方再以更小代价抢下。

### `build_rescue_missions(world, policy, planned_commitments, modes)`

构建救援任务。如果某个我方星球即将在防守窗口内失守，则尝试从其他我方星球提前派兵救援。

### `build_recapture_missions(world, policy, planned_commitments, modes)`

构建反抢任务。如果某个我方星球预计会失守，但可以在失守后短时间内重新夺回，则生成 recapture 任务。

### `build_reinforce_missions(world, policy, planned_commitments, modes, inventory_left_fn)`

构建提前支援任务。它偏向于高产、未来还值得保住的我方星球，并限制来源星球最多发出一定比例兵力。

### `build_crash_exploit_missions(world, policy, planned_commitments, modes)`

四人局专用任务。利用敌方不同玩家舰队在目标附近互相抵消后的空档，尝试卡时间抢下目标。

## 主规划函数

### `plan_moves(world, deadline=None)`

每回合的核心规划器，返回动作列表：

```python
[[src_id, angle, ships], ...]
```

主要流程：

1. 构建局势模式 `modes` 和策略状态 `policy`。
2. 创建防守、反抢、支援任务。
3. 遍历我方来源星球和所有非我方目标，生成单源进攻、截胡、多源合击候选。
4. 构建二源、三源合击任务。
5. 构建四人局碰撞利用任务。
6. 按任务分数排序执行。
7. 每执行一个动作后更新 `planned_commitments`，让后续任务看到已计划的到达事件。
8. 用剩余进攻预算做一轮 follow-up。
9. 对即将失守的星球做最后发兵或撤退。
10. 让后排星球向前线友方星球输送兵力。
11. 返回最终动作，并确保不会从同一星球发出超过现有船数。

### `plan_moves()` 内部辅助函数

这些函数只在 `plan_moves()` 内部使用：

- `expired()`: 判断是否超过本回合软时间预算。
- `time_left()`: 返回剩余规划时间。
- `allow_heavy_phase()`: 判断是否允许执行较重的路线和组合搜索。
- `allow_optional_phase()`: 判断是否允许执行可选补充策略。
- `source_inventory_left(source_id)`: 返回来源星球当前扣除已计划发兵后的剩余船数。
- `source_attack_left(source_id)`: 返回来源星球还能用于进攻的预算。
- `append_move(src_id, angle, ships)`: 添加动作并记录已花费船数。
- `finalize_moves()`: 最终清理动作，确保不会从同一星球超发。
- `compute_live_doomed()`: 找出短期内守不住且仍有可用船的我方星球。
- `time_filters_pass(target, turns, needed, src_cap)`: 合并后期、彗星、开局过滤条件。

## Agent 入口

### `_read(obs, key, default=None)`

兼容字典式 observation 和对象属性式 observation 的读取函数。

### `build_world(obs)`

从原始 `obs` 读取字段，构造 `Planet`、`Fleet`、`initial_by_id` 等结构，并返回 `WorldModel`。

### `agent(obs, config=None)`

Kaggle 环境调用的入口函数。

流程：

1. 记录开始时间。
2. 调用 `build_world(obs)`。
3. 若没有我方星球，返回空动作。
4. 根据 `config.actTimeout` 设置软时间预算。
5. 调用 `plan_moves(world, deadline)`。
6. 返回动作列表。

