# PPO 训练代码文件说明（新手版）

这份文档面向刚接触这个项目的新手，目标是回答两个问题：

1. 每个文件是做什么的？
2. PPO 训练时，这些文件是怎样一起工作的？

项目训练的是 Kaggle `orbit_wars` 环境里的智能体。你可以把整个系统想成一条流水线：

```text
Kaggle 游戏环境
  -> 当前局面 observation
  -> 特征工程 features.py
  -> 神经网络 policy.py
  -> 采样动作 ppo.py
  -> 环境执行 env.py
  -> 收集奖励 train.py
  -> PPO 更新 ppo.py
  -> 保存 checkpoint
```

## 先看入口文件

### `src/train.py`

这是 PPO 训练的主入口，也是最应该先读的文件。

它主要做这些事：

- 读取训练参数，比如学习率、训练多少轮、一次收集多少步。
- 创建多个 `OrbitWarsEnv` 环境，让模型并行收集经验。
- 创建 `PlanetPolicy` 神经网络。
- 根据配置选择对手，可以是随机对手，也可以是 self-play 对手。
- 循环执行：
  - 用当前模型在环境里玩一小段，收集样本。
  - 计算每个动作对应的回报和优势。
  - 调用 PPO 更新函数训练模型。
  - 定期打印日志、保存 checkpoint。

重点函数：

- `main()`：训练总流程。
- `collect_rollout()`：让当前策略和环境交互，收集 PPO 需要的训练数据。
- `bootstrap_values()`：在 rollout 结束时估计后续价值。
- `merge_batches()`：把多个环境产生的 batch 拼起来。
- `save_checkpoint()`：保存模型权重和优化器状态。

对新手来说，可以先理解一句话：`train.py` 是“导演”，它安排环境、模型、数据收集、训练更新和保存。

## 配置相关

### `default_cfg.yaml`

这是训练参数配置文件。运行训练时通常会读取它。

当前代码里的 `default_train_config_path()` 默认指向 `src/configs/default.yaml`，但这个仓库实际看到的配置文件是根目录的 `default_cfg.yaml`。所以运行训练或评估脚本时，建议显式传入配置路径：

```bash
python -m src.train --config default_cfg.yaml
```

里面比较重要的字段：

- `seed`：随机种子，方便复现实验。
- `run_name`：本次训练的名字。
- `device`：使用 `cpu`、`cuda`，或 `auto` 自动选择。
- `save_dir`：模型保存目录。
- `opponent`：训练对手，`random` 表示随机对手，`self` 表示自博弈对手。
- `env.candidate_count`：每个星球最多考虑多少个候选目标。
- `model.hidden_size`：神经网络隐藏层大小。
- `ppo.rollout_steps`：每次更新前，每个环境先玩多少步。
- `ppo.num_envs`：并行环境数量。
- `ppo.total_updates`：总共更新多少轮。
- `ppo.lr`：学习率。
- `ppo.gamma`：奖励折扣因子。
- `ppo.clip_coef`：PPO 裁剪范围，防止策略变化过猛。
- `ppo.ent_coef`：熵奖励系数，鼓励探索。
- `ppo.vf_coef`：价值函数损失权重。

### `src/config.py`

这个文件负责把 YAML 配置加载成 Python 对象。

主要数据结构：

- `EnvConfig`：环境和特征归一化相关参数。
- `ModelConfig`：模型结构参数。
- `PPOConfig`：PPO 算法参数。
- `TrainConfig`：总配置，包含上面几类配置。

重点函数：

- `load_train_config()`：读取 YAML 文件。
- `train_config_from_dict()`：把字典转成 `TrainConfig`。
- `_coerce_value()`：把字符串、数字等配置值转成正确类型。

新手可以把它理解成“配置翻译器”：把 `default_cfg.yaml` 里的文字参数变成训练代码能用的对象。

## 环境封装

### `src/env.py`

这个文件负责把 Kaggle 的 `orbit_wars` 环境包装成训练代码更容易使用的形式。

核心类：

- `OrbitWarsEnv`
- `StepResult`

`OrbitWarsEnv` 做的事情：

- `reset()`：创建或重置一局游戏，返回当前局面的特征 batch。
- `step(player_action)`：接收模型动作，同时让对手也行动，然后推进游戏一步。
- 自动处理玩家是 0 号还是 1 号。
- 游戏结束时计算奖励。

这个文件还包含一些小工具：

- `default_make_fn()`：导入 Kaggle 的 `make`。
- `extract_observation()`：从 Kaggle 状态对象里取 observation。
- `extract_status()`：取当前玩家状态，比如是否还在游戏中。
- `extract_reward()`：取最终奖励。
- `terminal_reward()`：处理终局奖励。

新手可以把它理解成“游戏接口适配器”：Kaggle 原始环境比较复杂，`env.py` 把它包装成训练循环容易调用的 `reset/step` 模式。

## 游戏状态数据结构

### `src/game_types.py`

这个文件定义游戏状态的数据类型，让后面的代码不用直接处理杂乱的原始列表。

主要数据结构：

- `PlanetState`：一个星球的信息，包括 id、归属、坐标、半径、飞船数、产能。
- `FleetState`：一支舰队的信息，包括位置、方向、来源星球、飞船数。
- `GameState`：整局当前状态，包括当前步数、当前玩家、所有星球、所有舰队。

重点函数：

- `parse_observation()`：把 Kaggle 环境给出的 observation 转成 `GameState`。

新手可以把它理解成“数据整理层”：把原始游戏数据整理成有名字、有类型的结构。

## 特征工程

### `src/features.py`

这是 PPO 模型能不能学好的关键文件之一。它负责把游戏局面转成神经网络能吃的数字。

核心思想：

模型不是直接看整局游戏，而是对“我的每个星球”分别做一次决策。对于每个自己的星球，模型会在若干个候选目标里选一个：

- 选第 `0` 个目标表示“不行动”。
- 选其他目标表示“从当前星球向该目标发船”。

主要数据结构：

- `DecisionContext`：记录某一行特征对应哪个源星球、候选目标是谁、要发多少船、目标角度是多少。
- `TurnBatch`：一整回合的特征 batch，包括自身星球特征、候选目标特征、全局特征、候选 mask。

重点函数：

- `encode_turn()`：把一帧 observation 转成 `TurnBatch`。
- `build_candidates()`：为一个源星球挑选候选目标。
- `build_self_features()`：构造当前源星球自身特征。
- `build_candidate_features()`：构造候选目标特征。
- `build_global_features()`：构造全局局面特征。
- `fixed_ship_count()`：决定一次攻击派多少船。
- `shot_crosses_sun()`：判断发射路线是否穿过太阳。

特征大致分三类：

- `self_features`：当前源星球的信息，比如位置、飞船数、产能、我方/敌方总体兵力。
- `candidate_features`：候选目标的信息，比如目标归属、距离、飞船数、是否会穿过太阳。
- `global_features`：全局局面信息，比如游戏进度、我方星球数、敌方舰队数等。

`candidate_mask` 很重要。它告诉模型哪些候选动作是合法的。非法目标会在 `policy.py` 里被屏蔽掉。

新手可以把它理解成“把棋盘翻译成数字表格”的地方。

## 策略网络

### `src/policy.py`

这个文件定义神经网络，也就是 PPO 里的 Actor-Critic 模型。

核心类：

- `PlanetPolicy`
- `PolicyOutput`

`PlanetPolicy` 输入三类特征：

- 当前源星球特征。
- 候选目标特征。
- 全局局面特征。

它输出两样东西：

- `target_logits`：每个候选目标的分数。分数越高，越可能选择这个目标。
- `value`：模型认为当前局面的价值，用来辅助 PPO 计算优势和价值损失。

网络内部大致分成几块：

- `self_encoder`：编码源星球特征。
- `candidate_encoder`：编码候选目标特征。
- `global_encoder`：编码全局特征。
- `target_head`：输出每个候选目标的动作分数。
- `value_head`：输出当前状态价值。

注意：`target_logits` 会用 `candidate_mask` 把非法动作变成极小值，这样采样时基本不会选到非法动作。

新手可以把它理解成“模型大脑”：它决定每个星球应该打哪里，顺便估计当前局面好不好。

## PPO 算法核心

### `src/ppo.py`

这个文件实现 PPO 更新中最核心的数学部分。

主要数据结构：

- `SampledAction`：采样出来的动作、动作 log probability、熵。
- `TransitionBatch`：训练 PPO 所需的一批样本，包括特征、动作、旧 log probability、回报、优势。

重点函数：

- `sample_actions()`：根据模型输出选择动作。
  - 训练时通常随机采样，方便探索。
  - 评估时可以用确定性模式，直接选分数最高的动作。
- `action_log_prob_and_entropy()`：计算动作的 log probability 和熵。
- `safe_target_logits()`：处理极端情况下没有有效 logits 的行，避免分布计算崩掉。
- `ppo_update()`：真正执行 PPO 参数更新。

`ppo_update()` 大致做这些事：

1. 把 rollout 收集到的数据搬到 CPU/GPU。
2. 标准化 advantage，让训练更稳定。
3. 多轮 epoch 遍历同一批数据。
4. 每次取一个 minibatch。
5. 用当前策略重新计算动作概率。
6. 计算新旧策略概率比值 `ratio`。
7. 用 PPO clip 公式限制策略更新幅度。
8. 同时计算 value loss 和 entropy bonus。
9. 反向传播，更新神经网络。

新手可以把它理解成“学习规则”：告诉模型怎样从刚才玩的经验里变强，同时不要一步改得太离谱。

## 对手策略

### `src/opponents.py`

这个文件定义训练时模型面对的对手。

主要内容：

- `OpponentPolicy`：对手接口，只要求实现 `act(observation)`。
- `KaggleRandomOpponent`：调用 Kaggle 自带随机 agent。
- `SelfPlayOpponent`：用另一个 `PlanetPolicy` 当对手，也就是自博弈。
- `build_opponent()`：根据配置创建对手。

`SelfPlayOpponent` 的关键点：

- 它内部也有一个 `PlanetPolicy`。
- 训练开始或每隔一段时间，会从当前正在训练的 policy 同步参数。
- 这样模型会逐渐和“过去版本的自己”对战。

新手可以把它理解成“陪练选择器”：可以找随机对手练，也可以让模型跟自己练。

## 评估和回放

### `eval_vs_sniper.py`

这个脚本用来评估训练好的模型，看看它对一个固定规则对手表现如何。

它会：

- 读取配置。
- 创建同样结构的 `PlanetPolicy`。
- 如果传入 checkpoint，就加载训练好的权重。
- 跑多局游戏。
- 输出每局胜负和总体胜率。

脚本里的 `nearest_planet_sniper()` 是一个简单规则对手：每个星球优先攻击最近的非己方星球。

常见用法示例：

```bash
python eval_vs_sniper.py --config default_cfg.yaml --checkpoint artifacts/ckpt_last.pt --games 20 --deterministic
```

### `play_vs_sniper.py`

这个脚本和 `eval_vs_sniper.py` 很像，但它只跑一局，并把对局保存成 HTML 回放。

它会：

- 加载模型。
- 和 `nearest_planet_sniper()` 对战一局。
- 调用 `env.render(mode="html")` 生成回放。
- 把回放写入指定路径。

常见用法示例：

```bash
python play_vs_sniper.py --config default_cfg.yaml --checkpoint artifacts/ckpt_last.pt --output artifacts/replays/vs_sniper.html --deterministic
```

新手可以把这两个文件理解成“考试工具”：训练完后看看模型到底会不会玩。

## 其他文件

### `src/__init__.py`

这是 Python 包标记文件。它让 `src` 可以被当成一个包导入，例如：

```python
from src.policy import PlanetPolicy
```

当前文件本身没有复杂逻辑。

### `main.py`

这是 Kaggle 提交时通常需要的入口文件。不过当前仓库里的 `main.py` 为空，所以它还没有把 PPO 模型封装成 Kaggle 可以直接调用的 agent。

如果后续要把训练好的 PPO 模型提交到 Kaggle，通常需要在这里写一个函数，比如：

```python
def agent(observation, configuration):
    ...
```

这个函数需要加载或内嵌模型逻辑，并返回 Orbit Wars 环境要求的动作列表。

### `README.md`

项目说明和常用命令记录。当前主要记录了 Kaggle CLI、比赛页面、提交、查看排行榜、快速测试等命令。

### Notebook 文件

仓库里有几个 `.ipynb` 文件：

- `getting-started.ipynb`
- `orbit-wars-reinforcement-learning-tutorial.ipynb`
- `orbit-wars-reinforcement-learning-tutorial-zh.ipynb`
- `test.ipynb`

它们更适合交互式学习、实验和展示。真正可复用的 PPO 训练代码主要在 `src/` 和几个脚本文件里。

### `otherMethods/` 和 `.generated_methods/`

这些目录里是其他策略或生成出来的 baseline/提交脚本。它们不是 PPO 训练主流程的一部分，但可以作为规则策略参考，或者用来对比 PPO 模型表现。

## PPO 训练时的完整流程

下面按一次训练更新来走一遍：

1. `src/train.py` 读取 `default_cfg.yaml`。
2. `src/config.py` 把 YAML 转成 `TrainConfig`。
3. `src/train.py` 创建 `OrbitWarsEnv`。
4. `src/env.py` 调用 Kaggle `make("orbit_wars")` 创建游戏。
5. 游戏返回 observation。
6. `src/features.py` 的 `encode_turn()` 把 observation 转成 `TurnBatch`。
7. `src/policy.py` 的 `PlanetPolicy` 根据特征输出动作分数和状态价值。
8. `src/ppo.py` 的 `sample_actions()` 从动作分数里采样目标。
9. `src/train.py` 把目标编号转换成 Orbit Wars 需要的动作格式：

```text
[source_planet_id, angle, ships]
```

10. `src/env.py` 执行动作，同时调用 `src/opponents.py` 生成对手动作。
11. 环境进入下一步，直到收集够 `rollout_steps`。
12. `src/train.py` 计算 returns 和 advantages。
13. `src/ppo.py` 的 `ppo_update()` 用这批数据更新模型。
14. `src/train.py` 定期保存 checkpoint。

## 新手阅读顺序建议

建议按这个顺序读：

1. `default_cfg.yaml`：先知道训练参数长什么样。
2. `src/train.py` 的 `main()`：了解总流程。
3. `src/env.py`：了解游戏环境怎么接进来。
4. `src/features.py`：了解 observation 怎样变成模型输入。
5. `src/policy.py`：了解神经网络输入输出。
6. `src/ppo.py`：了解 PPO 怎么更新。
7. `src/opponents.py`：了解训练对手。
8. `eval_vs_sniper.py` 和 `play_vs_sniper.py`：了解怎么评估。

## 一句话总结

- `config.py`：读配置。
- `env.py`：连游戏。
- `game_types.py`：整理游戏状态。
- `features.py`：把游戏状态变成数字特征。
- `policy.py`：神经网络做决策和估值。
- `ppo.py`：实现 PPO 采样和更新。
- `opponents.py`：提供训练对手。
- `train.py`：把所有模块串起来训练。
- `eval_vs_sniper.py`：批量评估模型。
- `play_vs_sniper.py`：生成单局可视化回放。
- `default_cfg.yaml`：控制训练超参数。
- `main.py`：预留给 Kaggle 提交入口，目前为空。
