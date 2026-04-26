
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class EnvConfig:
    board_size: float = 100.0
    episode_steps: int = 500
    sun_radius: float = 10.0
    ship_speed: float = 6.0
    comet_speed: float = 4.0
    candidate_count: int = 8
    ship_bucket_count: int = 4
    max_source_planets: int = 12
    min_attack_ships: int = 0
    future_horizon: int = 32
    local_radius: float = 20.0
    support_k: int = 3
    max_planets: int = 48
    max_ships: float = 400.0
    max_production: float = 5.0


@dataclass(slots=True)
class ModelConfig:
    hidden_size: int = 128
    learn_ship_policy: bool = False
    noop_logit_bias: float = -1.5
    heuristic_logit_scale: float = 1.5


@dataclass(slots=True)
class RewardConfig:
    use_shaping: bool = True
    gamma: float = 0.99
    alpha: float = 0.2
    w_ship_diff: float = 1.0
    w_production_diff: float = 0.35
    terminal_bonus: float = 1.0


@dataclass(slots=True)
class SearchConfig:
    top_k_targets: int = 3
    ship_options: int = 2
    simulation_horizon: int = 15
    greedy_sequential: bool = True
    beam_width: int = 5
    reward_discount: float = 0.97
    heuristic_weight: float = 0.5


@dataclass(slots=True)
class SearchTrainConfig:
    """Expert iteration training hyperparameters."""
    iterations: int = 20            # 专家迭代次数
    games_per_iter: int = 16        # 每次迭代收集数据的对局数
    epochs: int = 5                 # 每次迭代中监督训练的 epoch 数
    batch_size: int = 128           # 训练 batch size
    replay_buffer_size: int = 10000 # 经验回放缓冲区上限
    lr: float = 1e-4                # 学习率
    checkpoint_every: int = 5       # 每 N 次迭代保存一次 checkpoint


@dataclass(slots=True)
class TrainConfig:
    seed: int = 42
    run_name: str = "orbit_wars_template_ppo"
    device: str = "auto"
    save_dir: str = "artifacts/rl_template"
    checkpoint_every: int = 10
    log_every: int = 1
    eval_every: int = 50
    eval_games: int = 20
    eval_seed: int = 42
    opponent: str = "random"
    baseline_dir: str = "baselines"
    baseline_train_filter: str = "nearest_planet_sniper"
    mixed_self_weight: float = 0.7
    mixed_baseline_weight: float = 0.15
    mixed_random_weight: float = 0.15
    self_play_update_interval: int = 10
    self_play_deterministic: bool = False
    alternate_player_sides: bool = True
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    train_search: SearchTrainConfig = field(default_factory=SearchTrainConfig)


def default_train_config_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "default.yaml"


def load_train_config(path: str | Path) -> TrainConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {config_path}")
    return train_config_from_dict(data)


def train_config_from_dict(data: dict[str, Any]) -> TrainConfig:
    cfg = TrainConfig()
    _update_dataclass(cfg, data, skip={"env", "model", "reward", "search", "train_search"})
    _update_dataclass(cfg.env, data.get("env", {}))
    _update_dataclass(cfg.model, data.get("model", {}))
    _update_dataclass(cfg.reward, data.get("reward", {}))
    _update_dataclass(cfg.search, data.get("search", {}))
    _update_dataclass(cfg.train_search, data.get("train_search", {}))
    return cfg


def _update_dataclass(instance: Any, values: dict[str, Any], skip: set[str] | None = None) -> None:
    if not isinstance(values, dict):
        return
    skip = skip or set()
    for key, value in values.items():
        if key in skip or not hasattr(instance, key):
            continue
        default = getattr(instance, key)
        setattr(instance, key, _coerce_value(value, default))


def _coerce_value(value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value
