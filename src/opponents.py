
from __future__ import annotations

import importlib.util
import random
from pathlib import Path
from typing import Any, Callable, Protocol

import torch

from .config import TrainConfig
from .features import encode_turn
from .policy import PlanetPolicy, sample_actions


class OpponentPolicy(Protocol):
    def act(self, observation: Any) -> list[list[float | int]]:
        ...


class KaggleRandomOpponent:
    def __init__(self) -> None:
        from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent

        self._agent = random_agent

    def act(self, observation: Any) -> list[list[float | int]]:
        payload = {
            "player": obs_get(observation, "player", 0),
            "planets": list(obs_get(observation, "planets", [])),
        }
        return list(self._agent(payload))


class BaselinePoolOpponent:
    def __init__(self, baseline_dir: str | Path, seed: int = 0, name_filter: str = "") -> None:
        self.baseline_dir = Path(baseline_dir)
        selected_names = {
            item.strip().replace("-", "_")
            for item in name_filter.split(",")
            if item.strip()
        }
        self.paths = sorted(path for path in self.baseline_dir.glob("*.py") if not path.name.startswith("_"))
        if selected_names:
            self.paths = [
                path
                for path in self.paths
                if path.stem.replace("-", "_") in selected_names
            ]
        if not self.paths:
            raise FileNotFoundError(f"No baseline agent files found in {self.baseline_dir}")
        self.rng = random.Random(seed)
        self._agent: Callable[..., list[list[float | int]]] | None = None
        self._episode_index = 0

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng.seed(int(seed))
        path = self.rng.choice(self.paths)
        self._agent = load_baseline_agent(path, self._episode_index)
        self._episode_index += 1

    def act(self, observation: Any) -> list[list[float | int]]:
        if self._agent is None:
            self.reset()
        assert self._agent is not None
        return list(self._agent(observation))


class MixedOpponent:
    def __init__(self, cfg: TrainConfig, device: torch.device, seed: int = 0) -> None:
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.random_opponent = KaggleRandomOpponent()
        self.baseline_opponent = BaselinePoolOpponent(
            cfg.baseline_dir,
            seed=seed,
            name_filter=cfg.baseline_train_filter,
        )
        self.self_opponent = SelfPlayOpponent(cfg, device=device, deterministic=cfg.self_play_deterministic)
        self._active: OpponentPolicy = self.random_opponent
        self._episode_index = 0

    def sync_from(self, source_policy: PlanetPolicy) -> None:
        self.self_opponent.sync_from(source_policy)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng.seed(int(seed) + self._episode_index * 9973)
        choices: list[tuple[float, OpponentPolicy]] = [
            (max(self.cfg.mixed_self_weight, 0.0), self.self_opponent),
            (max(self.cfg.mixed_baseline_weight, 0.0), self.baseline_opponent),
            (max(self.cfg.mixed_random_weight, 0.0), self.random_opponent),
        ]
        total = sum(weight for weight, _ in choices)
        pick = self.rng.random() * total if total > 0.0 else 0.0
        cumulative = 0.0
        self._active = self.random_opponent
        for weight, opponent in choices:
            cumulative += weight
            if pick <= cumulative:
                self._active = opponent
                break
        reset_active = getattr(self._active, "reset", None)
        if callable(reset_active):
            reset_active(seed)
        self._episode_index += 1

    def act(self, observation: Any) -> list[list[float | int]]:
        return self._active.act(observation)


class SelfPlayOpponent:
    def __init__(self, cfg: TrainConfig, device: torch.device, deterministic: bool = True) -> None:
        from .features import candidate_feature_dim, global_feature_dim, self_feature_dim

        self.cfg = cfg
        self.device = device
        self.deterministic = deterministic
        self.policy = PlanetPolicy(
            self_dim=self_feature_dim(),
            candidate_dim=candidate_feature_dim(),
            global_dim=global_feature_dim(),
            candidate_count=cfg.env.candidate_count,
            ship_option_count=cfg.env.ship_bucket_count,
            hidden_size=cfg.model.hidden_size,
            noop_logit_bias=cfg.model.noop_logit_bias,
            heuristic_logit_scale=cfg.model.heuristic_logit_scale,
        ).to(device)
        self.policy.eval()

    def sync_from(self, source_policy: PlanetPolicy) -> None:
        self.policy.load_state_dict(source_policy.state_dict())
        self.policy.eval()

    def act(self, observation: Any) -> list[list[float | int]]:
        batch = encode_turn(observation, self.cfg.env, env_index=0)
        if batch.self_features.shape[0] == 0:
            return []
        with torch.inference_mode():
            outputs = self.policy(
                torch.from_numpy(batch.self_features).to(self.device),
                torch.from_numpy(batch.candidate_features).to(self.device),
                torch.from_numpy(batch.global_features).to(self.device),
                torch.from_numpy(batch.candidate_mask).to(self.device).bool(),
                torch.from_numpy(batch.ship_option_mask).to(self.device).bool(),
            )
            sampled = sample_actions(
                outputs,
                deterministic=self.deterministic,
                learn_ship_policy=self.cfg.model.learn_ship_policy,
            )
        target_indices = sampled.target_index.detach().cpu().numpy()
        ship_choices = sampled.ship_choice.detach().cpu().numpy()
        moves: list[list[float | int]] = []
        for row_idx, context in enumerate(batch.contexts):
            target_idx = int(target_indices[row_idx])
            if target_idx == 0:
                continue
            if target_idx >= len(context.candidate_ids):
                continue
            if not context.candidate_mask[target_idx]:
                continue
            ship_choice = int(ship_choices[row_idx])
            if ship_choice >= context.ship_options.shape[1]:
                continue
            if not context.ship_option_mask[target_idx, ship_choice]:
                continue
            ships = int(context.ship_options[target_idx, ship_choice])
            if ships <= 0:
                continue
            moves.append([context.source_id, float(context.target_angles[target_idx, ship_choice]), ships])
        return moves


def build_opponent(
    name: str,
    cfg: TrainConfig | None = None,
    device: torch.device | None = None,
) -> OpponentPolicy:
    if name == "random":
        return KaggleRandomOpponent()
    if name == "baseline":
        if cfg is None:
            raise ValueError("cfg is required for baseline opponent")
        return BaselinePoolOpponent(cfg.baseline_dir, seed=cfg.seed, name_filter=cfg.baseline_train_filter)
    if name == "mixed":
        if cfg is None or device is None:
            raise ValueError("cfg and device are required for mixed opponent")
        return MixedOpponent(cfg, device=device, seed=cfg.seed)
    if name == "self":
        if cfg is None or device is None:
            raise ValueError("cfg and device are required for self opponent")
        return SelfPlayOpponent(cfg, device=device, deterministic=cfg.self_play_deterministic)
    raise ValueError(f"Unknown opponent: {name}")


def obs_get(observation: Any, key: str, default: Any) -> Any:
    if isinstance(observation, dict):
        return observation.get(key, default)
    return getattr(observation, key, default)


def load_baseline_agent(path: Path, episode_index: int) -> Callable[..., list[list[float | int]]]:
    module_name = f"_orbit_wars_train_baseline_{path.stem.replace('-', '_')}_{episode_index}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load baseline module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = getattr(module, "agent", None)
    if agent is None or not callable(agent):
        raise AttributeError(f"Baseline file does not define callable agent(obs): {path}")
    return agent
