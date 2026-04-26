
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import TrainConfig
from .features import TurnBatch, encode_turn
from .game_types import parse_observation
from .opponents import OpponentPolicy
from .world_model import WorldModel


@dataclass(slots=True)
class StepResult:
    batch: TurnBatch
    reward: float
    done: bool
    info: dict[str, Any]


class OrbitWarsEnv:
    def __init__(
        self,
        cfg: TrainConfig,
        opponent: OpponentPolicy,
        make_fn: Any | None = None,
        env_index: int = 0,
    ) -> None:
        self.cfg = cfg
        self.opponent = opponent
        self.make_fn = make_fn
        self.env_index = env_index
        self.env: Any | None = None
        self.last_obs: Any | None = None
        self.last_opp_obs: Any | None = None
        self.episode_index = 0
        self.learner_player = 0

    def reset(self, seed: int | None = None) -> TurnBatch:
        make_fn = self.make_fn or default_make_fn()
        configuration: dict[str, Any] = {}
        if seed is not None:
            configuration["seed"] = int(seed)
            configuration["randomSeed"] = int(seed)
        if self.cfg.alternate_player_sides:
            self.learner_player = (self.env_index + self.episode_index) % 2
        else:
            self.learner_player = 0
        self.env = make_fn("orbit_wars", configuration=configuration, debug=False)
        self.env.reset(num_agents=2)
        reset_opponent = getattr(self.opponent, "reset", None)
        if callable(reset_opponent):
            reset_opponent(seed)
        states = self.env.step([[], []])
        learner_state = states[self.learner_player]
        opponent_state = states[1 - self.learner_player]
        self.last_obs = extract_observation(learner_state)
        self.last_opp_obs = extract_observation(opponent_state)
        self.episode_index += 1
        return encode_turn(self.last_obs, self.cfg.env, env_index=self.env_index)

    def step(self, player_action: list[list[float | int]]) -> StepResult:
        if self.env is None:
            raise RuntimeError("Call reset() before step().")
        prev_obs = self.last_obs
        opponent_action = self.opponent.act(self.last_opp_obs)
        if self.learner_player == 0:
            joint_action = [player_action, opponent_action]
        else:
            joint_action = [opponent_action, player_action]
        states = self.env.step(joint_action)
        player_state = states[self.learner_player]
        opp_state = states[1 - self.learner_player]
        self.last_obs = extract_observation(player_state)
        self.last_opp_obs = extract_observation(opp_state)
        done = extract_status(player_state) != "ACTIVE"
        reward = compute_reward(self.cfg, prev_obs, self.last_obs, player_state, opp_state, done)
        batch = encode_turn(self.last_obs, self.cfg.env, env_index=self.env_index)
        info = {
            "learner_player": self.learner_player,
            "player_status": extract_status(player_state),
            "opponent_status": extract_status(opp_state),
            "reward": reward,
        }
        return StepResult(batch=batch, reward=reward, done=done, info=info)


def default_make_fn() -> Any:
    from kaggle_environments import make

    return make


def extract_observation(state: Any) -> Any:
    if isinstance(state, dict):
        return state.get("observation")
    return getattr(state, "observation")


def extract_status(state: Any) -> str:
    if isinstance(state, dict):
        return str(state.get("status", "UNKNOWN"))
    return str(getattr(state, "status", "UNKNOWN"))


def extract_reward(state: Any) -> float:
    if isinstance(state, dict):
        value = state.get("reward", 0.0)
    else:
        value = getattr(state, "reward", 0.0)
    return 0.0 if value is None else float(value)


def terminal_reward(player_state: Any, opp_state: Any) -> float:
    player_reward = extract_reward(player_state)
    opponent_reward = extract_reward(opp_state)
    if player_reward > 0.0 and opponent_reward > 0.0:
        return 0.0
    return player_reward


def compute_reward(
    cfg: TrainConfig,
    prev_obs: Any,
    next_obs: Any,
    player_state: Any,
    opp_state: Any,
    done: bool,
) -> float:
    terminal = terminal_reward(player_state, opp_state) if done else 0.0
    if not cfg.reward.use_shaping or prev_obs is None or next_obs is None:
        return cfg.reward.terminal_bonus * terminal if done else 0.0

    prev_world = WorldModel(parse_observation(prev_obs), cfg.env)
    next_world = WorldModel(parse_observation(next_obs), cfg.env)
    shaping = cfg.reward.alpha * (
        cfg.reward.gamma * next_world.shaping_potential(cfg.reward) - prev_world.shaping_potential(cfg.reward)
    )
    if done:
        shaping += cfg.reward.terminal_bonus * terminal
    return float(shaping)
