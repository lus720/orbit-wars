
from __future__ import annotations

import argparse
import importlib
import importlib.util
import random
import sys
import types
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import TrainConfig, default_train_config_path, load_train_config
from src.features import TurnBatch, candidate_feature_dim, encode_turn, global_feature_dim, self_feature_dim
from src.policy import PlanetPolicy
from src.search_agent import SearchAgent

BaselineAgent = Callable[[Any], list[list[float | int]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint against every baseline agent.")
    parser.add_argument("--config", type=str, default=str(default_train_config_path()))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--baselines-dir", type=str, default="baselines")
    parser.add_argument("--baseline", action="append", default=None, help="Run only matching baseline names.")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_policy(cfg: TrainConfig, device: torch.device) -> PlanetPolicy:
    return PlanetPolicy(
        self_dim=self_feature_dim(),
        candidate_dim=candidate_feature_dim(),
        global_dim=global_feature_dim(),
        candidate_count=cfg.env.candidate_count,
        ship_option_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
        noop_logit_bias=cfg.model.noop_logit_bias,
        heuristic_logit_scale=cfg.model.heuristic_logit_scale,
    ).to(device)

def register_checkpoint_module_aliases() -> None:
    sys.modules.setdefault("src", types.ModuleType("src"))
    sys.modules.setdefault("src.rl_template", types.ModuleType("src.rl_template"))
    module_candidates = {
        "config": ["src.rl_template.config", "src.config", "config"],
        "features": ["src.rl_template.features", "src.features", "features"],
        "policy": ["src.rl_template.policy", "src.policy", "policy"],
        "game_types": ["src.rl_template.game_types", "src.game_types", "game_types"],
        "opponents": ["src.rl_template.opponents", "src.opponents", "opponents"],
        "env": ["src.rl_template.env", "src.env", "env"],
        "train": ["src.rl_template.train", "src.train", "train"],
    }

    for canonical_name, candidates in module_candidates.items():
        module = None
        for candidate in candidates:
            try:
                module = importlib.import_module(candidate)
                break
            except ModuleNotFoundError:
                continue
        if module is None:
            continue
        sys.modules[f"src.rl_template.{canonical_name}"] = module
        sys.modules[f"src.{canonical_name}"] = module

def load_checkpoint_if_available(policy: PlanetPolicy, checkpoint_path: str | None, device: torch.device) -> None:
    register_checkpoint_module_aliases()
    if checkpoint_path is None:
        return
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("policy", checkpoint)
    policy.load_state_dict(state_dict)


def baseline_name_from_path(path: Path) -> str:
    return path.stem.replace("-", "_")


def discover_baselines(baselines_dir: str | Path, selected_names: list[str] | None = None) -> list[Path]:
    base_dir = Path(baselines_dir)
    selected = {name.replace("-", "_") for name in selected_names or []}
    paths = sorted(path for path in base_dir.glob("*.py") if not path.name.startswith("_"))
    if selected:
        paths = [path for path in paths if baseline_name_from_path(path) in selected or path.stem in selected]
    if not paths:
        raise FileNotFoundError(f"No baseline agent files found in {base_dir}")
    return paths


def load_baseline_agent(path: Path, game_index: int) -> BaselineAgent:
    module_name = f"_orbit_wars_baseline_{baseline_name_from_path(path)}_{game_index}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load baseline module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = getattr(module, "agent", None)
    if agent is None or not callable(agent):
        raise AttributeError(f"Baseline file does not define callable agent(obs): {path}")
    return agent


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


def build_search_agent(policy: PlanetPolicy, cfg: TrainConfig, device: torch.device) -> SearchAgent:
    return SearchAgent(policy, cfg, device, deterministic=True)


def play_one_game(
    cfg: TrainConfig,
    policy: PlanetPolicy,
    device: torch.device,
    baseline_path: Path,
    *,
    seed: int,
    game_index: int,
    deterministic: bool,
) -> tuple[float, int]:
    from kaggle_environments import make

    search_agent = build_search_agent(policy, cfg, device)
    env = make(
        "orbit_wars",
        configuration={"seed": int(seed), "randomSeed": int(seed)},
        debug=False,
    )
    env.reset(num_agents=2)
    states = env.step([[], []])
    player_obs = extract_observation(states[0])
    opponent_obs = extract_observation(states[1])
    done = extract_status(states[0]) != "ACTIVE"
    step_count = 0
    baseline_agent = load_baseline_agent(baseline_path, game_index)

    while not done:
        player_action = search_agent.act(player_obs) if player_obs else []
        opponent_action = baseline_agent(opponent_obs)
        states = env.step([player_action, opponent_action])
        player_obs = extract_observation(states[0])
        opponent_obs = extract_observation(states[1])
        done = extract_status(states[0]) != "ACTIVE"
        step_count += 1

    return extract_reward(states[0]), step_count


def reward_to_label(reward: float) -> str:
    if reward > 0:
        return "win"
    if reward < 0:
        return "loss"
    return "draw"


def main() -> None:
    args = parse_args()
    cfg = load_train_config(args.config)
    device_name = args.device if args.device != "auto" else cfg.device
    device = resolve_device(device_name)
    seed_everything(args.seed)
    policy = build_policy(cfg, device)
    load_checkpoint_if_available(policy, args.checkpoint, device)
    policy.eval()
    baseline_paths = discover_baselines(args.baselines_dir, args.baseline)

    for baseline_path in baseline_paths:
        baseline_name = baseline_name_from_path(baseline_path)
        wins = 0
        draws = 0
        losses = 0
        print(f"baseline={baseline_name}")

        for game_idx in range(args.games):
            game_seed = args.seed + game_idx
            reward, steps = play_one_game(
                cfg,
                policy,
                device,
                baseline_path,
                seed=game_seed,
                game_index=game_idx,
                deterministic=args.deterministic,
            )
            label = reward_to_label(reward)
            if label == "win":
                wins += 1
            elif label == "loss":
                losses += 1
            else:
                draws += 1
            print(
                f"game={game_idx + 1} seed={game_seed} baseline={baseline_name} "
                f"result={label} reward={reward:.1f} steps={steps}"
            )

        total_games = max(args.games, 1)
        win_rate = wins / total_games
        print(
            f"summary baseline={baseline_name} wins={wins} losses={losses} "
            f"draws={draws} games={args.games} win_rate={win_rate:.4f}"
        )


if __name__ == "__main__":
    main()
