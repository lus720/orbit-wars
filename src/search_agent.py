from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch

from .config import TrainConfig, EnvConfig, SearchConfig
from .features import TurnBatch, encode_turn
from .game_types import GameState, PlanetState, parse_observation
from .policy import PlanetPolicy
from .simulator import OrbitSimulator
from .world_model import WorldModel


class SearchAgent:
    """Search-enhanced decision engine.

    For each source planet:
    1. Policy network proposes top-K action candidates (target × ships)
    2. Each candidate is evaluated via forward simulation + value network
    3. Best action is selected and executed

    Sources are processed greedily-sequential (strongest first).
    """

    def __init__(
        self,
        policy: PlanetPolicy,
        cfg: TrainConfig,
        device: torch.device,
        deterministic: bool = False,
    ) -> None:
        self.policy = policy
        self.cfg = cfg
        self.search_cfg = cfg.search
        self.env_cfg = cfg.env
        self.device = device
        self.deterministic = deterministic
        self.policy.eval()
        self.simulator = OrbitSimulator(self.env_cfg)

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, observation: Any) -> list[list[float | int]]:
        state = parse_observation(observation)
        batch = encode_turn(observation, self.env_cfg, env_index=0)

        if batch.self_features.shape[0] == 0:
            return []

        sources = [
            planet
            for planet in state.planets
            if planet.owner == state.player
        ]

        # Greedy sequential: sort sources by importance
        sources.sort(key=lambda p: (-p.ships, -p.production, p.id))

        all_moves: list[list[float | int]] = []
        temp_state = copy.deepcopy(state)

        for src in sources:
            src_batch = self._extract_src_batch(batch, src.id)
            if src_batch is None:
                continue

            # Generate action candidates
            candidates = self._get_candidates(src_batch, batch)

            if not candidates:
                continue

            # Evaluate each candidate
            best_action = None
            best_score = float("-inf")

            for cand in candidates:
                score = self._evaluate_candidate(cand, temp_state)
                if score > best_score:
                    best_score = score
                    best_action = cand

            if best_action is not None and best_score > -1.0:
                all_moves.append([
                    int(best_action["src_id"]),
                    float(best_action["angle"]),
                    int(best_action["ships"]),
                ])
                # Execute in temp state for next source's evaluation
                self._apply_to_state(temp_state, best_action)

        return all_moves

    # ------------------------------------------------------------------
    # Internal: candidate generation
    # ------------------------------------------------------------------

    def _extract_src_batch(
        self, batch: TurnBatch, src_id: int
    ) -> TurnBatch | None:
        """Extract the TurnBatch row for a single source planet."""
        for idx, ctx in enumerate(batch.contexts):
            if ctx.source_id == src_id:
                return TurnBatch(
                    self_features=batch.self_features[idx : idx + 1],
                    candidate_features=batch.candidate_features[idx : idx + 1],
                    global_features=batch.global_features[idx : idx + 1],
                    candidate_mask=batch.candidate_mask[idx : idx + 1],
                    ship_option_mask=batch.ship_option_mask[idx : idx + 1],
                    contexts=[ctx],
                    state=batch.state,
                )
        return None

    def _get_candidates(
        self, src_batch: TurnBatch, full_batch: TurnBatch
    ) -> list[dict]:
        """Generate action candidates from policy network top-K targets.

        Returns: [{target_idx, ship_idx, ships, src_id, angle, target_id, score}, ...]
        """
        features = src_batch
        if features.self_features.shape[0] == 0:
            return []

        t_features = (
            torch.from_numpy(features.self_features).to(self.device),
            torch.from_numpy(features.candidate_features).to(self.device),
            torch.from_numpy(features.global_features).to(self.device),
            torch.from_numpy(features.candidate_mask).to(self.device).bool(),
            torch.from_numpy(features.ship_option_mask).to(self.device).bool(),
        )

        candidates = self.policy.get_action_candidates(
            *t_features,
            contexts=features.contexts,
            top_k=self.search_cfg.top_k_targets,
        )
        return candidates[0] if candidates else []

    # ------------------------------------------------------------------
    # Internal: candidate evaluation
    # ------------------------------------------------------------------

    def _evaluate_candidate(self, cand: dict, state: GameState) -> float:
        """Score a candidate action analytically + value network."""
        cfg = self.env_cfg
        src_id = cand["src_id"]
        target_id = cand["target_id"]
        ships = cand["ships"]

        src = self._find_planet(state, src_id)
        tgt = self._find_planet(state, target_id)
        if src is None or tgt is None:
            return float("-inf")

        # --- Phase A: Analytical outcome estimation ---
        world = WorldModel(state, cfg)
        distance = math.hypot(tgt.x - src.x, tgt.y - src.y)
        speed = world.ship_speed(ships)
        eta = max(1, int(math.ceil(distance / max(speed, 1e-6))))

        expected_ships = world.target_expected_ships_at_eta(tgt, eta)
        expected_owner = world.target_expected_owner_at_eta(tgt, eta)

        # Can we capture?
        remaining_ships = 0.0
        capture = False
        will_strengthen_ally = False

        if tgt.owner == state.player:
            # Reinforcing own planet
            remaining_ships = ships
            capture = True
            will_strengthen_ally = True
        elif expected_owner == state.player:
            # Friendly fleet will arrive before us
            will_strengthen_ally = True
            remaining_ships = ships
        elif ships > expected_ships:
            # We capture: remaining = ships - expected_ships
            remaining_ships = ships - expected_ships
            capture = True

        # Heuristic score components
        production_gain = float(tgt.production if capture else 0)
        ships_spent = float(ships)
        ships_net = remaining_ships - ships_spent

        # Production efficiency (production per ship spent)
        prod_efficiency = production_gain / max(ships_spent, 1.0)

        # Distance penalty (closer = better)
        dist_ratio = distance / max(cfg.board_size, 1.0)

        # Score = analytical + value network
        analytical_score = (
            + 2.0 * ships_net / max(cfg.max_ships, 1.0)     # ship efficiency
            + 3.0 * prod_efficiency                           # production gain
            - 0.5 * dist_ratio                                # distance penalty
            + (0.5 if will_strengthen_ally else 0.0)          # reinforcement bonus
            + (0.3 if capture and target_id == src_id else 0.0)
        )

        # --- Phase B: Value network evaluation (if simulation horizon > 0) ---
        value_score = 0.0
        if self.search_cfg.simulation_horizon > 0:
            sim_state = copy.deepcopy(state)
            self._apply_to_state(sim_state, cand)
            value_score = self._simulate_evaluate(sim_state, eta)

        return analytical_score + self.search_cfg.heuristic_weight * value_score

    def _simulate_evaluate(self, state: GameState, eta: int) -> float:
        """Simulate forward and evaluate with value network."""
        horizon = min(self.search_cfg.simulation_horizon, eta + 5)

        # Simulate with no additional actions
        self.simulator.load_state(state)
        for _ in range(horizon):
            self.simulator.step([])

        # Encode simulated state and get value
        sim_obs = observation_from_state(self.simulator.state)
        sim_batch = encode_turn(sim_obs, self.env_cfg, env_index=0)

        if sim_batch.self_features.shape[0] == 0:
            return 0.0

        with torch.inference_mode():
            outputs = self.policy(
                torch.from_numpy(sim_batch.self_features).to(self.device),
                torch.from_numpy(sim_batch.candidate_features).to(self.device),
                torch.from_numpy(sim_batch.global_features).to(self.device),
                torch.from_numpy(sim_batch.candidate_mask).to(self.device).bool(),
                torch.from_numpy(sim_batch.ship_option_mask).to(self.device).bool(),
            )
            # Mean value across all source planets
            value = float(outputs.value.mean().cpu())

        return value

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _find_planet(self, state: GameState, planet_id: int) -> PlanetState | None:
        for p in state.planets:
            if p.id == planet_id:
                return p
        return None

    def _apply_to_state(self, state: GameState, cand: dict) -> None:
        """Apply action consequences to a state in-place (for sequential search)."""
        src = self._find_planet(state, cand["src_id"])
        if src is not None:
            src.ships = max(0, src.ships - cand["ships"])


def observation_from_state(state: GameState) -> dict[str, Any]:
    """Convert a GameState back to observation dict (for encode_turn)."""
    planets = [
        [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
        for p in state.planets
    ]
    fleets = [
        [f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships]
        for f in state.fleets
    ]
    initial_planets = [
        [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
        for p in state.initial_planets
    ]
    return {
        "step": state.step,
        "player": state.player,
        "planets": planets,
        "fleets": fleets,
        "angular_velocity": state.angular_velocity,
        "initial_planets": initial_planets,
    }
