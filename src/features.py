from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import EnvConfig
from .game_types import GameState, PlanetState, parse_observation
from .world_model import WorldModel

ARRIVAL_FLOW_HORIZON = 8
COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)

SELF_FEATURE_DIM = 42
CANDIDATE_FEATURE_DIM = 51
GLOBAL_FEATURE_DIM = 22


@dataclass(slots=True)
class DecisionContext:
    source_id: int
    candidate_ids: list[int]
    candidate_mask: np.ndarray
    target_angles: np.ndarray
    ship_options: np.ndarray
    ship_option_mask: np.ndarray


@dataclass(slots=True)
class TurnBatch:
    self_features: np.ndarray
    candidate_features: np.ndarray
    global_features: np.ndarray
    candidate_mask: np.ndarray
    ship_option_mask: np.ndarray
    contexts: list[DecisionContext]
    state: GameState


def self_feature_dim() -> int:
    return SELF_FEATURE_DIM


def candidate_feature_dim() -> int:
    return CANDIDATE_FEATURE_DIM


def global_feature_dim() -> int:
    return GLOBAL_FEATURE_DIM


def encode_turn(observation: object, cfg: EnvConfig, env_index: int = 0) -> TurnBatch:
    del env_index
    state = parse_observation(observation)
    world = WorldModel(state, cfg)
    sources = select_source_planets(
        [planet for planet in state.planets if planet.owner == state.player],
        cfg.max_source_planets,
    )
    ship_option_count = cfg.ship_bucket_count
    if ship_option_count != 4:
        raise ValueError("This implementation currently expects ship_bucket_count=4.")

    if not sources:
        return TurnBatch(
            self_features=np.zeros((0, SELF_FEATURE_DIM), dtype=np.float32),
            candidate_features=np.zeros((0, cfg.candidate_count, CANDIDATE_FEATURE_DIM), dtype=np.float32),
            global_features=np.zeros((0, GLOBAL_FEATURE_DIM), dtype=np.float32),
            candidate_mask=np.zeros((0, cfg.candidate_count), dtype=bool),
            ship_option_mask=np.zeros((0, cfg.candidate_count, ship_option_count), dtype=bool),
            contexts=[],
            state=state,
        )

    shared_global = build_global_features(world)
    self_rows: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    global_rows: list[np.ndarray] = []
    candidate_masks: list[np.ndarray] = []
    ship_option_masks: list[np.ndarray] = []
    contexts: list[DecisionContext] = []

    for src in sources:
        candidates = build_candidates(world, src, cfg.candidate_count - 1)
        candidate_ids = [-1] + [planet.id for planet in candidates]
        row_candidate_mask = np.zeros((cfg.candidate_count,), dtype=bool)
        row_candidate_mask[0] = True
        row_ship_option_mask = np.zeros((cfg.candidate_count, ship_option_count), dtype=bool)
        row_ship_option_mask[0, 0] = True
        row_target_angles = np.zeros((cfg.candidate_count, ship_option_count), dtype=np.float32)
        row_ship_options = np.zeros((cfg.candidate_count, ship_option_count), dtype=np.int64)
        row_candidate_features = np.zeros((cfg.candidate_count, CANDIDATE_FEATURE_DIM), dtype=np.float32)

        row_candidate_features[0] = build_noop_candidate_features(src, world)

        for slot, tgt in enumerate(candidates, start=1):
            options, option_mask = world.ship_options(src, tgt)
            adjusted_mask = []
            row_ship_options[slot] = np.asarray(options, dtype=np.int64)
            for ships, valid in zip(options, option_mask, strict=True):
                adjusted_mask.append(bool(valid) and not world.shot_is_invalid(src, tgt, ships))
            row_ship_option_mask[slot] = np.asarray(adjusted_mask, dtype=bool)
            row_candidate_mask[slot] = bool(any(adjusted_mask))
            for option_idx, ships in enumerate(options):
                if adjusted_mask[option_idx]:
                    row_target_angles[slot, option_idx] = float(world.aiming_angle(src, tgt, ships))
            row_candidate_features[slot] = build_candidate_features(src, tgt, world, cfg)

        self_rows.append(build_self_features(src, world, cfg))
        candidate_rows.append(row_candidate_features)
        global_rows.append(shared_global)
        candidate_masks.append(row_candidate_mask)
        ship_option_masks.append(row_ship_option_mask)
        contexts.append(
            DecisionContext(
                source_id=src.id,
                candidate_ids=candidate_ids,
                candidate_mask=row_candidate_mask,
                target_angles=row_target_angles,
                ship_options=row_ship_options,
                ship_option_mask=row_ship_option_mask,
            )
        )

    return TurnBatch(
        self_features=np.asarray(self_rows, dtype=np.float32),
        candidate_features=np.asarray(candidate_rows, dtype=np.float32),
        global_features=np.asarray(global_rows, dtype=np.float32),
        candidate_mask=np.asarray(candidate_masks, dtype=bool),
        ship_option_mask=np.asarray(ship_option_masks, dtype=bool),
        contexts=contexts,
        state=state,
    )


def build_candidates(world: WorldModel, src: PlanetState, limit: int) -> list[PlanetState]:
    if limit <= 0:
        return []

    others = [planet for planet in world.state.planets if planet.id != src.id]
    enemy_owners = set(world.enemy_owners())
    enemy_quota = max(1, int(math.ceil(limit * 0.4)))
    neutral_quota = max(1, int(math.ceil(limit * 0.4)))
    friendly_quota = max(limit - enemy_quota - neutral_quota, 0)

    enemies = ranked_candidates(
        [planet for planet in others if planet.owner in enemy_owners],
        world,
        src,
    )[:enemy_quota]
    neutrals = ranked_candidates(
        [planet for planet in others if planet.owner == -1],
        world,
        src,
    )[:neutral_quota]
    friendlies = ranked_candidates(
        [planet for planet in others if planet.owner == world.state.player],
        world,
        src,
    )[:friendly_quota]

    selected_ids = {planet.id for planet in enemies + neutrals + friendlies}
    ordered = enemies + neutrals + friendlies
    if len(ordered) >= limit:
        return ordered[:limit]

    fallback = ranked_candidates(
        [planet for planet in others if planet.id not in selected_ids],
        world,
        src,
    )
    ordered.extend(fallback[: limit - len(ordered)])
    return ordered


def select_source_planets(sources: list[PlanetState], limit: int) -> list[PlanetState]:
    if limit <= 0 or len(sources) <= limit:
        return sorted(sources, key=lambda planet: planet.id)
    selected = sorted(
        sources,
        key=lambda planet: (
            planet.ships,
            planet.production,
            planet.radius,
            -planet.id,
        ),
        reverse=True,
    )[:limit]
    return sorted(selected, key=lambda planet: planet.id)


def ranked_candidates(planets: list[PlanetState], world: WorldModel, src: PlanetState) -> list[PlanetState]:
    return sorted(
        planets,
        key=lambda planet: (
            candidate_is_currently_actionable(world, src, planet),
            world.strategic_value(planet, src),
            -math.hypot(planet.x - src.x, planet.y - src.y),
            -planet.id,
        ),
        reverse=True,
    )


def candidate_is_currently_actionable(world: WorldModel, src: PlanetState, tgt: PlanetState) -> float:
    options, option_mask = world.ship_options(src, tgt)
    for ships, valid in zip(options, option_mask, strict=True):
        if valid and not world.shot_is_invalid(src, tgt, ships):
            return 1.0
    return 0.0


def build_self_features(src: PlanetState, world: WorldModel, cfg: EnvConfig) -> np.ndarray:
    orbit_radius, orbit_sin, orbit_cos = world.polar_coords(src.x, src.y)
    comet_next_x, comet_next_y = world.comet_next_position(src)
    comet_next_radius, comet_next_sin, comet_next_cos = world.polar_coords(comet_next_x, comet_next_y)
    arrivals = world.arrival_summary(src.id, world.state.player, horizon=cfg.future_horizon)
    local_allies = world.local_planets(src, owner_filter={world.state.player})
    enemy_owners = set(world.enemy_owners())
    local_enemies = world.local_planets(src, owner_filter=enemy_owners)
    arrival_flow = normalize_arrival_flow(world.arrival_flow_vector(src.id, ARRIVAL_FLOW_HORIZON), cfg)

    values = [
        norm_pos(src.x, cfg),
        norm_pos(src.y, cfg),
        norm_dist(orbit_radius, cfg),
        orbit_sin,
        orbit_cos,
        norm_dist(src.radius, cfg),
        norm_ship_count(src.ships, cfg),
        norm_production_count(src.production, cfg),
        world.rotation_speed(src),
        world.comet_speed(src),
        norm_dist(src.radius if world.is_comet(src) else 0.0, cfg),
        norm_production_count(src.production if world.is_comet(src) else 0.0, cfg),
        world.comet_path_progress(src),
        norm_dist(comet_next_radius if world.is_comet(src) else 0.0, cfg),
        comet_next_sin if world.is_comet(src) else 0.0,
        comet_next_cos if world.is_comet(src) else 0.0,
        norm_eta(world.comet_remaining_life(src), cfg.future_horizon),
        norm_ship_flow(arrivals.friendly_ships, cfg),
        norm_ship_flow(arrivals.enemy_ships, cfg),
        norm_eta(arrivals.nearest_friendly_eta, cfg.future_horizon),
        norm_eta(arrivals.nearest_enemy_eta, cfg.future_horizon),
        len(local_allies) / max(cfg.max_planets, 1.0),
        len(local_enemies) / max(cfg.max_planets, 1.0),
        norm_ship_flow(sum(planet.ships for planet in local_allies), cfg),
        norm_ship_flow(sum(planet.ships for planet in local_enemies), cfg),
        world.has_overwhelming_enemy_fleet(src),
        norm_dist(world.nearest_planet_distance(src, enemy_owners), cfg),
        norm_dist(world.nearest_planet_distance(src, {world.state.player}), cfg),
        count_planets(world, {world.state.player}) / max(cfg.max_planets, 1.0),
        count_planets(world, enemy_owners) / max(cfg.max_planets, 1.0),
        norm_ship_total(world.my_total_ships(), cfg),
        norm_ship_total(world.strongest_enemy_total_ships(), cfg),
        norm_production_total(world.my_total_production(), cfg),
        norm_production_total(world.strongest_enemy_total_production(), cfg),
    ]
    values.extend(arrival_flow.tolist())
    return np.asarray(values, dtype=np.float32)


def build_candidate_features(src: PlanetState, tgt: PlanetState, world: WorldModel, cfg: EnvConfig) -> np.ndarray:
    owner_friend, owner_enemy, owner_neutral = owner_flags(tgt.owner, world.state.player)
    orbit_radius, orbit_sin, orbit_cos = world.polar_coords(tgt.x, tgt.y)
    comet_next_x, comet_next_y = world.comet_next_position(tgt)
    comet_next_radius, comet_next_sin, comet_next_cos = world.polar_coords(comet_next_x, comet_next_y)
    rel_dx = tgt.x - src.x
    rel_dy = tgt.y - src.y
    distance = math.hypot(rel_dx, rel_dy)
    bearing = math.atan2(rel_dy, rel_dx)
    planning_ships = choose_planning_ships(src, tgt, world)
    eta = max(1, int(math.ceil(distance / max(world.ship_speed(planning_ships), 1e-6))))
    expected_ships = world.target_expected_ships_at_eta(tgt, eta)
    expected_owner = world.target_expected_owner_at_eta(tgt, eta)
    exp_friend, exp_enemy, exp_neutral = owner_flags(expected_owner, world.state.player)
    future_radius, future_sin, future_cos = world.future_polar(tgt, eta)
    arrivals = world.arrival_summary(tgt.id, world.state.player, horizon=eta)
    inbound_friendly = arrivals.friendly_ships
    inbound_enemy = arrivals.enemy_ships
    capture_margin = planning_ships - expected_ships
    required_ships = max(float(tgt.ships + 1), 1.0)
    arrival_flow = normalize_arrival_flow(world.arrival_flow_vector(tgt.id, ARRIVAL_FLOW_HORIZON), cfg)
    enemy_owners = set(world.enemy_owners())
    local_friendly_support = world.local_support_ships(tgt, {world.state.player})
    local_enemy_support = world.local_support_ships(tgt, enemy_owners)

    values = [
        owner_friend,
        owner_enemy,
        owner_neutral,
        norm_pos(tgt.x, cfg),
        norm_pos(tgt.y, cfg),
        norm_dist(orbit_radius, cfg),
        orbit_sin,
        orbit_cos,
        norm_dist(tgt.radius, cfg),
        norm_ship_count(tgt.ships, cfg),
        norm_production_count(tgt.production, cfg),
        world.rotation_speed(tgt),
        world.comet_speed(tgt),
        norm_dist(tgt.radius if world.is_comet(tgt) else 0.0, cfg),
        norm_production_count(tgt.production if world.is_comet(tgt) else 0.0, cfg),
        world.comet_path_progress(tgt),
        norm_dist(comet_next_radius if world.is_comet(tgt) else 0.0, cfg),
        comet_next_sin if world.is_comet(tgt) else 0.0,
        comet_next_cos if world.is_comet(tgt) else 0.0,
        norm_eta(world.comet_remaining_life(tgt), cfg.future_horizon),
        norm_dist(rel_dx, cfg),
        norm_dist(rel_dy, cfg),
        norm_dist(distance, cfg),
        math.sin(bearing),
        math.cos(bearing),
        norm_eta(eta, cfg.future_horizon),
        norm_ship_flow(expected_ships, cfg),
        exp_friend,
        exp_enemy,
        exp_neutral,
        norm_ship_flow(inbound_friendly, cfg),
        norm_ship_flow(inbound_enemy, cfg),
        norm_ship_flow(capture_margin, cfg),
        safe_div(float(tgt.production), required_ships),
        safe_div(float(tgt.production), float(eta)),
        norm_dist(future_radius, cfg),
        future_sin,
        future_cos,
        1.0 if world.future_crosses_sun(src, tgt, planning_ships) else 0.0,
        norm_ship_flow(local_friendly_support, cfg),
        norm_ship_flow(local_enemy_support, cfg),
        float(world.strategic_value(tgt, src)),
        norm_ship_count(src.ships, cfg),
    ]
    values.extend(arrival_flow.tolist())
    return np.asarray(values, dtype=np.float32)


def build_noop_candidate_features(src: PlanetState, world: WorldModel) -> np.ndarray:
    del src, world
    return np.zeros((CANDIDATE_FEATURE_DIM,), dtype=np.float32)


def build_global_features(world: WorldModel) -> np.ndarray:
    cfg = world.cfg
    enemy_owners = set(world.enemy_owners())
    friendly_planets = [planet for planet in world.state.planets if planet.owner == world.state.player]
    enemy_planets = [planet for planet in world.state.planets if planet.owner in enemy_owners]
    neutral_planets = [planet for planet in world.state.planets if planet.owner == -1]
    friendly_fleet_ships = sum(fleet.ships for fleet in world.state.fleets if fleet.owner == world.state.player)
    enemy_fleet_ships = sum(fleet.ships for fleet in world.state.fleets if fleet.owner in enemy_owners)
    comet_planets = world.comet_planets()
    my_comets = sum(1 for planet in comet_planets if planet.owner == world.state.player)
    enemy_comets = sum(1 for planet in comet_planets if planet.owner in enemy_owners)
    neutral_comets = sum(1 for planet in comet_planets if planet.owner == -1)

    values = [
        min(world.state.step / max(cfg.episode_steps, 1), 1.0),
        len(friendly_planets) / max(cfg.max_planets, 1.0),
        len(enemy_planets) / max(cfg.max_planets, 1.0),
        len(neutral_planets) / max(cfg.max_planets, 1.0),
        norm_ship_total(world.my_total_ships(), cfg),
        norm_ship_total(world.strongest_enemy_total_ships(), cfg),
        norm_ship_total(friendly_fleet_ships, cfg),
        norm_ship_total(enemy_fleet_ships, cfg),
        norm_production_total(world.my_total_production(), cfg),
        norm_production_total(world.strongest_enemy_total_production(), cfg),
        world.normalized_ship_diff(),
        world.normalized_production_diff(),
        world.state.angular_velocity,
        1.0 if comet_planets else 0.0,
        len(comet_planets) / max(cfg.max_planets, 1.0),
        len(world.state.comet_groups) / max(cfg.max_planets, 1.0),
        my_comets / max(cfg.max_planets, 1.0),
        enemy_comets / max(cfg.max_planets, 1.0),
        neutral_comets / max(cfg.max_planets, 1.0),
        norm_ship_total(sum(planet.ships for planet in comet_planets), cfg),
        norm_production_total(sum(planet.production for planet in comet_planets), cfg),
        norm_eta(next_comet_spawn_eta(world.state.step), cfg.episode_steps),
    ]
    return np.asarray(values, dtype=np.float32)


def owner_flags(owner: int, player: int) -> tuple[float, float, float]:
    if owner == player:
        return 1.0, 0.0, 0.0
    if owner == -1:
        return 0.0, 0.0, 1.0
    return 0.0, 1.0, 0.0


def count_planets(world: WorldModel, owners: set[int]) -> int:
    return sum(1 for planet in world.state.planets if planet.owner in owners)


def choose_planning_ships(src: PlanetState, tgt: PlanetState, world: WorldModel) -> int:
    options, option_mask = world.ship_options(src, tgt)
    for ships, valid in zip(options, option_mask, strict=True):
        if valid:
            return ships
    return max(min(tgt.ships + 1, src.ships), 1)


def next_comet_spawn_eta(step: int) -> int:
    for spawn_step in COMET_SPAWN_STEPS:
        if spawn_step >= step:
            return spawn_step - step
    return 0


def norm_pos(value: float, cfg: EnvConfig) -> float:
    return float(value) / max(cfg.board_size, 1.0)


def norm_dist(value: float, cfg: EnvConfig) -> float:
    return float(value) / max(cfg.board_size, 1.0)


def norm_ship_count(value: float, cfg: EnvConfig) -> float:
    scale = max(cfg.max_ships, 1.0)
    return float(value) / scale


def norm_ship_flow(value: float, cfg: EnvConfig) -> float:
    scale = max(cfg.max_ships, 1.0)
    return float(value) / scale


def norm_ship_total(value: float, cfg: EnvConfig) -> float:
    scale = max(cfg.max_planets * cfg.max_ships, 1.0)
    return float(value) / scale


def norm_production_count(value: float, cfg: EnvConfig) -> float:
    scale = max(cfg.max_production, 1.0)
    return float(value) / scale


def norm_production_total(value: float, cfg: EnvConfig) -> float:
    scale = max(cfg.max_planets * cfg.max_production, 1.0)
    return float(value) / scale


def norm_eta(value: int | None, horizon: int) -> float:
    if value is None:
        return 1.0
    return min(float(value) / max(horizon, 1), 1.0)


def safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-6:
        return 0.0
    return numerator / denominator


def normalize_arrival_flow(flow: np.ndarray, cfg: EnvConfig) -> np.ndarray:
    scale = max(cfg.max_ships, 1.0)
    return (flow / scale).astype(np.float32)
