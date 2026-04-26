from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import EnvConfig, RewardConfig
from .game_types import CometGroupState, FleetState, GameState, PlanetState

BOARD_CENTER = (50.0, 50.0)
MAX_LOG_SHIPS = math.log(1000.0)


@dataclass(slots=True)
class FleetTargetInfo:
    target_id: int | None
    distance: float
    eta: int
    blocked_by_sun: bool


@dataclass(slots=True)
class PlanetArrivalSummary:
    friendly_ships: int
    enemy_ships: int
    nearest_friendly_eta: int | None
    nearest_enemy_eta: int | None


@dataclass
class WorldModel:
    state: GameState
    cfg: EnvConfig

    def __post_init__(self) -> None:
        self.center = (self.cfg.board_size / 2.0, self.cfg.board_size / 2.0)
        self.planet_by_id = {planet.id: planet for planet in self.state.planets}
        self.initial_planet_by_id = {planet.id: planet for planet in self.state.initial_planets}
        self.comet_group_by_planet: dict[int, CometGroupState] = {}
        self.comet_path_by_planet: dict[int, tuple[tuple[float, float], ...]] = {}
        for group in self.state.comet_groups:
            for idx, planet_id in enumerate(group.planet_ids):
                self.comet_group_by_planet[planet_id] = group
                if idx < len(group.paths):
                    self.comet_path_by_planet[planet_id] = group.paths[idx]

        self._polar_cache: dict[tuple[float, float], tuple[float, float, float]] = {}
        self._is_rotating_cache: dict[int, bool] = {}
        self._planet_position_cache: dict[tuple[int, int], tuple[float, float]] = {}
        self._future_polar_cache: dict[tuple[int, int], tuple[float, float, float]] = {}
        self._ship_speed_cache: dict[int, float] = {}
        self._future_crosses_sun_cache: dict[tuple[int, int, int], bool] = {}
        self._aiming_angle_cache: dict[tuple[int, int, int], float] = {}
        self._arrival_summary_cache: dict[tuple[int, int, int | None], PlanetArrivalSummary] = {}
        self._local_planets_cache: dict[tuple[int, tuple[int, ...] | None, float], list[PlanetState]] = {}
        self._ship_options_cache: dict[tuple[int, int], tuple[tuple[int, ...], tuple[bool, ...]]] = {}
        self._capture_need_cache: dict[tuple[int, int], int] = {}

        self.fleet_target_cache: dict[int, FleetTargetInfo] = {}
        self.arrivals_by_planet: dict[int, list[tuple[int, FleetState]]] = {planet.id: [] for planet in self.state.planets}
        for fleet in self.state.fleets:
            target = self.infer_fleet_target(fleet)
            self.fleet_target_cache[fleet.id] = target
            if target.target_id is not None:
                self.arrivals_by_planet.setdefault(target.target_id, []).append((target.eta, fleet))
        for entries in self.arrivals_by_planet.values():
            entries.sort(key=lambda item: item[0])

    def polar_coords(self, x: float, y: float) -> tuple[float, float, float]:
        key = (round(float(x), 4), round(float(y), 4))
        cached = self._polar_cache.get(key)
        if cached is not None:
            return cached
        dx = x - self.center[0]
        dy = y - self.center[1]
        radius = math.hypot(dx, dy)
        angle = math.atan2(dy, dx)
        result = (radius, math.sin(angle), math.cos(angle))
        self._polar_cache[key] = result
        return result

    def is_rotating(self, planet: PlanetState) -> bool:
        cached = self._is_rotating_cache.get(planet.id)
        if cached is not None:
            return cached
        radius, _, _ = self.polar_coords(planet.x, planet.y)
        result = radius + planet.radius < 50.0
        self._is_rotating_cache[planet.id] = result
        return result

    def rotation_speed(self, planet: PlanetState) -> float:
        return self.state.angular_velocity if self.is_rotating(planet) else 0.0

    def is_comet(self, planet: PlanetState) -> bool:
        return planet.id in self.state.comet_planet_ids

    def comet_speed(self, planet: PlanetState) -> float:
        return self.cfg.comet_speed if self.is_comet(planet) else 0.0

    def comet_path_progress(self, planet: PlanetState) -> float:
        if not self.is_comet(planet):
            return 0.0
        group = self.comet_group_by_planet.get(planet.id)
        path = self.comet_path_by_planet.get(planet.id, ())
        if group is None or not path:
            return 0.0
        return float(group.path_index) / max(len(path) - 1, 1)

    def comet_remaining_life(self, planet: PlanetState) -> int:
        if not self.is_comet(planet):
            return 0
        group = self.comet_group_by_planet.get(planet.id)
        path = self.comet_path_by_planet.get(planet.id, ())
        if group is None or not path:
            return 0
        return max(len(path) - group.path_index - 1, 0)

    def comet_next_position(self, planet: PlanetState) -> tuple[float, float]:
        if not self.is_comet(planet):
            return planet.x, planet.y
        group = self.comet_group_by_planet.get(planet.id)
        path = self.comet_path_by_planet.get(planet.id, ())
        if group is None or not path:
            return planet.x, planet.y
        idx = min(group.path_index + 1, len(path) - 1)
        return path[idx]

    def planet_position_at(self, planet: PlanetState, delta_turns: int) -> tuple[float, float]:
        delta_turns = max(int(delta_turns), 0)
        key = (planet.id, delta_turns)
        cached = self._planet_position_cache.get(key)
        if cached is not None:
            return cached
        if self.is_comet(planet):
            group = self.comet_group_by_planet.get(planet.id)
            path = self.comet_path_by_planet.get(planet.id, ())
            if group is None or not path:
                return planet.x, planet.y
            idx = min(group.path_index + delta_turns, len(path) - 1)
            result = path[idx]
            self._planet_position_cache[key] = result
            return result

        if not self.is_rotating(planet):
            result = (planet.x, planet.y)
            self._planet_position_cache[key] = result
            return result

        initial = self.initial_planet_by_id.get(planet.id, planet)
        dx = initial.x - self.center[0]
        dy = initial.y - self.center[1]
        orbital_radius = math.hypot(dx, dy)
        theta0 = math.atan2(dy, dx)
        rotations = max(0, self.state.step - 1 + delta_turns)
        theta = theta0 + self.state.angular_velocity * rotations
        result = (
            self.center[0] + orbital_radius * math.cos(theta),
            self.center[1] + orbital_radius * math.sin(theta),
        )
        self._planet_position_cache[key] = result
        return result

    def future_polar(self, planet: PlanetState, delta_turns: int) -> tuple[float, float, float]:
        key = (planet.id, max(int(delta_turns), 0))
        cached = self._future_polar_cache.get(key)
        if cached is not None:
            return cached
        x, y = self.planet_position_at(planet, delta_turns)
        result = self.polar_coords(x, y)
        self._future_polar_cache[key] = result
        return result

    def ship_speed(self, ships: int) -> float:
        ships = max(int(ships), 1)
        cached = self._ship_speed_cache.get(ships)
        if cached is not None:
            return cached
        if ships <= 1:
            self._ship_speed_cache[ships] = 1.0
            return 1.0
        ratio = math.log(float(ships)) / MAX_LOG_SHIPS
        result = 1.0 + (self.cfg.ship_speed - 1.0) * max(ratio, 0.0) ** 1.5
        self._ship_speed_cache[ships] = result
        return result

    def infer_fleet_target(self, fleet: FleetState) -> FleetTargetInfo:
        direction = (math.cos(fleet.angle), math.sin(fleet.angle))
        origin = (fleet.x, fleet.y)
        nearest_t = float("inf")
        target_id: int | None = None
        for planet in self.state.planets:
            if planet.id == fleet.from_planet_id:
                continue
            t = ray_circle_intersection(origin, direction, (planet.x, planet.y), planet.radius)
            if t is None or t < 0.0:
                continue
            if t < nearest_t:
                nearest_t = t
                target_id = planet.id

        if target_id is None:
            return FleetTargetInfo(target_id=None, distance=0.0, eta=0, blocked_by_sun=False)

        speed = self.ship_speed(fleet.ships)
        eta = max(1, int(math.ceil(nearest_t / max(speed, 1e-6))))
        target_planet = self.planet_by_id[target_id]
        blocked_by_sun = self.segment_crosses_sun(origin, (target_planet.x, target_planet.y))
        return FleetTargetInfo(target_id=target_id, distance=nearest_t, eta=eta, blocked_by_sun=blocked_by_sun)

    def segment_crosses_sun(self, start: tuple[float, float], end: tuple[float, float]) -> bool:
        return point_to_segment_distance(self.center, start, end) < self.cfg.sun_radius

    def aiming_angle(self, src: PlanetState, tgt: PlanetState, ships: int) -> float:
        key = (src.id, tgt.id, max(int(ships), 1))
        cached = self._aiming_angle_cache.get(key)
        if cached is not None:
            return cached
        if self.shot_is_invalid(src, tgt, ships):
            self._aiming_angle_cache[key] = 0.0
            return 0.0
        if not self.is_rotating(tgt):
            result = math.atan2(tgt.y - src.y, tgt.x - src.x)
            self._aiming_angle_cache[key] = result
            return result
        speed = self.ship_speed(ships)
        current_distance = math.hypot(tgt.x - src.x, tgt.y - src.y)
        eta = max(1, int(math.ceil(current_distance / max(speed, 1e-6))))
        predicted_x, predicted_y = self.planet_position_at(tgt, eta)
        distance = math.hypot(predicted_x - src.x, predicted_y - src.y)
        eta = max(1, int(math.ceil(distance / max(speed, 1e-6))))
        predicted_x, predicted_y = self.planet_position_at(tgt, eta)
        angle = math.atan2(predicted_y - src.y, predicted_x - src.x)
        if self.future_crosses_sun(src, tgt, ships):
            self._aiming_angle_cache[key] = 0.0
            return 0.0
        self._aiming_angle_cache[key] = angle
        return angle

    def shot_is_invalid(self, src: PlanetState, tgt: PlanetState, ships: int) -> bool:
        return self.is_comet(tgt) or self.future_crosses_sun(src, tgt, ships)

    def future_crosses_sun(self, src: PlanetState, tgt: PlanetState, ships: int) -> bool:
        key = (src.id, tgt.id, max(int(ships), 1))
        cached = self._future_crosses_sun_cache.get(key)
        if cached is not None:
            return cached
        if self.is_comet(tgt):
            self._future_crosses_sun_cache[key] = True
            return True
        if not self.is_rotating(tgt):
            angle = math.atan2(tgt.y - src.y, tgt.x - src.x)
            start_x = src.x + math.cos(angle) * (src.radius + 0.1)
            start_y = src.y + math.sin(angle) * (src.radius + 0.1)
            result = self.segment_crosses_sun((start_x, start_y), (tgt.x, tgt.y))
            self._future_crosses_sun_cache[key] = result
            return result
        speed = self.ship_speed(ships)
        current_distance = math.hypot(tgt.x - src.x, tgt.y - src.y)
        eta = max(1, int(math.ceil(current_distance / max(speed, 1e-6))))
        predicted_x, predicted_y = self.planet_position_at(tgt, eta)
        angle = math.atan2(predicted_y - src.y, predicted_x - src.x)
        start_x = src.x + math.cos(angle) * (src.radius + 0.1)
        start_y = src.y + math.sin(angle) * (src.radius + 0.1)
        result = self.segment_crosses_sun((start_x, start_y), (predicted_x, predicted_y))
        self._future_crosses_sun_cache[key] = result
        return result

    def arrival_summary(self, planet_id: int, owner_for_friendly: int, horizon: int | None = None) -> PlanetArrivalSummary:
        key = (planet_id, owner_for_friendly, horizon)
        cached = self._arrival_summary_cache.get(key)
        if cached is not None:
            return cached
        entries = self.arrivals_by_planet.get(planet_id, [])
        friendly = 0
        enemy = 0
        nearest_friendly: int | None = None
        nearest_enemy: int | None = None
        for eta, fleet in entries:
            if horizon is not None and eta > horizon:
                break
            if fleet.owner == owner_for_friendly:
                friendly += fleet.ships
                nearest_friendly = eta if nearest_friendly is None else min(nearest_friendly, eta)
            else:
                enemy += fleet.ships
                nearest_enemy = eta if nearest_enemy is None else min(nearest_enemy, eta)
        result = PlanetArrivalSummary(
            friendly_ships=friendly,
            enemy_ships=enemy,
            nearest_friendly_eta=nearest_friendly,
            nearest_enemy_eta=nearest_enemy,
        )
        self._arrival_summary_cache[key] = result
        return result

    def local_planets(self, planet: PlanetState, owner_filter: set[int] | None = None, radius: float | None = None) -> list[PlanetState]:
        radius = self.cfg.local_radius if radius is None else radius
        owner_key = None if owner_filter is None else tuple(sorted(owner_filter))
        key = (planet.id, owner_key, float(radius))
        cached = self._local_planets_cache.get(key)
        if cached is not None:
            return cached
        result = []
        for other in self.state.planets:
            if other.id == planet.id:
                continue
            if owner_filter is not None and other.owner not in owner_filter:
                continue
            if math.hypot(other.x - planet.x, other.y - planet.y) <= radius:
                result.append(other)
        self._local_planets_cache[key] = result
        return result

    def k_nearest_friendly_ships(self, planet: PlanetState, k: int) -> float:
        allies = sorted(
            (
                other
                for other in self.state.planets
                if other.id != planet.id and other.owner == planet.owner and other.owner != -1
            ),
            key=lambda other: (math.hypot(other.x - planet.x, other.y - planet.y), other.id),
        )
        return float(sum(other.ships for other in allies[: max(k, 0)]))

    def has_overwhelming_enemy_fleet(self, planet: PlanetState) -> float:
        target_id = planet.id
        threshold = planet.ships + self.k_nearest_friendly_ships(planet, self.cfg.support_k)
        for _, fleet in self.arrivals_by_planet.get(target_id, []):
            if fleet.owner != planet.owner and fleet.ships >= threshold:
                return 1.0
        return 0.0

    def nearest_planet_distance(self, planet: PlanetState, owner_filter: set[int]) -> float:
        distances = [
            math.hypot(other.x - planet.x, other.y - planet.y)
            for other in self.state.planets
            if other.id != planet.id and other.owner in owner_filter
        ]
        return min(distances) if distances else self.cfg.board_size

    def total_ships_for_owner(self, owner: int) -> float:
        planet_ships = sum(planet.ships for planet in self.state.planets if planet.owner == owner)
        fleet_ships = sum(fleet.ships for fleet in self.state.fleets if fleet.owner == owner)
        return float(planet_ships + fleet_ships)

    def total_production_for_owner(self, owner: int) -> float:
        return float(sum(planet.production for planet in self.state.planets if planet.owner == owner))

    def enemy_owners(self) -> list[int]:
        return sorted({planet.owner for planet in self.state.planets if planet.owner not in {-1, self.state.player}})

    def strongest_enemy_total_ships(self) -> float:
        owners = self.enemy_owners()
        return max((self.total_ships_for_owner(owner) for owner in owners), default=0.0)

    def strongest_enemy_total_production(self) -> float:
        owners = self.enemy_owners()
        return max((self.total_production_for_owner(owner) for owner in owners), default=0.0)

    def my_total_ships(self) -> float:
        return self.total_ships_for_owner(self.state.player)

    def my_total_production(self) -> float:
        return self.total_production_for_owner(self.state.player)

    def normalized_ship_diff(self) -> float:
        scale = max(self.cfg.max_planets * self.cfg.max_ships, 1.0)
        return (self.my_total_ships() - self.strongest_enemy_total_ships()) / scale

    def normalized_production_diff(self) -> float:
        scale = max(self.cfg.max_planets * self.cfg.max_production, 1.0)
        return (self.my_total_production() - self.strongest_enemy_total_production()) / scale

    def shaping_potential(self, reward_cfg: RewardConfig) -> float:
        return reward_cfg.w_ship_diff * self.normalized_ship_diff() + reward_cfg.w_production_diff * self.normalized_production_diff()

    def comet_planets(self) -> list[PlanetState]:
        return [planet for planet in self.state.planets if self.is_comet(planet)]

    def target_expected_ships_at_eta(self, target: PlanetState, eta: int) -> float:
        eta = max(int(eta), 0)
        owner = target.owner
        ships = float(target.ships)
        if owner != -1:
            ships += float(target.production * eta)
        arrivals = self.arrivals_by_planet.get(target.id, [])
        my_arrivals = 0.0
        owner_arrivals = 0.0
        enemy_arrivals = 0.0
        for arrival_eta, fleet in arrivals:
            if arrival_eta > eta:
                break
            if fleet.owner == self.state.player:
                my_arrivals += fleet.ships
            if fleet.owner == owner:
                owner_arrivals += fleet.ships
            elif fleet.owner != owner:
                enemy_arrivals += fleet.ships

        if owner == self.state.player:
            ships += owner_arrivals - enemy_arrivals
        elif owner == -1:
            ships -= my_arrivals
        else:
            ships += owner_arrivals - my_arrivals
        return max(ships, 0.0)

    def target_expected_owner_at_eta(self, target: PlanetState, eta: int) -> int:
        eta = max(int(eta), 0)
        defenders = float(target.ships)
        if target.owner != -1:
            defenders += float(target.production * eta)
        arrivals = self.arrivals_by_planet.get(target.id, [])
        ships_by_owner: dict[int, float] = {}
        for arrival_eta, fleet in arrivals:
            if arrival_eta > eta:
                break
            ships_by_owner[fleet.owner] = ships_by_owner.get(fleet.owner, 0.0) + float(fleet.ships)
        if not ships_by_owner:
            return target.owner

        top_owner, top_ships = max(ships_by_owner.items(), key=lambda item: item[1])
        second_ships = max((ships for owner, ships in ships_by_owner.items() if owner != top_owner), default=0.0)
        surviving_attackers = max(top_ships - second_ships, 0.0)
        if surviving_attackers <= 0.0:
            return target.owner
        if target.owner == top_owner:
            return top_owner
        if surviving_attackers > defenders:
            return top_owner
        return target.owner

    def target_inbound_ships(self, target: PlanetState, owner: int, horizon: int) -> float:
        total = 0.0
        for eta, fleet in self.arrivals_by_planet.get(target.id, []):
            if eta > horizon:
                break
            if fleet.owner == owner:
                total += float(fleet.ships)
        return total

    def local_support_ships(self, target: PlanetState, owner_filter: set[int]) -> float:
        return float(sum(planet.ships for planet in self.local_planets(target, owner_filter)))

    def strategic_value(self, target: PlanetState, src: PlanetState | None = None) -> float:
        dist = math.hypot(target.x - src.x, target.y - src.y) if src is not None else self.cfg.board_size
        ship_term = float(target.ships + 1)
        production_term = float(target.production) / max(ship_term, 1.0)
        distance_term = 1.0 - min(dist / max(self.cfg.board_size, 1.0), 1.0)
        comet_bonus = 0.35 if self.is_comet(target) else 0.0
        enemy_bonus = 0.25 if target.owner not in {-1, self.state.player} else 0.0
        neutral_bonus = 0.15 if target.owner == -1 else 0.0
        return production_term + 0.35 * distance_term + comet_bonus + enemy_bonus + neutral_bonus

    def ship_options(self, src: PlanetState, target: PlanetState) -> tuple[list[int], list[bool]]:
        key = (src.id, target.id)
        cached = self._ship_options_cache.get(key)
        if cached is not None:
            options, mask = cached
            return list(options), list(mask)
        if target.owner == src.owner and target.owner != -1:
            options = [
                max(1, int(math.ceil(src.ships * 0.50))),
                int(math.ceil(src.ships * 0.50)),
                int(math.ceil(src.ships * 0.75)),
                int(math.ceil(src.ships * 1.00)),
            ]
            arrivals = self.arrival_summary(target.id, target.owner, horizon=self.cfg.future_horizon)
            enemy_pressure = max(arrivals.enemy_ships - arrivals.friendly_ships, 0)
            needs_reinforcement = enemy_pressure > 0 or self.has_overwhelming_enemy_fleet(target) > 0.0
            mask = [needs_reinforcement and 0 < ships <= src.ships for ships in options]
            self._ship_options_cache[key] = (tuple(options), tuple(mask))
            return options, mask
        capture_need = self.capture_ship_need(src, target)
        options = [
            capture_need,
            int(math.ceil(src.ships * 0.50)),
            int(math.ceil(src.ships * 0.75)),
            int(math.ceil(src.ships * 1.00)),
        ]
        required = capture_need
        is_attack = target.owner not in {-1, src.owner}
        is_expand = target.owner == -1
        if is_attack or is_expand:
            required = max(required, int(self.cfg.min_attack_ships))
        mask = []
        for ships in options:
            valid = 0 < ships <= src.ships
            if valid and (is_attack or is_expand):
                valid = ships >= required
            mask.append(valid)
        seen: set[int] = set()
        for idx, ships in enumerate(options):
            if not mask[idx]:
                continue
            if ships in seen:
                mask[idx] = False
                continue
            seen.add(ships)
        self._ship_options_cache[key] = (tuple(options), tuple(mask))
        return options, mask

    def capture_ship_need(self, src: PlanetState, target: PlanetState) -> int:
        key = (src.id, target.id)
        cached = self._capture_need_cache.get(key)
        if cached is not None:
            return cached
        if target.owner == src.owner and target.owner != -1:
            result = max(1, int(math.ceil(src.ships * 0.50)))
            self._capture_need_cache[key] = result
            return result

        ships = max(int(target.ships + 1), 1)
        distance = math.hypot(target.x - src.x, target.y - src.y)
        for _ in range(3):
            eta = max(1, int(math.ceil(distance / max(self.ship_speed(ships), 1e-6))))
            expected_ships = self.target_expected_ships_at_eta(target, eta)
            margin = 2.0
            if target.owner != -1:
                margin += min(float(target.production), 6.0)
            ships = max(ships, int(math.ceil(expected_ships + margin)))
        result = max(ships, int(target.ships + 1))
        self._capture_need_cache[key] = result
        return result

    def arrival_flow_vector(self, planet_id: int, horizon: int) -> np.ndarray:
        flow = np.zeros((max(horizon, 0),), dtype=np.float32)
        planet = self.planet_by_id.get(planet_id)
        if planet is None:
            return flow
        for eta, fleet in self.arrivals_by_planet.get(planet_id, []):
            if eta <= 0 or eta > horizon:
                continue
            flow[eta - 1] += float(fleet.ships if fleet.owner == planet.owner else -fleet.ships)
        return flow


def ray_circle_intersection(
    origin: tuple[float, float],
    direction: tuple[float, float],
    center: tuple[float, float],
    radius: float,
) -> float | None:
    ox, oy = origin
    dx, dy = direction
    cx, cy = center
    fx = ox - cx
    fy = oy - cy

    a = dx * dx + dy * dy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    disc_sqrt = math.sqrt(disc)
    t1 = (-b - disc_sqrt) / (2.0 * a)
    t2 = (-b + disc_sqrt) / (2.0 * a)
    positive = [t for t in (t1, t2) if t >= 0.0]
    return min(positive) if positive else None


def point_to_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    segment_len_sq = (start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2
    if segment_len_sq == 0.0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    projection = (
        ((point[0] - start[0]) * (end[0] - start[0]) + (point[1] - start[1]) * (end[1] - start[1]))
        / segment_len_sq
    )
    projection = max(0.0, min(1.0, projection))
    closest_x = start[0] + projection * (end[0] - start[0])
    closest_y = start[1] + projection * (end[1] - start[1])
    return math.hypot(point[0] - closest_x, point[1] - closest_y)
