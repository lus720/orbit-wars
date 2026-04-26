from __future__ import annotations

import copy
import math
from typing import Any, Callable

from .config import EnvConfig
from .game_types import CometGroupState, FleetState, GameState, PlanetState

BOARD_SIZE = 100.0
SUN_CENTER = (50.0, 50.0)
COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)
MAX_LOG_SHIPS = math.log(1000.0)
_NEXT_FLEET_ID = -1


def _next_fleet_id() -> int:
    global _NEXT_FLEET_ID
    _NEXT_FLEET_ID -= 1
    return _NEXT_FLEET_ID


def _reset_fleet_id_counter() -> None:
    global _NEXT_FLEET_ID
    _NEXT_FLEET_ID = -1


class OrbitSimulator:
    """Lightweight game simulator for Orbit Wars.

    Simulates game steps without requiring the kaggle_environments package.
    All game logic replicates the official Kaggle environment rules.

    Usage:
        sim = OrbitSimulator(cfg)
        sim.load_state(game_state)
        next_state = sim.step(actions)  # actions = [[src_id, angle, ships], ...]
    """

    def __init__(
        self, cfg: EnvConfig, comet_spawn_fn: Callable[[GameState, int], None] | None = None
    ) -> None:
        self.cfg = cfg
        self.state: GameState | None = None
        self._comet_spawn_fn = comet_spawn_fn

    def load_state(self, state: GameState) -> None:
        """Load a game state (deep-copied internally)."""
        self.state = copy.deepcopy(state)
        # Cache initial planet positions for rotation calculations
        self._initial_positions: dict[int, tuple[float, float]] = {}
        for p in state.initial_planets:
            self._initial_positions[p.id] = (p.x, p.y)
        # Build planet lookup for planet_position_at
        self._planet_by_id: dict[int, PlanetState] = {p.id: p for p in state.planets}

    def step(self, actions: list[list[float | int]]) -> GameState:
        """Advance one step. actions = [[src_id, angle, ships], ...]

        Returns the new GameState after this step.
        """
        if self.state is None:
            raise RuntimeError("Call load_state() before step().")
        state = self.state
        cfg = self.cfg
        # Set to True to indicate this is a simulation
        state.step += 1
        current_step = state.step

        # --- 1. Remove expired comets ---
        self._remove_expired_comets()

        # --- 2. Spawn new comets ---
        self._spawn_comets(current_step)

        # --- 3. Process actions: create fleets ---
        for src_id, angle, ships in actions:
            src_id = int(src_id)
            ships = int(ships)
            if ships <= 0:
                continue
            src = self._planet_by_id.get(src_id)
            if src is None or src.owner != state.player:
                continue
            if src.ships < ships:
                continue
            src.ships -= ships
            start_x = src.x + math.cos(float(angle)) * (src.radius + 0.1)
            start_y = src.y + math.sin(float(angle)) * (src.radius + 0.1)
            fleet = FleetState(
                id=_next_fleet_id(),
                owner=state.player,
                x=start_x,
                y=start_y,
                angle=float(angle),
                from_planet_id=src_id,
                ships=ships,
            )
            state.fleets.append(fleet)

        # --- 4. Ship production on all owned planets + comets ---
        for planet in state.planets:
            if planet.owner not in (-1,):
                planet.ships += planet.production

        # --- 5. Move fleets and detect collisions ---
        surviving_fleets: list[FleetState] = []
        combat_queue: dict[int, list[FleetState]] = {}

        for fleet in state.fleets:
            speed = self._ship_speed(fleet.ships)
            new_x = fleet.x + math.cos(fleet.angle) * speed
            new_y = fleet.y + math.sin(fleet.angle) * speed

            # Check out of bounds
            if not (0.0 <= new_x <= BOARD_SIZE and 0.0 <= new_y <= BOARD_SIZE):
                continue

            # Check segment crosses sun
            if self._segment_crosses_sun((fleet.x, fleet.y), (new_x, new_y), cfg.sun_radius):
                continue

            # Check collision with any planet
            hit_planet = self._find_planet_collision(
                (fleet.x, fleet.y), (new_x, new_y), fleet.owner, speed
            )
            if hit_planet is not None:
                fleet.x = hit_planet.x
                fleet.y = hit_planet.y
                combat_queue.setdefault(hit_planet.id, []).append(fleet)
                continue

            fleet.x = new_x
            fleet.y = new_y
            surviving_fleets.append(fleet)

        state.fleets = surviving_fleets

        # --- 6. Update rotating planet positions ---
        self._update_planet_positions()

        # --- 7. Advance comet trajectories and check collisions ---
        self._advance_comets()

        # --- 8. Resolve combat ---
        self._resolve_combat(combat_queue)

        # Recompute the planet_by_id cache for next step
        self._planet_by_id = {p.id: p for p in state.planets}

        return state

    def simulate(
        self, state: GameState, actions: list[list[float | int]], horizon: int
    ) -> GameState:
        """Simulate taking actions and running for `horizon` steps.

        Returns the GameState after all steps.
        """
        self.load_state(state)
        for _ in range(horizon):
            self.step(actions if _ == 0 else [])
        return self.state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ship_speed(self, ships: int) -> float:
        ships = max(int(ships), 1)
        if ships <= 1:
            return 1.0
        ratio = math.log(float(ships)) / MAX_LOG_SHIPS
        return 1.0 + (self.cfg.ship_speed - 1.0) * max(ratio, 0.0) ** 1.5

    def _is_rotating(self, planet: PlanetState) -> bool:
        dx = planet.x - SUN_CENTER[0]
        dy = planet.y - SUN_CENTER[1]
        orbital_radius = math.hypot(dx, dy)
        return orbital_radius + planet.radius < 50.0

    def _segment_crosses_sun(
        self, start: tuple[float, float], end: tuple[float, float], sun_radius: float
    ) -> bool:
        return point_to_segment_distance(start, end, SUN_CENTER) < sun_radius

    def _find_planet_collision(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        fleet_owner: int,
        speed: float,
    ) -> PlanetState | None:
        best_t = float("inf")
        best_planet: PlanetState | None = None
        for planet in self.state.planets:
            # segment is from start to end, so t in [0, 1] means within segment
            t = ray_circle_intersection(start, end, (planet.x, planet.y), planet.radius)
            if t is not None and 0.0 <= t <= 1.0 and t < best_t:
                best_t = t
                best_planet = planet
        return best_planet

    def _remove_expired_comets(self) -> None:
        """Remove comets that have reached the end of their path."""
        state = self.state
        expired_ids: set[int] = set()
        for group in list(state.comet_groups):
            path_len = len(group.paths[0]) if group.paths else 0
            if path_len > 0 and group.path_index >= path_len - 1:
                expired_ids.update(group.planet_ids)
                state.comet_groups.remove(group)

        if expired_ids:
            state.planets = [p for p in state.planets if p.id not in expired_ids]
            state.comet_planet_ids = state.comet_planet_ids - expired_ids
            for pid in expired_ids:
                self._planet_by_id.pop(pid, None)

    def _spawn_comets(self, current_step: int) -> None:
        """Spawn comets at predefined steps."""
        if current_step not in COMET_SPAWN_STEPS:
            return
        if self._comet_spawn_fn is not None:
            self._comet_spawn_fn(self.state, current_step)

    def _update_planet_positions(self) -> None:
        """Update positions of rotating planets.

        Note: The Kaggle environment applies the first rotation between
        step 1 and step 2. So at step S, planets have been rotated
        (S - 1) times from their initial positions.
        """
        state = self.state
        num_rotations = max(0, state.step - 1)
        for planet in state.planets:
            if not self._is_rotating(planet):
                continue
            initial = self._initial_positions.get(planet.id)
            if initial is None:
                continue
            dx = initial[0] - SUN_CENTER[0]
            dy = initial[1] - SUN_CENTER[1]
            orbital_radius = math.hypot(dx, dy)
            theta0 = math.atan2(dy, dx)
            theta = theta0 + state.angular_velocity * num_rotations
            planet.x = SUN_CENTER[0] + orbital_radius * math.cos(theta)
            planet.y = SUN_CENTER[1] + orbital_radius * math.sin(theta)

    def _advance_comets(self) -> None:
        """Advance comet positions along their paths and check collisions."""
        state = self.state
        for group in state.comet_groups:
            group.path_index += 1
            for idx, planet_id in enumerate(group.planet_ids):
                planet = self._planet_by_id.get(planet_id)
                if planet is None:
                    continue
                if idx < len(group.paths) and group.path_index < len(group.paths[idx]):
                    planet.x = group.paths[idx][group.path_index][0]
                    planet.y = group.paths[idx][group.path_index][1]

    def _resolve_combat(self, combat_queue: dict[int, list[FleetState]]) -> None:
        """Resolve all queued combats."""
        state = self.state
        for planet_id, arriving_fleets in combat_queue.items():
            planet = self._planet_by_id.get(planet_id)
            if planet is None:
                continue

            # Group arriving fleets by owner
            ships_by_owner: dict[int, int] = {}
            for fleet in arriving_fleets:
                ships_by_owner[fleet.owner] = ships_by_owner.get(fleet.owner, 0) + fleet.ships

            if not ships_by_owner:
                continue

            # Sort owners by total ships descending
            ranked = sorted(ships_by_owner.items(), key=lambda x: -x[1])

            # Largest vs second-largest fight
            if len(ranked) >= 2:
                top_owner, top_ships = ranked[0]
                second_ships = ranked[1][1]
                if top_ships > second_ships:
                    survivors = top_ships - second_ships
                    winner_owner = top_owner
                else:
                    # Tie or second larger — all attackers destroyed
                    continue
            else:
                survivors = ranked[0][1]
                winner_owner = ranked[0][0]

            # Survivor vs planet garrison
            garrison = planet.ships
            if winner_owner == planet.owner:
                planet.ships = garrison + survivors
            elif survivors > garrison:
                planet.ships = survivors - garrison
                planet.owner = winner_owner
            else:
                planet.ships = garrison - survivors


def pop_relevant_fleets(
    state: GameState, planet_positions: dict[int, tuple[float, float]]
) -> list[FleetState]:
    """Filter fleets that are near their target planets (used by search)."""
    return state.fleets


# ------------------------------------------------------------------
# Geometric helpers (ported from world_model)
# ------------------------------------------------------------------


def ray_circle_intersection(
    start: tuple[float, float],
    end: tuple[float, float],
    center: tuple[float, float],
    radius: float,
) -> float | None:
    """Find the distance along segment (start→end) where it first hits the circle.
    If start is inside the circle, the intersection beyond start is returned.
    Returns None if no intersection.
    """
    sx, sy = start
    ex, ey = end
    cx, cy = center
    dx = ex - sx
    dy = ey - sy
    fx = sx - cx
    fy = sy - cy

    a = dx * dx + dy * dy
    if a == 0.0:
        return None
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    disc_sqrt = math.sqrt(disc)
    t1 = (-b - disc_sqrt) / (2.0 * a)
    t2 = (-b + disc_sqrt) / (2.0 * a)
    # Normalize t to [0, 1] range (fraction along segment)
    positive = [t for t in (t1, t2) if t >= 0.0]
    return min(positive) if positive else None


def point_to_segment_distance(
    start: tuple[float, float], end: tuple[float, float], point: tuple[float, float]
) -> float:
    """Minimum distance from point to segment (start, end)."""
    sx, sy = start
    ex, ey = end
    px, py = point
    seg_len_sq = (sx - ex) ** 2 + (sy - ey) ** 2
    if seg_len_sq == 0.0:
        return math.hypot(px - sx, py - sy)
    proj = ((px - sx) * (ex - sx) + (py - sy) * (ey - sy)) / seg_len_sq
    proj = max(0.0, min(1.0, proj))
    closest_x = sx + proj * (ex - sx)
    closest_y = sy + proj * (ey - sy)
    return math.hypot(px - closest_x, py - closest_y)
