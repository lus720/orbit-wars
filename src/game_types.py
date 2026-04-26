
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PlanetState:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int


@dataclass(slots=True)
class FleetState:
    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: int


@dataclass(slots=True)
class CometGroupState:
    planet_ids: tuple[int, ...]
    paths: tuple[tuple[tuple[float, float], ...], ...]
    path_index: int


@dataclass(slots=True)
class GameState:
    step: int
    player: int
    planets: list[PlanetState]
    fleets: list[FleetState]
    angular_velocity: float
    initial_planets: list[PlanetState]
    comet_groups: list[CometGroupState]
    comet_planet_ids: frozenset[int]


def parse_observation(observation: Any) -> GameState:
    def obs_get(key: str, default: Any) -> Any:
        if isinstance(observation, dict):
            return observation.get(key, default)
        return getattr(observation, key, default)

    planets = [
        PlanetState(
            id=int(row[0]),
            owner=int(row[1]),
            x=float(row[2]),
            y=float(row[3]),
            radius=float(row[4]),
            ships=int(row[5]),
            production=int(row[6]),
        )
        for row in obs_get("planets", [])
    ]
    fleets = [
        FleetState(
            id=int(row[0]),
            owner=int(row[1]),
            x=float(row[2]),
            y=float(row[3]),
            angle=float(row[4]),
            from_planet_id=int(row[5]),
            ships=int(row[6]),
        )
        for row in obs_get("fleets", [])
    ]
    initial_planets = [
        PlanetState(
            id=int(row[0]),
            owner=int(row[1]),
            x=float(row[2]),
            y=float(row[3]),
            radius=float(row[4]),
            ships=int(row[5]),
            production=int(row[6]),
        )
        for row in obs_get("initial_planets", [])
    ]
    comet_groups = [
        CometGroupState(
            planet_ids=tuple(int(planet_id) for planet_id in entry.get("planet_ids", [])),
            paths=tuple(
                tuple((float(point[0]), float(point[1])) for point in path)
                for path in entry.get("paths", [])
            ),
            path_index=int(entry.get("path_index", 0)),
        )
        for entry in obs_get("comets", [])
    ]
    return GameState(
        step=int(obs_get("step", 0)),
        player=int(obs_get("player", 0)),
        planets=planets,
        fleets=fleets,
        angular_velocity=float(obs_get("angular_velocity", 0.0)),
        initial_planets=initial_planets,
        comet_groups=comet_groups,
        comet_planet_ids=frozenset(int(planet_id) for planet_id in obs_get("comet_planet_ids", [])),
    )
