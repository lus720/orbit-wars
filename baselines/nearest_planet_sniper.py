from __future__ import annotations

import math
from collections import namedtuple
from typing import Any

Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])


def agent(obs: Any, config: Any | None = None) -> list[list[float | int]]:
    del config
    moves: list[list[float | int]] = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    planets = [Planet(*planet) for planet in raw_planets]
    my_planets = [planet for planet in planets if planet.owner == player]
    targets = [planet for planet in planets if planet.owner != player]
    if not targets:
        return moves

    for mine in my_planets:
        nearest = min(
            targets,
            key=lambda target: (math.hypot(mine.x - target.x, mine.y - target.y), target.id),
        )
        ships_needed = max(nearest.ships + 1, 20)
        if mine.ships < ships_needed:
            continue
        angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
        moves.append([mine.id, angle, ships_needed])
    return moves
