import math
from itertools import combinations
from collections import defaultdict, namedtuple


Planet = namedtuple(
    "Planet", ["id", "owner", "x", "y", "radius", "ships", "production"]
)
Fleet = namedtuple(
    "Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"]
)


BOARD = 100.0
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
ROTATION_LIMIT = 50.0
MAX_SPEED = 6.0
TOTAL_STEPS = 500
ROUTE_LOOKAHEAD = 64
DEFENSE_HORIZON = 22
FORECAST_HORIZON = 32
LAUNCH_CLEARANCE = 0.1
FRONTIER_TRANSFER_RATIO = 0.58


def _read(mapping, key, default=None):
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    ratio = math.log(max(ships, 1)) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio**1.5)


def orbital_radius(planet):
    return dist(planet.x, planet.y, CENTER_X, CENTER_Y)


def is_rotating(planet):
    return orbital_radius(planet) + planet.radius < ROTATION_LIMIT


def launch_point(source, angle):
    clearance = source.radius + LAUNCH_CLEARANCE
    return (
        source.x + math.cos(angle) * clearance,
        source.y + math.sin(angle) * clearance,
    )


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    span = dx * dx + dy * dy
    if span <= 1e-9:
        return dist(px, py, x1, y1)
    t = ((px - x1) * dx + (py - y1) * dy) / span
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return dist(px, py, proj_x, proj_y)


def segment_hits_sun(x1, y1, x2, y2):
    return point_to_segment_distance(CENTER_X, CENTER_Y, x1, y1, x2, y2) < SUN_RADIUS


def resolve_battle(counts):
    alive = [(owner, ships) for owner, ships in counts.items() if ships > 0]
    if not alive:
        return -1, 0
    alive.sort(key=lambda item: item[1], reverse=True)
    if len(alive) == 1:
        return alive[0][0], alive[0][1]
    if alive[0][1] == alive[1][1]:
        return -1, 0
    return alive[0][0], alive[0][1] - alive[1][1]


def direct_route(source, tx, ty, tradius, ships):
    angle = math.atan2(ty - source.y, tx - source.x)
    start_x, start_y = launch_point(source, angle)
    travel = max(0.0, dist(start_x, start_y, tx, ty) - tradius)
    end_x = start_x + math.cos(angle) * travel
    end_y = start_y + math.sin(angle) * travel
    if segment_hits_sun(start_x, start_y, end_x, end_y):
        return None
    eta = max(1, int(math.ceil(travel / fleet_speed(ships))))
    return angle, eta


def route_intersects_planet(source, tx, ty, target_radius, blocker):
    start_x, start_y = launch_point(source, math.atan2(ty - source.y, tx - source.x))
    travel = max(0.0, dist(start_x, start_y, tx, ty) - target_radius)
    end_x = start_x + (tx - start_x) / max(dist(start_x, start_y, tx, ty), 1e-9) * travel
    end_y = start_y + (ty - start_y) / max(dist(start_x, start_y, tx, ty), 1e-9) * travel
    return point_to_segment_distance(blocker.x, blocker.y, start_x, start_y, end_x, end_y) < blocker.radius


def segment_circle_hit_distance(x1, y1, x2, y2, cx, cy, radius):
    dx = x2 - x1
    dy = y2 - y1
    span = math.hypot(dx, dy)
    if span <= 1e-9:
        return None
    dir_x = dx / span
    dir_y = dy / span
    rel_x = cx - x1
    rel_y = cy - y1
    proj = rel_x * dir_x + rel_y * dir_y
    if proj < 0.0 or proj > span:
        return None
    perp_sq = rel_x * rel_x + rel_y * rel_y - proj * proj
    radius_sq = radius * radius
    if perp_sq >= radius_sq:
        return None
    return max(0.0, proj - math.sqrt(max(0.0, radius_sq - perp_sq)))


def fleet_target_planet(fleet, planets):
    best = None
    best_eta = None
    dx = math.cos(fleet.angle)
    dy = math.sin(fleet.angle)
    speed = fleet_speed(fleet.ships)
    for planet in planets:
        rel_x = planet.x - fleet.x
        rel_y = planet.y - fleet.y
        proj = rel_x * dx + rel_y * dy
        if proj < 0:
            continue
        perp_sq = rel_x * rel_x + rel_y * rel_y - proj * proj
        radius_sq = planet.radius * planet.radius
        if perp_sq >= radius_sq:
            continue
        hit = max(0.0, proj - math.sqrt(max(0.0, radius_sq - perp_sq)))
        eta = max(1, int(math.ceil(hit / speed)))
        if best_eta is None or eta < best_eta:
            best_eta = eta
            best = planet
    return best, best_eta


class GameState:
    def __init__(self, obs):
        self.player = _read(obs, "player", 0)
        self.step = _read(obs, "step", 0) or 0
        self.remaining_steps = TOTAL_STEPS - self.step
        self.angular_velocity = _read(obs, "angular_velocity", 0.0) or 0.0
        self.planets = [Planet(*item) for item in (_read(obs, "planets", []) or [])]
        self.fleets = [Fleet(*item) for item in (_read(obs, "fleets", []) or [])]
        self.initial_by_id = {
            p.id: p for p in [Planet(*item) for item in (_read(obs, "initial_planets", []) or [])]
        }
        self.planet_by_id = {planet.id: planet for planet in self.planets}
        self.comet_ids = set(_read(obs, "comet_planet_ids", []) or [])
        self.comet_positions = self._build_comet_positions(_read(obs, "comets", []) or [])

        self.my_planets = [p for p in self.planets if p.owner == self.player]
        self.neutral_planets = [p for p in self.planets if p.owner == -1]
        self.enemy_planets = [p for p in self.planets if p.owner not in (-1, self.player)]
        self.enemy_ids = sorted({p.owner for p in self.enemy_planets} | {f.owner for f in self.fleets if f.owner != self.player})
        self.arrivals = self._build_arrivals()
        self.frontier_distance = {
            planet.id: self.distance_to_enemy_front(planet) for planet in self.my_planets
        }
        self.power = defaultdict(int)
        self.production = defaultdict(int)
        for planet in self.planets:
            if planet.owner != -1:
                self.power[planet.owner] += int(planet.ships)
                self.production[planet.owner] += int(planet.production)
        for fleet in self.fleets:
            self.power[fleet.owner] += int(fleet.ships)
        self.weakest_enemy = self._pick_weakest_enemy()
        self.strongest_enemy = self._pick_strongest_enemy()
        self.route_cache = {}
        self.reaction_cache = {}
        self.base_timeline = {}
        self.planned = defaultdict(list)

    def _build_comet_positions(self, comets):
        positions = {}
        for comet_group in comets:
            ids = comet_group.get("planet_ids", []) or comet_group.get("comet_planet_ids", [])
            paths = comet_group.get("paths", [])
            path_index = comet_group.get("path_index", 0)
            for planet_id, path in zip(ids, paths):
                positions[planet_id] = (path, path_index)
        return positions

    def _build_arrivals(self):
        arrivals = defaultdict(list)
        for fleet in self.fleets:
            target, eta = fleet_target_planet(fleet, self.planets)
            if target is None or eta is None or eta > FORECAST_HORIZON:
                continue
            arrivals[target.id].append((eta, fleet.owner, int(fleet.ships)))
        return arrivals

    def _pick_weakest_enemy(self):
        if not self.enemy_ids:
            return None
        return min(
            self.enemy_ids,
            key=lambda owner: (
                self.power.get(owner, 0) + 5 * self.production.get(owner, 0),
                self.power.get(owner, 0),
                owner,
            ),
        )

    def _pick_strongest_enemy(self):
        if not self.enemy_ids:
            return None
        return max(
            self.enemy_ids,
            key=lambda owner: (
                self.production.get(owner, 0),
                self.power.get(owner, 0),
                -owner,
            ),
        )

    def runaway_enemy(self):
        if self.strongest_enemy is None:
            return None
        leader = self.strongest_enemy
        my_prod = self.production.get(self.player, 0)
        my_power = self.power.get(self.player, 0)
        if self.production.get(leader, 0) >= my_prod + 4:
            return leader
        if self.power.get(leader, 0) >= my_power + 45:
            return leader
        return None

    def threatening_enemies(self):
        my_prod = self.production.get(self.player, 0)
        my_power = self.power.get(self.player, 0)
        threats = []
        for owner in self.enemy_ids:
            if self.production.get(owner, 0) >= my_prod + 2:
                threats.append(owner)
                continue
            if self.power.get(owner, 0) >= my_power + 35:
                threats.append(owner)
        if threats:
            return sorted(set(threats))
        if self.strongest_enemy is not None:
            return [self.strongest_enemy]
        return []

    def predict_position(self, target_id, turns):
        planet = self.planet_by_id[target_id]
        if target_id in self.comet_positions:
            path, path_index = self.comet_positions[target_id]
            if not path:
                return planet.x, planet.y
            idx = min(len(path) - 1, max(0, path_index + turns))
            px, py = path[idx]
            return px, py
        initial = self.initial_by_id.get(target_id)
        if initial is None or not is_rotating(initial):
            return planet.x, planet.y
        radius = orbital_radius(initial)
        angle = math.atan2(initial.y - CENTER_Y, initial.x - CENTER_X) + self.angular_velocity * turns
        return CENTER_X + radius * math.cos(angle), CENTER_Y + radius * math.sin(angle)

    def route(self, source_id, target_id, ships):
        ships = int(max(1, ships))
        key = (source_id, target_id, ships)
        if key in self.route_cache:
            return self.route_cache[key]
        source = self.planet_by_id[source_id]
        target = self.planet_by_id[target_id]

        initial_target = self.initial_by_id.get(target_id, target)
        moving_target = target_id in self.comet_positions or is_rotating(initial_target)
        direct_now = direct_route(source, target.x, target.y, target.radius, ships)
        if direct_now is None:
            self.route_cache[key] = None
            return None
        if not moving_target:
            tx, ty = target.x, target.y
            blocked = False
            for blocker in self.planets:
                if blocker.id in (source_id, target_id):
                    continue
                if route_intersects_planet(source, tx, ty, target.radius, blocker):
                    blocked = True
                    break
            best = None if blocked else direct_now
            self.route_cache[key] = best
            return best

        base_eta = direct_now[1]
        guess_from = max(1, base_eta - (4 if moving_target else 1))
        guess_to = min(ROUTE_LOOKAHEAD, base_eta + (12 if moving_target else 3))

        best = None
        best_gap = None
        for guess_eta in range(guess_from, guess_to + 1):
            tx, ty = self.predict_position(target_id, max(0, guess_eta - 1))
            angle = math.atan2(ty - source.y, tx - source.x)
            hit_id, eta = self._simulate_launch(source_id, target_id, angle, ships, guess_eta + 4)
            if hit_id != target_id or eta is None:
                continue
            gap = abs(eta - guess_eta)
            if eta > ROUTE_LOOKAHEAD:
                continue
            if best is None or gap < best_gap or (gap == best_gap and eta < best[1]):
                best = (angle, eta)
                best_gap = gap
                if gap == 0:
                    break

        self.route_cache[key] = best
        return best

    def _simulate_launch(self, source_id, target_id, angle, ships, horizon):
        source = self.planet_by_id[source_id]
        x, y = launch_point(source, angle)
        speed = fleet_speed(ships)
        dir_x = math.cos(angle)
        dir_y = math.sin(angle)
        for turn in range(1, min(ROUTE_LOOKAHEAD, horizon) + 1):
            next_x = x + dir_x * speed
            next_y = y + dir_y * speed
            if not (0.0 <= next_x <= BOARD and 0.0 <= next_y <= BOARD):
                return None, None
            if segment_hits_sun(x, y, next_x, next_y):
                return None, None

            first_hit = None
            first_dist = None
            for planet in self.planets:
                if planet.id == source_id:
                    continue
                px, py = self.predict_position(planet.id, turn - 1)
                hit_dist = segment_circle_hit_distance(
                    x,
                    y,
                    next_x,
                    next_y,
                    px,
                    py,
                    planet.radius,
                )
                if hit_dist is None:
                    continue
                if first_dist is None or hit_dist < first_dist:
                    first_hit = planet.id
                    first_dist = hit_dist

            if first_hit is not None:
                return first_hit, turn
            x, y = next_x, next_y
        return None, None

    def ensure_timeline(self, planet_id, horizon):
        current = self.base_timeline.get(planet_id)
        if current is not None and len(current) >= horizon:
            return current
        timeline = self.simulate_planet(planet_id, horizon)
        self.base_timeline[planet_id] = timeline
        return timeline

    def simulate_planet(self, planet_id, horizon, extra=None, ship_offset=0):
        planet = self.planet_by_id[planet_id]
        owner = planet.owner
        ships = max(0, int(planet.ships) - int(ship_offset))
        events = defaultdict(lambda: defaultdict(int))
        for eta, owner_id, amount in self.arrivals.get(planet_id, []):
            if eta <= horizon:
                events[eta][owner_id] += int(amount)
        for eta, owner_id, amount in self.planned.get(planet_id, []):
            if eta <= horizon:
                events[eta][owner_id] += int(amount)
        if extra:
            for eta, owner_id, amount in extra:
                if eta <= horizon:
                    events[int(eta)][owner_id] += int(amount)

        result = []
        for turn in range(1, horizon + 1):
            if owner != -1 and ships > 0:
                ships += int(planet.production)
            counts = defaultdict(int)
            if ships > 0 or owner != -1:
                counts[owner] += ships
            for owner_id, amount in events.get(turn, {}).items():
                counts[owner_id] += amount
            owner, ships = resolve_battle(counts)
            result.append((turn, owner, ships))
        return result

    def projected_state(self, planet_id, turn, extra=None, ship_offset=0):
        if not extra and ship_offset == 0:
            timeline = self.ensure_timeline(planet_id, turn)
            return timeline[turn - 1][1], timeline[turn - 1][2]
        timeline = self.simulate_planet(planet_id, turn, extra=extra, ship_offset=ship_offset)
        return timeline[turn - 1][1], timeline[turn - 1][2]

    def loss_turn(self, planet_id, ship_offset=0):
        for turn, owner, _ships in self.simulate_planet(planet_id, DEFENSE_HORIZON, ship_offset=ship_offset):
            if owner != self.player:
                return turn
        return None

    def safe_send(self, planet_id):
        planet = self.planet_by_id[planet_id]
        if planet.owner != self.player or planet.ships <= 1:
            return 0
        low = 0
        high = int(planet.ships) - 1
        while low < high:
            mid = (low + high + 1) // 2
            if self.loss_turn(planet_id, ship_offset=mid) is None:
                low = mid
            else:
                high = mid - 1
        return low

    def ships_needed_to_own(self, target_id, turn, upper_bound, owner_id=None, hold_turns=0):
        owner_id = self.player if owner_id is None else owner_id
        current_owner, _current_ships = self.projected_state(target_id, turn)
        if current_owner == owner_id and hold_turns <= 0:
            return 0

        def succeeds(amount):
            extra = [(turn, owner_id, amount)]
            horizon = turn + hold_turns
            timeline = self.simulate_planet(target_id, horizon, extra=extra)
            start = turn - 1
            stop = horizon - 1
            for idx in range(start, stop + 1):
                if timeline[idx][1] != owner_id:
                    return False
            return True

        low = 1
        high = max(1, int(upper_bound))
        if not succeeds(high):
            return high + 1
        while low < high:
            mid = (low + high) // 2
            if succeeds(mid):
                high = mid
            else:
                low = mid + 1
        return low

    def fastest_hostile_eta(self, target_id):
        cached = self.reaction_cache.get(("hostile", target_id))
        if cached is not None:
            return cached
        best = 10**9
        for source in self.enemy_planets:
            route = self.route(source.id, target_id, max(1, int(source.ships)))
            if route is None:
                continue
            best = min(best, route[1])
        self.reaction_cache[("hostile", target_id)] = best
        return best

    def fastest_friendly_eta(self, target_id):
        cached = self.reaction_cache.get(("friendly", target_id))
        if cached is not None:
            return cached
        best = 10**9
        for source in self.my_planets:
            route = self.route(source.id, target_id, max(1, int(source.ships)))
            if route is None:
                continue
            best = min(best, route[1])
        self.reaction_cache[("friendly", target_id)] = best
        return best

    def distance_to_enemy_front(self, planet):
        if not self.enemy_planets:
            return 10**9
        return min(dist(planet.x, planet.y, enemy.x, enemy.y) for enemy in self.enemy_planets)

    def exposed_enemy_score(self, target):
        my_eta = self.fastest_friendly_eta(target.id)
        enemy_eta = 10**9
        for planet in self.enemy_planets:
            if planet.owner != target.owner or planet.id == target.id:
                continue
            route = self.route(planet.id, target.id, max(1, int(planet.ships)))
            if route is None:
                continue
            enemy_eta = min(enemy_eta, route[1])
        if enemy_eta == 10**9:
            return 2.0
        gap = enemy_eta - my_eta
        if gap >= 4:
            return 1.8
        if gap >= 2:
            return 1.45
        if gap >= 0:
            return 1.15
        return 0.85


def defense_candidates(state, surplus):
    plans = []
    for target in state.my_planets:
        fall_turn = state.loss_turn(target.id)
        if fall_turn is None:
            continue
        best = None
        for source in state.my_planets:
            if source.id == target.id:
                continue
            available = surplus.get(source.id, 0)
            if available <= 0:
                continue
            route = state.route(source.id, target.id, available)
            if route is None:
                continue
            angle, eta = route
            if eta > fall_turn:
                continue
            need = state.ships_needed_to_own(
                target.id,
                fall_turn,
                available,
                hold_turns=min(6, FORECAST_HORIZON - fall_turn),
            )
            if need <= 0 or need > available:
                continue
            score = 10000.0 - 200.0 * fall_turn - 3.0 * need - eta
            if best is None or score > best["score"]:
                best = {
                    "kind": "defense",
                    "source_id": source.id,
                    "target_id": target.id,
                    "angle": angle,
                    "eta": eta,
                    "ships": need,
                    "score": score,
                }
        if best is not None:
            plans.append(best)
    plans.sort(key=lambda item: item["score"], reverse=True)
    return plans


def contested_takeovers(state, surplus):
    plans = []
    for target in state.planets:
        if target.owner == state.player:
            continue
        base = state.simulate_planet(target.id, min(16, FORECAST_HORIZON))
        for turn, owner, ships in base:
            if turn < 4 or owner == state.player:
                continue
            if ships > 10:
                continue
            multi_enemy = len({owner_id for eta, owner_id, _ships in state.arrivals.get(target.id, []) if eta <= turn and owner_id != state.player}) >= 2
            if not multi_enemy and target.owner == -1:
                continue
            for source in state.my_planets:
                available = surplus.get(source.id, 0)
                if available <= 0:
                    continue
                seed = max(1, min(available, ships + 8))
                route = state.route(source.id, target.id, seed)
                if route is None:
                    continue
                angle, eta = route
                if eta < max(1, turn - 1) or eta > turn + 2:
                    continue
                need = state.ships_needed_to_own(
                    target.id,
                    eta,
                    available,
                    hold_turns=3,
                )
                if need <= 0 or need > available:
                    continue
                score = (
                    140.0
                    + 30.0 * target.production
                    + 6.0 * min(10, 10 - ships)
                    - 2.0 * need
                    - 1.5 * eta
                )
                if target.owner == state.weakest_enemy or owner == state.weakest_enemy:
                    score += 18.0
                plans.append(
                    {
                        "kind": "takeover",
                        "source_id": source.id,
                        "target_id": target.id,
                        "angle": angle,
                        "eta": eta,
                        "ships": need,
                        "score": score,
                    }
                )
                break
    plans.sort(key=lambda item: item["score"], reverse=True)
    return plans


def opportunistic_enemy_takeovers(state, surplus):
    plans = []
    runaway = state.runaway_enemy()
    for target in state.enemy_planets:
        base = state.simulate_planet(target.id, min(18, FORECAST_HORIZON))
        home_owner = target.owner
        for turn, owner, ships in base:
            if turn < 4 or owner == state.player:
                continue
            if owner == home_owner and ships > max(8, 2 * target.production):
                continue
            for source in state.my_planets:
                available = surplus.get(source.id, 0)
                if available <= 0:
                    continue
                probe = max(1, min(available, ships + 8))
                route = state.route(source.id, target.id, probe)
                if route is None:
                    continue
                angle, eta = route
                if eta < max(1, turn - 1) or eta > turn + 2:
                    continue
                need = state.ships_needed_to_own(
                    target.id,
                    eta,
                    available,
                    hold_turns=3,
                )
                if need <= 0 or need > available:
                    continue
                score = (
                    180.0
                    + 36.0 * target.production
                    + 7.0 * max(0, 10 - ships)
                    - 2.1 * need
                    - 1.5 * eta
                )
                if target.owner == state.weakest_enemy or owner == state.weakest_enemy:
                    score += 24.0
                if target.owner == runaway or home_owner == runaway:
                    score += 28.0 + 5.0 * target.production
                plans.append(
                    {
                        "kind": "enemy_takeover",
                        "source_id": source.id,
                        "target_id": target.id,
                        "angle": angle,
                        "eta": eta,
                        "ships": need,
                        "score": score,
                    }
                )
                break
    plans.sort(key=lambda item: item["score"], reverse=True)
    return plans


def attack_candidates(state, surplus):
    plans = []
    runaway = state.runaway_enemy()
    threats = set(state.threatening_enemies())
    for source in state.my_planets:
        available = surplus.get(source.id, 0)
        if available <= 0:
            continue

        nearby = sorted(
            [p for p in state.planets if p.owner != state.player and p.id != source.id],
            key=lambda target: (
                dist(source.x, source.y, target.x, target.y),
                -target.production,
                target.id,
            ),
        )[:14]

        candidate_map = {target.id: target for target in nearby}
        if runaway is not None:
            for target in sorted(
                [p for p in state.enemy_planets if p.owner == runaway],
                key=lambda planet: (-planet.production, planet.ships, planet.id),
            )[:6]:
                candidate_map[target.id] = target
        if state.step >= 24:
            for target in sorted(
                [p for p in state.neutral_planets if p.production >= 3],
                key=lambda planet: (-planet.production, planet.ships, planet.id),
            )[:6]:
                candidate_map[target.id] = target

        for target in candidate_map.values():
            probe = max(1, min(available, int(target.ships) + max(2, int(target.production))))
            route = state.route(source.id, target.id, probe)
            if route is None:
                continue
            angle, eta = route
            if eta >= state.remaining_steps - 3:
                continue

            if target.owner == -1:
                if len(state.enemy_ids) >= 3 and state.production.get(state.player, 0) < 8:
                    if target.production <= 1 and state.step > 10:
                        continue
                    if len(state.my_planets) <= 2 and eta > 14:
                        continue
                    if eta > 18 and target.production < 5:
                        continue
                if state.step > 35 and target.production <= 1:
                    continue
                hostile_eta = state.fastest_hostile_eta(target.id)
                safety = hostile_eta - eta
                if safety < -1:
                    continue
                need = state.ships_needed_to_own(target.id, eta, available, hold_turns=3)
                if need <= 0 or need > available:
                    continue
                if (
                    len(state.enemy_ids) >= 3
                    and state.production.get(state.player, 0) < 8
                    and len(state.my_planets) <= 2
                    and eta > 10
                    and need < 3
                ):
                    continue
                value = target.production * max(1, state.remaining_steps - eta)
                value += 3.5 * target.production
                if orbital_radius(target) + target.radius >= ROTATION_LIMIT:
                    value *= 1.18
                if safety >= 2:
                    value *= 1.20
                elif safety == 1:
                    value *= 1.08
                elif safety < 0:
                    value *= 0.78
                if state.remaining_steps <= 180 and threats:
                    value *= 0.86
                score = value / max(need + 1.05 * eta, 1)
            else:
                need = state.ships_needed_to_own(target.id, eta, available, hold_turns=2)
                if need <= 0 or need > available:
                    continue
                if (
                    len(state.enemy_ids) >= 3
                    and state.production.get(state.player, 0) < 8
                    and len(state.my_planets) <= 2
                    and eta > 12
                    and need < 3
                ):
                    continue
                value = target.production * max(1, state.remaining_steps - eta)
                value += 0.45 * target.ships
                value *= state.exposed_enemy_score(target)
                if target.owner == state.weakest_enemy:
                    value *= 1.28
                if target.owner == runaway:
                    value *= 1.32
                    if target.production >= 4:
                        value *= 1.15
                if target.owner in threats:
                    value *= 1.18
                    if state.remaining_steps <= 180:
                        value *= 1.18
                score = value / max(need + 1.1 * eta, 1)

            plans.append(
                {
                    "kind": "attack",
                    "source_id": source.id,
                    "target_id": target.id,
                    "angle": angle,
                    "eta": eta,
                    "ships": need,
                    "score": score,
                }
            )
    plans.sort(key=lambda item: item["score"], reverse=True)
    return plans


def coordinated_attack_candidates(state, surplus):
    if len(state.my_planets) < 2:
        return []

    runaway = state.runaway_enemy()
    threats = set(state.threatening_enemies())
    plans = []
    targets = sorted(
        [planet for planet in state.planets if planet.owner != state.player],
        key=lambda planet: (
            planet.owner in threats,
            planet.owner == runaway,
            -planet.production,
            planet.owner != -1,
            planet.ships,
            planet.id,
        ),
    )[:18]

    for target in targets:
        source_infos = []
        for source in state.my_planets:
            available = surplus.get(source.id, 0)
            if available < 4:
                continue
            route = state.route(source.id, target.id, available)
            if route is None:
                continue
            angle, eta = route
            if eta > 24:
                continue
            source_infos.append(
                {
                    "source_id": source.id,
                    "available": available,
                    "angle": angle,
                    "eta": eta,
                }
            )

        source_infos.sort(key=lambda item: (item["eta"], -item["available"], item["source_id"]))
        if len(source_infos) < 2:
            continue

        best = None
        source_pool = source_infos[:5]
        for team_size in (2, 3):
            for group in combinations(source_pool, team_size):
                etas = [item["eta"] for item in group]
                if max(etas) - min(etas) > 6:
                    continue
                total_available = sum(item["available"] for item in group)
                if total_available < max(8, int(target.ships) + 1):
                    continue

                hold_turns = 3 if target.owner == -1 else 2
                last_eta = max(etas)
                best_combo = None
                for scale in (0.4, 0.55, 0.7, 0.85, 1.0):
                    sends = [
                        max(1, min(item["available"], int(item["available"] * scale)))
                        for item in group
                    ]
                    extras = [
                        (item["eta"], state.player, send)
                        for item, send in zip(group, sends)
                    ]
                    timeline = state.simulate_planet(
                        target.id,
                        last_eta + hold_turns,
                        extra=extras,
                    )
                    if any(
                        timeline[idx][1] != state.player
                        for idx in range(last_eta - 1, last_eta + hold_turns)
                    ):
                        continue
                    total_send = sum(sends)
                    score = (
                        150.0
                        + 44.0 * target.production
                        + (22.0 if target.owner == runaway else 0.0)
                        + (24.0 if target.owner in threats else 0.0)
                        + (14.0 if target.owner != -1 else 0.0)
                        - 1.6 * total_send
                        - 1.4 * last_eta
                        - 0.35 * (max(etas) - min(etas))
                    )
                    if target.owner == -1 and target.production >= 4:
                        score += 16.0
                    if state.remaining_steps <= 160:
                        if target.owner in threats:
                            score += 50.0
                        elif target.owner == -1:
                            score -= 35.0
                    if team_size == 3:
                        score -= 6.0
                    combo_plan = {
                        "kind": "coordinated_attack",
                        "target_id": target.id,
                        "score": score,
                        "moves": [
                            {
                                "source_id": item["source_id"],
                                "target_id": target.id,
                                "ships": send,
                            }
                            for item, send in zip(group, sends)
                        ],
                    }
                    if best_combo is None or combo_plan["score"] > best_combo["score"]:
                        best_combo = combo_plan

                if best_combo is not None and (best is None or best_combo["score"] > best["score"]):
                    best = best_combo

        if best is not None:
            plans.append(best)

    plans.sort(key=lambda item: item["score"], reverse=True)
    return plans


def coordinated_pressure_candidates(state, surplus):
    if len(state.my_planets) < 2 or state.step < 40:
        return []

    threats = set(state.threatening_enemies())
    if not threats:
        return []

    plans = []
    for target in [planet for planet in state.enemy_planets if planet.owner in threats]:
        source_infos = []
        for source in state.my_planets:
            available = surplus.get(source.id, 0)
            if available < 5:
                continue
            route = state.route(source.id, target.id, available)
            if route is None:
                continue
            angle, eta = route
            if eta > 22:
                continue
            source_infos.append(
                {
                    "source_id": source.id,
                    "available": available,
                    "eta": eta,
                }
            )

        source_infos.sort(key=lambda item: (item["eta"], -item["available"], item["source_id"]))
        if len(source_infos) < 2:
            continue

        best = None
        for team_size in (2, 3):
            for group in combinations(source_infos[:5], team_size):
                etas = [item["eta"] for item in group]
                if max(etas) - min(etas) > 6:
                    continue
                for scale in (0.55, 0.75, 1.0):
                    sends = [
                        max(1, min(item["available"], int(item["available"] * scale)))
                        for item in group
                    ]
                    extras = [
                        (item["eta"], state.player, send)
                        for item, send in zip(group, sends)
                    ]
                    last_eta = max(etas)
                    timeline = state.simulate_planet(target.id, last_eta + 1, extra=extras)
                    owner = timeline[last_eta - 1][1]
                    ships = timeline[last_eta - 1][2]
                    if owner == target.owner and ships > max(2, target.production):
                        continue
                    total_send = sum(sends)
                    score = (
                        120.0
                        + 24.0 * target.production
                        + 0.5 * max(0, target.ships - ships)
                        - 1.25 * total_send
                        - 1.1 * last_eta
                    )
                    if owner == state.player:
                        score += 40.0
                    elif owner == -1:
                        score += 18.0
                    if state.remaining_steps <= 150:
                        score += 28.0
                    plan = {
                        "kind": "coordinated_pressure",
                        "target_id": target.id,
                        "score": score,
                        "moves": [
                            {
                                "source_id": item["source_id"],
                                "target_id": target.id,
                                "ships": send,
                            }
                            for item, send in zip(group, sends)
                        ],
                    }
                    if best is None or plan["score"] > best["score"]:
                        best = plan

        if best is not None:
            plans.append(best)

    plans.sort(key=lambda item: item["score"], reverse=True)
    return plans


def pressure_strike_candidates(state, surplus):
    if state.step < 32:
        return []

    threats = set(state.threatening_enemies())
    if not threats:
        return []

    plans = []
    for target in state.enemy_planets:
        if target.owner not in threats:
            continue
        for source in state.my_planets:
            available = surplus.get(source.id, 0)
            if available < 6:
                continue
            route = state.route(source.id, target.id, available)
            if route is None:
                continue
            angle, eta = route
            if eta > 18:
                continue
            owner, ships = state.projected_state(target.id, eta)
            if owner != target.owner or ships <= 0:
                continue
            takeover_need = state.ships_needed_to_own(target.id, eta, available, hold_turns=2)
            if takeover_need <= available:
                continue
            send = min(available, max(6, int(ships * 0.6) + 1))
            if send >= takeover_need or send > available:
                continue
            score = (
                110.0
                + 18.0 * target.production
                + 0.8 * ships
                - 1.2 * send
                - 1.1 * eta
            )
            if target.owner == state.strongest_enemy:
                score += 26.0
            if target.owner == state.runaway_enemy():
                score += 22.0
            plans.append(
                {
                    "kind": "pressure",
                    "source_id": source.id,
                    "target_id": target.id,
                    "angle": angle,
                    "eta": eta,
                    "ships": send,
                    "score": score,
                }
            )
    plans.sort(key=lambda item: item["score"], reverse=True)
    return plans


def transfer_candidates(state, surplus):
    if len(state.my_planets) < 2 or not state.enemy_planets:
        return []

    front = min(
        state.my_planets,
        key=lambda planet: (
            state.frontier_distance.get(planet.id, 10**9),
            -planet.production,
            -planet.ships,
        ),
    )
    plans = []
    front_dist = state.frontier_distance.get(front.id, 10**9)

    for source in state.my_planets:
        available = surplus.get(source.id, 0)
        if source.id == front.id or available < 12:
            continue
        source_dist = state.frontier_distance.get(source.id, 10**9)
        if source_dist <= front_dist * 1.18:
            continue
        send = int(available * FRONTIER_TRANSFER_RATIO)
        if send < 10:
            continue
        route = state.route(source.id, front.id, send)
        if route is None:
            continue
        angle, eta = route
        if eta > 26:
            continue
        score = 40.0 + 0.7 * source_dist - 1.2 * eta + 2.0 * front.production
        plans.append(
            {
                "kind": "transfer",
                "source_id": source.id,
                "target_id": front.id,
                "angle": angle,
                "eta": eta,
                "ships": send,
                "score": score,
            }
        )
    plans.sort(key=lambda item: item["score"], reverse=True)
    return plans


def opening_burst(state):
    if state.step <= 0 or state.step > 28:
        return None
    if len(state.my_planets) > 4 or state.production.get(state.player, 0) >= 10:
        return None
    actions = []
    claimed_targets = set()
    early_commitment = len(state.my_planets) == 1 and any(
        fleet.owner == state.player for fleet in state.fleets
    )

    for source in sorted(
        state.my_planets,
        key=lambda planet: (
            planet.production,
            -int(planet.ships),
            -state.frontier_distance.get(planet.id, 10**9),
        ),
    ):
        if source.ships < 5:
            continue
        incoming_hostile = any(owner != state.player for _eta, owner, _ships in state.arrivals.get(source.id, []))
        if state.step <= 6 and len(state.my_planets) == 1 and not incoming_hostile:
            available = int(source.ships)
        else:
            available = max(int(source.ships) - 1, state.safe_send(source.id))
        if available < 4:
            continue

        best = None
        for target in sorted(
            state.neutral_planets,
            key=lambda planet: (
                -planet.production,
                dist(source.x, source.y, planet.x, planet.y),
                planet.ships,
            ),
        ):
            if target.id in claimed_targets:
                continue
            if any(owner == state.player for _eta, owner, _ships in state.arrivals.get(target.id, [])):
                continue
            probe = max(1, min(available, int(target.ships) + max(2, int(target.production))))
            route = state.route(source.id, target.id, probe)
            if route is None:
                continue
            angle, eta = route
            if eta > 20:
                continue
            hostile_eta = state.fastest_hostile_eta(target.id)
            margin = hostile_eta - eta
            if hostile_eta < eta - 2:
                continue
            if early_commitment and target.production < 5 and eta > 12:
                continue
            if state.step <= 18 and target.production <= 1 and eta > 6:
                continue
            if state.step <= 24 and target.production <= 3 and eta > 16:
                continue
            need = int(target.ships) + 1
            if hostile_eta <= eta + 1:
                need += 2
            if target.production >= 4:
                need += 1
            need = max(4, need)
            if need > available:
                continue
            payback_window = 26.0 if len(state.my_planets) == 1 else 22.0
            payback_turns = max(2.0, payback_window - eta)
            score = (
                26.0 * target.production * payback_turns
                + 10.0 * target.production
                + 2.2 * margin
                - 3.0 * need
                - 1.9 * eta
            )
            if target.production <= 1:
                score -= 45.0 + 1.5 * state.step
            if eta > 14:
                score -= 6.0 * (eta - 14)
            if orbital_radius(target) + target.radius >= ROTATION_LIMIT:
                score += 8.0
            if hostile_eta >= eta + 2:
                score += 10.0
            elif hostile_eta <= eta:
                score -= 6.0
            if early_commitment and eta > 10:
                score -= 30.0
            if best is None or score > best["score"]:
                best = {
                    "source_id": source.id,
                    "target_id": target.id,
                    "angle": angle,
                    "ships": need,
                    "score": score,
                }
        if best is not None:
            actions.append([best["source_id"], best["angle"], best["ships"]])
            claimed_targets.add(best["target_id"])
    return actions


def choose_plan(state):
    if not state.my_planets or state.step == 0:
        return []

    burst = opening_burst(state)
    if burst is not None and (burst or (state.step <= 12 and len(state.my_planets) <= 2)):
        return burst

    surplus = {}
    for planet in state.my_planets:
        safe = state.safe_send(planet.id)
        if len(state.enemy_ids) >= 3:
            ratio = 0.72 if state.runaway_enemy() is not None and state.step >= 30 else 0.58
            aggressive = max(0, min(int(planet.ships) - 1, int(planet.ships * ratio)))
            surplus[planet.id] = max(safe, aggressive)
        else:
            surplus[planet.id] = safe
    actions = []
    used_pairs = set()

    def commit(plan):
        if "moves" in plan:
            prepared = []
            for move in plan["moves"]:
                pair = (move["source_id"], move["target_id"])
                if pair in used_pairs:
                    return False
                send = min(move["ships"], surplus.get(move["source_id"], 0))
                if send <= 0:
                    return False
                route = state.route(move["source_id"], move["target_id"], send)
                if route is None:
                    return False
                prepared.append((move["source_id"], move["target_id"], send, route))
            for source_id, target_id, send, route in prepared:
                angle, eta = route
                actions.append([source_id, angle, send])
                surplus[source_id] -= send
                state.planned[target_id].append((eta, state.player, send))
                used_pairs.add((source_id, target_id))
            return True

        pair = (plan["source_id"], plan["target_id"])
        if pair in used_pairs:
            return False
        send = min(plan["ships"], surplus.get(plan["source_id"], 0))
        if send <= 0:
            return False
        route = state.route(plan["source_id"], plan["target_id"], send)
        if route is None:
            return False
        angle, eta = route
        actions.append([plan["source_id"], angle, send])
        surplus[plan["source_id"]] -= send
        state.planned[plan["target_id"]].append((eta, state.player, send))
        used_pairs.add(pair)
        return True

    for plan in defense_candidates(state, surplus):
        if surplus.get(plan["source_id"], 0) < plan["ships"]:
            continue
        commit(plan)

    mission_budget = max(4, min(8, len(state.my_planets) + 1))
    mission_count = 0

    while mission_count < mission_budget:
        best = None
        for pool in (
            coordinated_pressure_candidates(state, surplus),
            coordinated_attack_candidates(state, surplus),
            opportunistic_enemy_takeovers(state, surplus),
            pressure_strike_candidates(state, surplus),
            contested_takeovers(state, surplus),
            attack_candidates(state, surplus),
        ):
            for plan in pool:
                if "moves" in plan:
                    if any(surplus.get(move["source_id"], 0) < move["ships"] for move in plan["moves"]):
                        continue
                elif surplus.get(plan["source_id"], 0) < plan["ships"]:
                    continue
                best = plan
                break
            if best is not None:
                break
        if best is None:
            break
        if not commit(best):
            break
        mission_count += 1

    transfer_count = 0
    for plan in transfer_candidates(state, surplus):
        if transfer_count >= 2:
            break
        if surplus.get(plan["source_id"], 0) < plan["ships"]:
            continue
        if commit(plan):
            transfer_count += 1

    return actions


def smart_agent(obs, config=None):
    state = GameState(obs)
    return choose_plan(state)


agent = smart_agent
