import math

try:
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
except Exception:
    Planet = None


CENTER_X, CENTER_Y = 50.0, 50.0
SUN_RADIUS = 10.0
ROTATION_LIMIT = 50.0  # orbital_radius + planet_radius < 50 => rotating


class SimplePlanet:
    def __init__(self, arr):
        self.id = arr[0]
        self.owner = arr[1]
        self.x = float(arr[2])
        self.y = float(arr[3])
        self.radius = float(arr[4])
        self.ships = int(arr[5])
        self.production = int(arr[6])


def as_planet(p):
    if Planet is not None:
        try:
            return Planet(*p)
        except Exception:
            pass
    return SimplePlanet(p)


def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def angle_to(ax, ay, bx, by):
    return math.atan2(by - ay, bx - ax)


def is_rotating(planet):
    orbital_radius = dist(planet.x, planet.y, CENTER_X, CENTER_Y)
    return orbital_radius + planet.radius < ROTATION_LIMIT


def segment_intersects_circle(x1, y1, x2, y2, cx, cy, r):
    """
    线段与圆是否相交。用于避开太阳。
    """
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return dist(x1, y1, cx, cy) <= r

    t = ((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    px = x1 + t * dx
    py = y1 + t * dy
    return dist(px, py, cx, cy) <= r


def path_crosses_sun(src, tgt):
    # 简化版：从星球中心到星球中心判断。
    # 第一版先用这个近似，已经能过滤掉大部分送死路线。
    return segment_intersects_circle(
        src.x, src.y, tgt.x, tgt.y,
        CENTER_X, CENTER_Y, SUN_RADIUS
    )


def reserve_ships(planet, step):
    if step < 40:
        return max(6, 2 + planet.production)
    return max(4, 1 + planet.production)


def neutral_attack_need(target):
    # 中立星不会产兵，所以只看当前 ships + buffer
    buffer = 2
    if target.production >= 4:
        buffer += 2
    return target.ships + buffer


def target_score(src, tgt):
    """
    第一版打分：偏好高产、近、守军低的中立静止星
    """
    d = dist(src.x, src.y, tgt.x, tgt.y)
    return (
        9.0 * tgt.production
        - 1.3 * tgt.ships
        - 0.22 * d
    )


def get_step(obs):
    if isinstance(obs, dict):
        return int(obs.get("step", 0))
    return int(getattr(obs, "step", 0))


def get_field(obs, name, default=None):
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def agent(obs, config):
    """
    第一版 baseline：
    - 只吃静止中立星
    - 不抢彗星
    - 不打旋转星
    - 不打敌人
    - 避开太阳
    """
    player = get_field(obs, "player", 0)
    planets_raw = get_field(obs, "planets", [])
    comet_ids = set(get_field(obs, "comet_planet_ids", []))
    step = get_step(obs)

    planets = [as_planet(p) for p in planets_raw]

    my_planets = [p for p in planets if p.owner == player]
    neutral_planets = [
        p for p in planets
        if p.owner == -1
        and p.id not in comet_ids
        and not is_rotating(p)
    ]

    if not my_planets or not neutral_planets:
        return []

    actions = []
    used_targets = set()

    # 让高产/高兵的己方星优先出手
    my_planets.sort(key=lambda p: (p.production, p.ships), reverse=True)

    for src in my_planets:
        keep = reserve_ships(src, step)
        available = src.ships - keep
        if available <= 0:
            continue

        best = None
        best_score = -1e18

        for tgt in neutral_planets:
            if tgt.id in used_targets:
                continue

            if path_crosses_sun(src, tgt):
                continue

            need = neutral_attack_need(tgt)
            if available < need:
                continue

            score = target_score(src, tgt)
            if score > best_score:
                best_score = score
                best = (tgt, need)

        if best is None:
            continue

        tgt, send_ships = best
        actions.append([
            int(src.id),
            float(angle_to(src.x, src.y, tgt.x, tgt.y)),
            int(send_ships),
        ])
        used_targets.add(tgt.id)

    return actions