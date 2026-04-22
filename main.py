import math

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    CENTER,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
    Planet,
)


def fastest_capture_planets(obs, top_k=3):
    # 返回当前回合“最早可能被我方占领”的目标星球列表。
    # 排序规则是确定性的：
    # 1. 最早到达并能打下优先
    # 2. 需要发兵更少优先
    # 3. 当前守军更少优先
    # 4. 生产更高优先
    player = obs.get("player", 0)
    angular_velocity = obs.get("angular_velocity", 0.0)
    planets = [Planet(*planet) for planet in obs.get("planets", [])]

    my_planets = [planet for planet in planets if planet.owner == player]
    targets = [planet for planet in planets if planet.owner != player]
    if not my_planets or not targets:
        return []

    candidates = []

    for target in targets:
        best_plan = None

        for source in my_planets:
            reserve = max(12, int(4 * source.production), int(source.ships * 0.35))
            available = source.ships - reserve
            if available < 20:
                continue

            for turns in range(1, 41):
                orbital_radius = math.hypot(target.x - CENTER, target.y - CENTER)
                if orbital_radius + target.radius < ROTATION_RADIUS_LIMIT:
                    dx = target.x - CENTER
                    dy = target.y - CENTER
                    angle_at_turn = math.atan2(dy, dx) + angular_velocity * turns
                    future_target = (
                        CENTER + orbital_radius * math.cos(angle_at_turn),
                        CENTER + orbital_radius * math.sin(angle_at_turn),
                    )
                else:
                    future_target = (target.x, target.y)

                vx = future_target[0] - source.x
                vy = future_target[1] - source.y
                travel_distance = math.hypot(vx, vy)

                buffer = 2 if target.owner == -1 else 3
                defenders = target.ships
                if target.owner != -1:
                    defenders += target.production * turns
                send = max(defenders + buffer, 20)
                if send > available:
                    continue

                speed = 1.0 + (6.0 - 1.0) * (math.log(send) / math.log(1000)) ** 1.5
                speed = min(speed, 6.0)
                if abs(travel_distance - speed * turns) > 1.0:
                    continue

                sun_dx = CENTER - source.x
                sun_dy = CENTER - source.y
                vv = vx * vx + vy * vy
                if vv == 0:
                    closest_distance_to_sun = math.hypot(source.x - CENTER, source.y - CENTER)
                else:
                    t = max(0.0, min(1.0, (sun_dx * vx + sun_dy * vy) / vv))
                    closest_x = source.x + t * vx
                    closest_y = source.y + t * vy
                    closest_distance_to_sun = math.hypot(closest_x - CENTER, closest_y - CENTER)
                if closest_distance_to_sun < SUN_RADIUS + 0.5:
                    continue

                plan = (turns, send, target.ships, -target.production)
                if best_plan is None or plan < best_plan:
                    best_plan = plan
                break

        if best_plan is not None:
            candidates.append((best_plan, target))

    candidates.sort(key=lambda item: item[0])
    return [target for _, target in candidates[:top_k]]


def earliest_falling_my_planets(obs):
    # 返回当前己方最先可能被敌方攻下的星球列表。
    # 这里只分析当前已经在途的敌方舰队，不猜测未来新发射的舰队。
    player = obs.get("player", 0)
    angular_velocity = obs.get("angular_velocity", 0.0)
    planets = [Planet(*planet) for planet in obs.get("planets", [])]
    fleets = obs.get("fleets", [])

    my_planets = [planet for planet in planets if planet.owner == player]
    if not my_planets:
        return []

    impacts = {planet.id: [] for planet in my_planets}

    for fleet in fleets:
        fleet_owner = fleet[1]
        if fleet_owner == player:
            continue

        fleet_x = fleet[2]
        fleet_y = fleet[3]
        fleet_angle = fleet[4]
        fleet_ships = fleet[6]
        fleet_speed = 1.0 + (6.0 - 1.0) * (math.log(max(int(fleet_ships), 1)) / math.log(1000)) ** 1.5
        fleet_speed = min(fleet_speed, 6.0)

        best_hit = None
        for planet in my_planets:
            orbital_radius = math.hypot(planet.x - CENTER, planet.y - CENTER)
            for turns in range(1, 41):
                if orbital_radius + planet.radius < ROTATION_RADIUS_LIMIT:
                    dx = planet.x - CENTER
                    dy = planet.y - CENTER
                    angle_at_turn = math.atan2(dy, dx) + angular_velocity * turns
                    future_planet = (
                        CENTER + orbital_radius * math.cos(angle_at_turn),
                        CENTER + orbital_radius * math.sin(angle_at_turn),
                    )
                else:
                    future_planet = (planet.x, planet.y)

                fleet_future = (
                    fleet_x + math.cos(fleet_angle) * fleet_speed * turns,
                    fleet_y + math.sin(fleet_angle) * fleet_speed * turns,
                )
                if math.hypot(fleet_future[0] - future_planet[0], fleet_future[1] - future_planet[1]) > planet.radius + 1.0:
                    continue

                if best_hit is None or turns < best_hit[0]:
                    best_hit = (turns, planet.id, fleet_ships)
                break

        if best_hit is not None:
            impacts[best_hit[1]].append((best_hit[0], best_hit[2]))

    earliest_turn = None
    falling_ids = []

    for planet in my_planets:
        events = sorted(impacts[planet.id])
        if not events:
            continue

        defenders = planet.ships
        previous_turn = 0

        for turns, enemy_ships in events:
            defenders += planet.production * (turns - previous_turn)
            defenders -= enemy_ships
            previous_turn = turns

            if defenders < 0:
                if earliest_turn is None or turns < earliest_turn:
                    earliest_turn = turns
                    falling_ids = [planet.id]
                elif turns == earliest_turn:
                    falling_ids.append(planet.id)
                break

    return [planet for planet in my_planets if planet.id in falling_ids]


def agent(obs):
    # 两阶段策略：
    # 1. 前期 0-30 回合：只追求尽快占点，优先选择最早能打下的近点。
    # 2. 中后期 30+ 回合：先分析当前已在途舰队。
    #    - 己方星球可能失守时优先支援。
    #    - 非己方星球若会被别人打残或打掉，则尝试卡时间接手。
    #    - 如果没有明显机会，再继续占最近的点。
    player = obs.get("player", 0)
    step = obs.get("step", 0)
    angular_velocity = obs.get("angular_velocity", 0.0)
    planets = [Planet(*planet) for planet in obs.get("planets", [])]
    fleets = obs.get("fleets", [])

    my_planets = [planet for planet in planets if planet.owner == player]
    if not my_planets:
        return []

    moves = []
    committed_ships = {planet.id: 0 for planet in planets}
    reserved_neutrals = set()
    threatened = {}

    if step >= 30:
        # 估算当前已发射舰队最可能最先撞上的星球，以及撞上时会造成多大损失。
        for fleet in fleets:
            fleet_owner = fleet[1]
            fleet_x = fleet[2]
            fleet_y = fleet[3]
            fleet_angle = fleet[4]
            fleet_ships = fleet[6]

            fleet_speed = 1.0 + (6.0 - 1.0) * (math.log(max(int(fleet_ships), 1)) / math.log(1000)) ** 1.5
            fleet_speed = min(fleet_speed, 6.0)
            best_hit = None

            for planet in planets:
                orbital_radius = math.hypot(planet.x - CENTER, planet.y - CENTER)
                for turns in range(1, 41):
                    if orbital_radius + planet.radius < ROTATION_RADIUS_LIMIT:
                        dx = planet.x - CENTER
                        dy = planet.y - CENTER
                        angle_at_turn = math.atan2(dy, dx) + angular_velocity * turns
                        future_planet = (
                            CENTER + orbital_radius * math.cos(angle_at_turn),
                            CENTER + orbital_radius * math.sin(angle_at_turn),
                        )
                    else:
                        future_planet = (planet.x, planet.y)

                    fleet_future = (
                        fleet_x + math.cos(fleet_angle) * fleet_speed * turns,
                        fleet_y + math.sin(fleet_angle) * fleet_speed * turns,
                    )
                    if math.hypot(fleet_future[0] - future_planet[0], fleet_future[1] - future_planet[1]) > planet.radius + 1.0:
                        continue

                    if best_hit is None or turns < best_hit["turns"]:
                        best_hit = {"turns": turns, "planet_id": planet.id}
                    break

            if best_hit is None:
                continue

            for planet in planets:
                if planet.id != best_hit["planet_id"]:
                    continue

                defenders = planet.ships
                if planet.owner != -1:
                    defenders += planet.production * best_hit["turns"]

                remaining = defenders - fleet_ships
                damage_ratio = fleet_ships / max(defenders, 1)

                # 只记录会造成明显削弱的在途舰队。
                if damage_ratio < 0.6 and remaining > 0:
                    break

                current = threatened.get(planet.id)
                record = {
                    "turns": best_hit["turns"],
                    "remaining": remaining,
                    "fleet_owner": fleet_owner,
                    "planet_owner": planet.owner,
                }
                if current is None or record["turns"] < current["turns"]:
                    threatened[planet.id] = record
                break

    # 兵多、产能高的己方星球先行动。
    for source in sorted(my_planets, key=lambda planet: (planet.ships, planet.production), reverse=True):
        reserve = max(12, int(4 * source.production), int(source.ships * 0.35))
        available = source.ships - reserve
        if available < 20:
            continue

        best_choice = None

        for target in planets:
            if target.id == source.id:
                continue

            # 前期只做简单扩张。
            if step < 30:
                if target.owner == player:
                    continue
                if target.owner == -1 and target.id in reserved_neutrals:
                    continue
                modes = ["expand"]
            else:
                modes = []
                if target.owner == player and target.id in threatened and threatened[target.id]["remaining"] < 0:
                    modes.append("support")
                if target.owner != player and target.id in threatened:
                    modes.append("steal")
                if target.owner != player and target.id not in threatened:
                    modes.append("expand")

            start = (source.x, source.y)

            for mode in modes:
                local_reserve = reserve
                if mode == "support":
                    # 支援时允许更积极地抽调兵力。
                    local_reserve = max(8, int(2 * source.production), int(source.ships * 0.2))
                local_available = source.ships - local_reserve
                if local_available < 20:
                    continue

                earliest_plan = None

                for turns in range(1, 41):
                    orbital_radius = math.hypot(target.x - CENTER, target.y - CENTER)
                    if orbital_radius + target.radius < ROTATION_RADIUS_LIMIT:
                        dx = target.x - CENTER
                        dy = target.y - CENTER
                        angle_at_turn = math.atan2(dy, dx) + angular_velocity * turns
                        future_target = (
                            CENTER + orbital_radius * math.cos(angle_at_turn),
                            CENTER + orbital_radius * math.sin(angle_at_turn),
                        )
                    else:
                        future_target = (target.x, target.y)

                    vx = future_target[0] - start[0]
                    vy = future_target[1] - start[1]
                    travel_distance = math.hypot(vx, vy)

                    # 先看该回合是否能飞到目标位置附近。
                    provisional_send = 20
                    if mode == "support":
                        provisional_send = max(-threatened[target.id]["remaining"] + 1, 20)
                    elif mode == "steal":
                        provisional_send = max(max(threatened[target.id]["remaining"], 0) + 3, 20)
                    else:
                        provisional_send = max(target.ships + committed_ships[target.id] + 3, 20)

                    if provisional_send > local_available:
                        continue

                    speed = 1.0 + (6.0 - 1.0) * (math.log(provisional_send) / math.log(1000)) ** 1.5
                    speed = min(speed, 6.0)
                    if abs(travel_distance - speed * turns) > 1.0:
                        continue

                    # 如果路线穿过太阳，就跳过。
                    sun_dx = CENTER - start[0]
                    sun_dy = CENTER - start[1]
                    vv = vx * vx + vy * vy
                    if vv == 0:
                        closest_distance_to_sun = math.hypot(start[0] - CENTER, start[1] - CENTER)
                    else:
                        t = max(0.0, min(1.0, (sun_dx * vx + sun_dy * vy) / vv))
                        closest_x = start[0] + t * vx
                        closest_y = start[1] + t * vy
                        closest_distance_to_sun = math.hypot(closest_x - CENTER, closest_y - CENTER)
                    if closest_distance_to_sun < SUN_RADIUS + 0.5:
                        continue

                    if mode == "support":
                        info = threatened[target.id]
                        if turns > info["turns"]:
                            continue
                        send = max(-info["remaining"] + 1, 20)
                    elif mode == "steal":
                        info = threatened[target.id]
                        # 等别人先把目标打残，再尽量快速接手。
                        if turns + 2 < info["turns"]:
                            continue
                        defenders = max(info["remaining"], 0)
                        if defenders > 0 and target.owner != -1:
                            defenders += target.production * max(turns - info["turns"], 0)
                        send = max(defenders + 3, 20)
                    else:
                        defenders = target.ships + committed_ships[target.id]
                        if target.owner != -1:
                            defenders += target.production * turns
                        send = max(defenders + 3, 20)

                    if send > local_available:
                        continue

                    launch_angle = math.atan2(vy, vx)
                    mode_priority = 2
                    if mode == "support":
                        mode_priority = 0
                    if mode == "steal":
                        mode_priority = 1

                    earliest_plan = (
                        mode_priority,
                        turns,
                        send,
                        target.ships,
                        -target.production,
                        [source.id, launch_angle, int(send)],
                        target.id,
                    )
                    break

                if earliest_plan is None:
                    continue

                # 完全确定性的选择规则：
                # 1. 支援 > 抢残局 > 普通扩张
                # 2. 更早到达优先
                # 3. 所需兵更少优先
                # 4. 当前守军更少优先
                # 5. 生产更高优先
                if best_choice is None or earliest_plan[:5] < best_choice[:5]:
                    best_choice = earliest_plan

        if best_choice is None:
            continue

        moves.append(best_choice[5])
        committed_ships[best_choice[6]] += best_choice[5][2]
        if any(planet.id == best_choice[6] and planet.owner == -1 for planet in planets):
            reserved_neutrals.add(best_choice[6])

    return moves
