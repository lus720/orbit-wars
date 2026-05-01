import argparse
import contextlib
import copy
import hashlib
import io
import json
import os
import re
from collections import Counter
from pathlib import Path


LOG_PREFIX = "OWLOG "
DEFAULT_CHECKPOINTS = (0, 5, 10, 20, 30, 40, 60, 80, 100, 120, 150, 200, 300, 400, 499)
DEFAULT_RESAMPLE_REASONS = (
    "no_attack_budget",
    "all_candidates_filtered",
    "partial_options_unassembled",
    "multi_eta_filtered",
    "missions_rejected",
    "near_soft_deadline",
)
INTERESTING_SAMPLE_KEYS = (
    "accepted_mission",
    "route_guard_reject",
    "route_blocker_too_strong",
    "route_blocker_pass",
    "route_target_miss",
    "aim_no_solution",
    "mission_settle_none",
    "mission_need_fail",
    "mission_midgame_block",
    "append_short",
    "append_zero",
    "multi_missing_fail",
    "multi_total_fail",
    "multi_eta_fail",
    "multi_owner_fail",
    "source_hold_cap",
    "capture_hold_need",
    "capture_reaction_need",
    "boxed_accept",
    "boxed_settle_none",
    "boxed_source_swing_guard",
    "boxed_source_swing_reject",
    "timeout_owner_fail",
    "timeout_accept",
    "opening_filter",
    "append_poor_opening_target",
    "append_opening_lowprod_block",
    "opening_wait_relaxed",
    "opening_soften_accept",
    "opening_soften_no_followup",
    "opening_soften_capture_fail",
    "opening_soften_mobile_partial_block",
    "opening_fast_expand_low_target_near_core_block",
    "low_prod_home_reserve_relief",
    "reserve_relief",
    "rescue_budget_relief",
    "swing_rescue_budget_relief",
    "rescue_pressure_send",
    "proactive_reinforce_target",
    "proactive_reinforce_budget_relief",
    "proactive_reinforce_mission_built",
    "proactive_reinforce_low_budget",
    "proactive_reinforce_no_probe",
    "proactive_reinforce_settle_none",
    "opening_soften_commit",
    "opening_soften_followup_accept",
    "opening_soften_followup_wait",
    "opening_soften_followup_inflight",
    "opening_soften_followup_no_probe",
    "opening_soften_followup_need_fail",
    "opening_soften_followup_owner_fail",
    "opening_soften_followup_stolen",
    "opening_soften_followup_expired",
    "threatened_prod_retreat_block",
    "rear_stage_accept",
    "rear_stage_low_home_block",
    "route_guard_relaxed",
    "prod_pressure_low_neutral",
    "prod_pressure_heavy_neutral",
    "prod_pressure_low_hostile",
    "prod_pressure_hostile_core",
    "prod_pressure_low_rescue",
    "prod_pressure_low_recapture",
)


def load_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def episode_id_from_path(path):
    match = re.search(r"episode-(.+)\.json$", Path(path).name)
    return match.group(1) if match else None


def stable_json_hash(payload):
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]


def map_signature_from_episode(episode):
    for step in episode.get("steps", []):
        for state in step:
            obs = state.get("observation") if isinstance(state, dict) else None
            if not isinstance(obs, dict):
                continue
            planets = obs.get("initial_planets") or obs.get("planets") or []
            if not planets:
                continue
            payload = {
                "planets": sorted(
                    [
                        [
                            int(planet[0]),
                            int(planet[1]),
                            round(float(planet[2]), 3),
                            round(float(planet[3]), 3),
                            round(float(planet[4]), 3),
                            int(planet[5]),
                            int(planet[6]),
                        ]
                        for planet in planets
                    ],
                    key=lambda row: row[0],
                ),
                "angular_velocity": round(float(obs.get("angular_velocity", 0.0)), 8),
                "comets": obs.get("comets") or [],
                "comet_planet_ids": obs.get("comet_planet_ids") or [],
            }
            return {
                "hash": stable_json_hash(payload),
                "planet_count": len(payload["planets"]),
            }
    return {"hash": "unknown", "planet_count": 0}


def load_match_manifest(replay_dir, episode_id):
    if episode_id is None:
        return None
    path = replay_dir / f"manifest-{episode_id}.json"
    if not path.exists():
        return None
    manifest = load_json(path)
    manifest["path"] = str(path)
    return manifest


def team_names(episode):
    info = episode.get("info", {})
    names = info.get("TeamNames")
    if names:
        return list(names)
    return [agent.get("Name", f"slot{idx}") for idx, agent in enumerate(info.get("Agents", []))]


def find_player_slot(episode, player_name):
    names = team_names(episode)
    lowered = player_name.lower()
    for slot, name in enumerate(names):
        if str(name).lower() == lowered:
            return slot
    for slot, name in enumerate(names):
        if lowered in str(name).lower():
            return slot
    return None


def infer_slot_from_log_file(replay_dir, episode_id):
    if episode_id is None:
        return None
    pattern = re.compile(rf"^{re.escape(episode_id)}-(\d+)\.json$")
    matches = []
    for path in replay_dir.glob(f"{episode_id}-*.json"):
        match = pattern.match(path.name)
        if match:
            matches.append(int(match.group(1)))
    if len(matches) == 1:
        return matches[0]
    return None


def slot_from_match_manifest(match_manifest):
    if not match_manifest or match_manifest.get("my_slot") is None:
        return None
    try:
        return int(match_manifest["my_slot"])
    except (TypeError, ValueError):
        return None


def resolve_player_slot(episode, replay_dir, episode_id, player_name, match_manifest):
    slot = slot_from_match_manifest(match_manifest)
    if slot is not None:
        return slot, "manifest"
    slot = find_player_slot(episode, player_name)
    if slot is not None:
        return slot, "team_name"
    slot = infer_slot_from_log_file(replay_dir, episode_id)
    if slot is not None:
        return slot, "log_file"
    return None, "unknown"


def canonical_observation(episode, turn):
    step = episode["steps"][turn]
    preferred = [0] + [idx for idx in range(len(step)) if idx != 0]
    for slot in preferred:
        state = step[slot]
        obs = state.get("observation") if isinstance(state, dict) else None
        if isinstance(obs, dict) and obs.get("planets") is not None:
            return obs
    raise ValueError(f"No observation with planets at turn {turn}")


def action_at(episode, turn, slot):
    step = episode["steps"][turn]
    if slot is None or slot >= len(step):
        return []
    state = step[slot]
    if not isinstance(state, dict):
        return []
    return state.get("action") or []


def score_table(obs, num_players):
    planets = [0] * num_players
    prod = [0] * num_players
    planet_ships = [0.0] * num_players
    fleet_ships = [0.0] * num_players
    for planet in obs.get("planets", []):
        owner = int(planet[1])
        if 0 <= owner < num_players:
            planets[owner] += 1
            prod[owner] += int(planet[6])
            planet_ships[owner] += float(planet[5])
    for fleet in obs.get("fleets", []):
        owner = int(fleet[1])
        if 0 <= owner < num_players:
            fleet_ships[owner] += float(fleet[6])
    scores = [planet_ships[idx] + fleet_ships[idx] for idx in range(num_players)]
    return {
        "planets": planets,
        "production": prod,
        "planet_ships": [round(value, 1) for value in planet_ships],
        "fleet_ships": [round(value, 1) for value in fleet_ships],
        "scores": [round(value, 1) for value in scores],
    }


def production_gap_rows(episode, slot, num_players):
    if slot is None:
        return []
    rows = []
    previous_gap = None
    for turn in range(len(episode.get("steps", []))):
        table = score_table(canonical_observation(episode, turn), num_players)
        my_prod = table["production"][slot]
        enemy_prod = sum(
            value for idx, value in enumerate(table["production"]) if idx != slot
        )
        gap = my_prod - enemy_prod
        delta = 0 if previous_gap is None else gap - previous_gap
        rows.append(
            {
                "turn": turn,
                "my_prod": my_prod,
                "enemy_prod": enemy_prod,
                "gap": gap,
                "delta": delta,
            }
        )
        previous_gap = gap
    return rows


def prod_gap_delta_for_owner_change(old_owner, new_owner, production, slot):
    delta = 0
    if old_owner == slot:
        delta -= production
    elif old_owner not in (None, -1):
        delta += production
    if new_owner == slot:
        delta += production
    elif new_owner not in (None, -1):
        delta -= production
    return delta


def production_swing_events(episode, slot, limit=24):
    if slot is None:
        return []
    events = []
    start_turn = 0
    previous = {}
    while start_turn < len(episode.get("steps", [])):
        previous = {
            planet[0]: int(planet[1])
            for planet in canonical_observation(episode, start_turn).get("planets", [])
        }
        if previous:
            break
        start_turn += 1
    for turn in range(start_turn + 1, len(episode.get("steps", []))):
        obs = canonical_observation(episode, turn)
        planets_by_id = {planet[0]: planet for planet in obs.get("planets", [])}
        current = {pid: int(planet[1]) for pid, planet in planets_by_id.items()}
        for planet_id, new_owner in current.items():
            old_owner = previous.get(planet_id)
            if old_owner == new_owner:
                continue
            planet = planets_by_id[planet_id]
            production = int(planet[6])
            delta = prod_gap_delta_for_owner_change(old_owner, new_owner, production, slot)
            if delta == 0:
                continue
            events.append(
                {
                    "turn": turn,
                    "planet": planet_id,
                    "from": old_owner,
                    "to": new_owner,
                    "production": production,
                    "prod_gap_delta": delta,
                    "swing": max(0, -delta),
                }
            )
        previous = current
    events.sort(key=lambda item: (item["turn"], item["prod_gap_delta"], item["planet"]))
    if limit is None:
        return events
    important = [
        event
        for event in events
        if event["prod_gap_delta"] <= -4 or event["swing"] >= 4
    ]
    return important[:limit]


def production_drop_summary(episode, slot, num_players):
    rows = production_gap_rows(episode, slot, num_players)
    if not rows:
        return {}
    negative_delta_sum = sum(max(0, -row["delta"]) for row in rows)
    max_negative_delta = max((max(0, -row["delta"]) for row in rows), default=0)
    windows = {}
    for window in (5, 10, 20):
        worst = 0
        worst_row = None
        for idx, row in enumerate(rows):
            end_idx = min(len(rows) - 1, idx + window)
            drop = row["gap"] - rows[end_idx]["gap"]
            if drop > worst:
                worst = drop
                worst_row = {
                    "start": row["turn"],
                    "end": rows[end_idx]["turn"],
                    "from": row["gap"],
                    "to": rows[end_idx]["gap"],
                    "drop": drop,
                }
        windows[f"max_drop_{window}"] = worst_row or {
            "start": 0,
            "end": 0,
            "from": rows[0]["gap"],
            "to": rows[0]["gap"],
            "drop": 0,
        }
    rapid_turns = [
        {
            "turn": row["turn"],
            "gap": row["gap"],
            "delta": row["delta"],
            "my_prod": row["my_prod"],
            "enemy_prod": row["enemy_prod"],
        }
        for row in rows
        if row["delta"] <= -4
    ][:12]
    return {
        "final_gap": rows[-1]["gap"],
        "min_gap": min(row["gap"] for row in rows),
        "max_gap": max(row["gap"] for row in rows),
        "negative_delta_sum": negative_delta_sum,
        "max_negative_delta": max_negative_delta,
        "rapid_turns": rapid_turns,
        **windows,
    }


def player_result(slot, rewards):
    if slot is None or slot >= len(rewards):
        return "UNKNOWN"
    winners = [idx for idx, reward in enumerate(rewards) if reward == 1]
    if slot not in winners:
        return "LOSS"
    return "WIN" if len(winners) == 1 else "TIE_WIN"


def compact_counter(counter, limit=6):
    return [[key, value] for key, value in counter.most_common(limit)]


def compact_debug_samples(debug, limit_keys=6, limit_rows=2):
    samples = debug.get("samples") or {}
    result = {}
    for key in INTERESTING_SAMPLE_KEYS:
        rows = samples.get(key)
        if rows:
            result[key] = rows[:limit_rows]
        if len(result) >= limit_keys:
            break
    return result


def parse_owlog_line(line):
    if not line.startswith(LOG_PREFIX):
        return None
    try:
        return json.loads(line[len(LOG_PREFIX) :])
    except json.JSONDecodeError:
        return None


def load_log_records(log_path):
    if log_path is None or not log_path.exists():
        return []
    data = load_json(log_path)
    records = []
    for turn_index, row in enumerate(data):
        items = row if isinstance(row, list) else [row]
        for item in items:
            if not isinstance(item, dict):
                continue
            for stream_name in ("stderr", "stdout"):
                text = item.get(stream_name) or ""
                for line in str(text).splitlines():
                    record = parse_owlog_line(line)
                    if record is None:
                        continue
                    record["_log_turn"] = turn_index
                    records.append(record)
    return records


def analyze_log_records(records):
    empty = Counter()
    stages = Counter()
    phases = Counter()
    profiles = Counter()
    flags = Counter()
    diag_tags = Counter()
    debug_counts = Counter()
    first_actions = []
    issue_turns = []
    route_trace_turns = []
    mission_trace_turns = []
    timeout_trace_turns = []
    boxed_samples = []
    pressure_turns = 0
    max_ms = 0.0
    sent_total = 0
    first_action_step = None
    max_prod_deficit = 0
    max_total_deficit = 0
    negative_logged_prod_delta = 0
    max_logged_prod_drop = 0
    max_threatened_prod_sum = 0
    max_avoidable_prod_swing = 0
    avoidable_swing_turns = []

    for record in records:
        if record.get("empty"):
            empty[record.get("empty")] += 1
        if record.get("phase") is not None:
            phases[str(record.get("phase"))] += 1
        debug = record.get("dbg") or {}
        if debug.get("stage") is not None:
            stages[str(debug.get("stage"))] += 1
        if record.get("profile") is not None:
            profiles[str(record.get("profile"))] += 1
        for flag in record.get("flags") or []:
            flags[str(flag)] += 1
        for key, value in (debug.get("counts") or {}).items():
            try:
                debug_counts[str(key)] += int(value)
            except (TypeError, ValueError):
                pass
        for tag in record.get("diag") or []:
            diag_tags[str(tag)] += 1
        if record.get("pressure"):
            pressure_turns += 1
        try:
            max_ms = max(max_ms, float(record.get("ms") or 0.0))
        except (TypeError, ValueError):
            pass
        sent_total += int(record.get("sent") or 0)
        if record.get("act") and len(first_actions) < 6:
            if first_action_step is None:
                first_action_step = record.get("step", record.get("_log_turn"))
            first_actions.append(
                {
                    "step": record.get("step", record.get("_log_turn")),
                    "act": record.get("act"),
                    "actx": record.get("actx"),
                    "actf": record.get("actf"),
                    "stage": debug.get("stage"),
                }
            )
        lead = record.get("lead") or []
        if len(lead) >= 2:
            try:
                max_prod_deficit = max(max_prod_deficit, max(0, -int(lead[0])))
                max_total_deficit = max(max_total_deficit, max(0, -int(lead[1])))
            except (TypeError, ValueError):
                pass
        try:
            logged_delta = int(record.get("prod_gap_delta") or 0)
            if logged_delta < 0:
                negative_logged_prod_delta += -logged_delta
                max_logged_prod_drop = max(max_logged_prod_drop, -logged_delta)
        except (TypeError, ValueError):
            logged_delta = 0
        try:
            threatened_prod = int(record.get("threatened_prod_sum") or 0)
            avoidable_swing = int(record.get("avoidable_prod_swing") or 0)
        except (TypeError, ValueError):
            threatened_prod = 0
            avoidable_swing = 0
        max_threatened_prod_sum = max(max_threatened_prod_sum, threatened_prod)
        max_avoidable_prod_swing = max(max_avoidable_prod_swing, avoidable_swing)
        if (avoidable_swing or threatened_prod) and len(avoidable_swing_turns) < 12:
            avoidable_swing_turns.append(
                {
                    "step": record.get("step", record.get("_log_turn")),
                    "prod_gap_delta": logged_delta,
                    "threatened_prod_sum": threatened_prod,
                    "avoidable_prod_swing": avoidable_swing,
                    "prod_swing": record.get("prod_swing"),
                    "lead": record.get("lead"),
                    "stage": debug.get("stage"),
                    "empty": record.get("empty"),
                    "actx": record.get("actx"),
                }
            )
        tags = record.get("diag") or []
        if tags and len(issue_turns) < 12:
            issue_turns.append(
                {
                    "step": record.get("step", record.get("_log_turn")),
                    "phase": record.get("phase"),
                    "tags": tags,
                    "empty": record.get("empty"),
                    "lead": record.get("lead"),
                    "src": record.get("src"),
                    "targets": record.get("targets"),
                    "actf": record.get("actf"),
                    "stage": debug.get("stage"),
                    "policy_fallback": debug.get("policy_fallback_budget"),
                    "top_missions": debug.get("top_missions"),
                    "route_trace": debug.get("route_trace"),
                    "mission_trace": debug.get("mission_trace"),
                    "timeout_trace": debug.get("timeout_trace"),
                    "samples": compact_debug_samples(debug),
                }
            )
        if debug.get("route_trace") and len(route_trace_turns) < 12:
            route_trace_turns.append(
                {
                    "step": record.get("step", record.get("_log_turn")),
                    "phase": record.get("phase"),
                    "stage": debug.get("stage"),
                    "lead": record.get("lead"),
                    "trace": debug.get("route_trace"),
                }
            )
        if debug.get("mission_trace") and len(mission_trace_turns) < 12:
            mission_trace_turns.append(
                {
                    "step": record.get("step", record.get("_log_turn")),
                    "stage": debug.get("stage"),
                    "lead": record.get("lead"),
                    "top_missions": debug.get("top_missions"),
                    "trace": debug.get("mission_trace"),
                }
            )
        if debug.get("timeout_trace") and len(timeout_trace_turns) < 12:
            timeout_trace_turns.append(
                {
                    "step": record.get("step", record.get("_log_turn")),
                    "stage": debug.get("stage"),
                    "lead": record.get("lead"),
                    "trace": debug.get("timeout_trace"),
                }
            )
        samples = debug.get("samples") or {}
        for sample in samples.get("boxed_accept") or []:
            if len(boxed_samples) < 6:
                boxed_samples.append({"step": record.get("step", record.get("_log_turn")), "sample": sample})

    return {
        "records": len(records),
        "empty": compact_counter(empty),
        "stages": compact_counter(stages),
        "phases": compact_counter(phases),
        "profiles": compact_counter(profiles),
        "flags": compact_counter(flags),
        "diag": compact_counter(diag_tags, limit=16),
        "debug_counts": compact_counter(debug_counts, limit=20),
        "first_actions": first_actions,
        "issue_turns": issue_turns,
        "route_trace_turns": route_trace_turns,
        "mission_trace_turns": mission_trace_turns,
        "timeout_trace_turns": timeout_trace_turns,
        "boxed_accepts": boxed_samples,
        "pressure_turns": pressure_turns,
        "sent_total": sent_total,
        "max_ms": round(max_ms, 2),
        "first_action_step": first_action_step,
        "max_prod_deficit": max_prod_deficit,
        "max_total_deficit": max_total_deficit,
        "negative_logged_prod_delta": negative_logged_prod_delta,
        "max_logged_prod_drop": max_logged_prod_drop,
        "max_threatened_prod_sum": max_threatened_prod_sum,
        "max_avoidable_prod_swing": max_avoidable_prod_swing,
        "avoidable_swing_turns": avoidable_swing_turns,
    }


def replay_observation_for_slot(episode, slot, turn):
    obs = copy.deepcopy(canonical_observation(episode, turn))
    obs["player"] = slot
    obs["step"] = turn
    obs.setdefault("remainingOverageTime", 60)
    return obs


def run_current_agent_turn(agent_module, episode, slot, turn):
    obs = replay_observation_for_slot(episode, slot, turn)
    config = episode.get("configuration") or {"actTimeout": 1.0}
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        actions = agent_module.agent(obs, config)

    current_record = None
    for line in (stderr_buffer.getvalue() + "\n" + stdout_buffer.getvalue()).splitlines():
        parsed = parse_owlog_line(line)
        if parsed is not None:
            current_record = parsed
    return actions or [], current_record or {}


def resample_current_agent(episode, slot, records, limit_per_reason, reasons):
    try:
        import submission as agent_module
    except Exception as exc:
        return {"error": f"could not import submission: {type(exc).__name__}: {exc}"}

    reasons = set(reasons or DEFAULT_RESAMPLE_REASONS)
    per_reason = Counter()
    selected = []
    max_turn = len(episode.get("steps", []))
    for record in records:
        reason = record.get("empty")
        if reason not in reasons:
            continue
        try:
            turn = int(record.get("step", record.get("_log_turn", -1)))
        except (TypeError, ValueError):
            continue
        if turn < 0 or turn >= max_turn:
            continue
        if per_reason[reason] >= limit_per_reason:
            continue
        selected.append((turn, reason))
        per_reason[reason] += 1

    summary = Counter()
    examples = []
    old_orbit_log = os.environ.get("ORBIT_LOG")
    old_orbit_every = os.environ.get("ORBIT_LOG_EVERY")
    os.environ["ORBIT_LOG"] = "1"
    os.environ["ORBIT_LOG_EVERY"] = "1"
    try:
        for turn, old_reason in selected:
            actions, current_record = run_current_agent_turn(agent_module, episode, slot, turn)
            debug = current_record.get("dbg") or {}
            current_reason = "act" if actions else (current_record.get("empty") or "no_log")
            summary[(old_reason, current_reason)] += 1
            if len(examples) < 12:
                examples.append(
                    {
                        "turn": turn,
                        "old": old_reason,
                        "current": current_reason,
                        "changed": bool(actions or current_reason != old_reason),
                        "actions": actions,
                        "stage": debug.get("stage"),
                        "top_counts": compact_counter(Counter(debug.get("counts") or {}), limit=6),
                        "top_missions": debug.get("top_missions"),
                        "route_trace": debug.get("route_trace"),
                        "mission_trace": debug.get("mission_trace"),
                        "timeout_trace": debug.get("timeout_trace"),
                        "samples": compact_debug_samples(debug),
                    }
                )
    finally:
        if old_orbit_log is None:
            os.environ.pop("ORBIT_LOG", None)
        else:
            os.environ["ORBIT_LOG"] = old_orbit_log
        if old_orbit_every is None:
            os.environ.pop("ORBIT_LOG_EVERY", None)
        else:
            os.environ["ORBIT_LOG_EVERY"] = old_orbit_every

    return {
        "sampled": len(selected),
        "reasons": sorted(reasons),
        "summary": [[old, new, count] for (old, new), count in summary.most_common()],
        "examples": examples,
    }


def detect_replay_data_issues(episode, slot):
    issues = []
    obs0 = canonical_observation(episode, 0)
    current_owned = [planet[0] for planet in obs0.get("planets", []) if int(planet[1]) == slot]
    initial = obs0.get("initial_planets") or []
    initial_by_id = {planet[0]: planet for planet in initial}
    if current_owned and initial:
        initial_home_owners = [initial_by_id.get(pid, [None, None])[1] for pid in current_owned]
        if any(owner != slot for owner in initial_home_owners):
            issues.append(
                {
                    "kind": "initial_planets_owner_mismatch",
                    "home_ids": current_owned,
                    "initial_home_owners": initial_home_owners,
                }
            )

    missing_step_slots = []
    if episode.get("steps"):
        first_step = episode["steps"][0]
        for idx, state in enumerate(first_step):
            obs = state.get("observation") if isinstance(state, dict) else None
            if isinstance(obs, dict) and obs.get("step") is None:
                missing_step_slots.append(idx)
    if missing_step_slots:
        issues.append({"kind": "slot_observation_missing_step", "slots": missing_step_slots})

    return issues


def ownership_events(episode, slot, limit=None):
    events = []
    obs0 = canonical_observation(episode, 0)
    initial = obs0.get("initial_planets") or obs0.get("planets") or []
    initial_by_id = {planet[0]: planet for planet in initial}
    previous = {planet[0]: int(planet[1]) for planet in obs0.get("planets", [])}

    for turn in range(1, len(episode.get("steps", []))):
        obs = canonical_observation(episode, turn)
        planets_by_id = {planet[0]: planet for planet in obs.get("planets", [])}
        current = {pid: int(planet[1]) for pid, planet in planets_by_id.items()}
        for planet_id, owner in current.items():
            old_owner = previous.get(planet_id)
            if old_owner == owner:
                continue
            if old_owner != slot and owner != slot:
                continue
            planet = planets_by_id[planet_id]
            initial_planet = initial_by_id.get(planet_id, planet)
            events.append(
                {
                    "turn": turn,
                    "planet": planet_id,
                    "change": "gain" if owner == slot else "loss",
                    "from": old_owner,
                    "to": owner,
                    "production": int(planet[6]),
                    "ships": int(planet[5]),
                    "x": round(float(planet[2]), 1),
                    "y": round(float(planet[3]), 1),
                    "initial_owner": int(initial_planet[1]),
                    "initial_ships": int(initial_planet[5]),
                }
            )
            if limit is not None and len(events) >= limit:
                return events
        previous = current
    return events


def checkpoints(episode, num_players):
    max_turn = len(episode.get("steps", [])) - 1
    turns = sorted({turn for turn in DEFAULT_CHECKPOINTS if 0 <= turn <= max_turn} | {max_turn})
    rows = []
    for turn in turns:
        obs = canonical_observation(episode, turn)
        table = score_table(obs, num_players)
        rows.append(
            {
                "turn": turn,
                "planets": table["planets"],
                "production": table["production"],
                "scores": table["scores"],
            }
        )
    return rows


def first_elimination_turn(episode, slot, num_players):
    if slot is None:
        return None
    for turn in range(len(episode.get("steps", []))):
        obs = canonical_observation(episode, turn)
        table = score_table(obs, num_players)
        if sum(table["scores"]) <= 0:
            continue
        if table["scores"][slot] <= 0:
            return turn
    return None


def build_notes(report):
    notes = []
    if report["result"] == "LOSS":
        notes.append("xite lost this episode")
    if report.get("elimination_turn") is not None:
        notes.append(f"xite eliminated at turn {report['elimination_turn']}")

    log = report.get("log", {})
    empty = dict(log.get("empty") or [])
    debug_counts = dict(log.get("debug_counts") or [])
    diag = dict(log.get("diag") or [])
    if empty.get("no_attack_budget", 0) or empty.get("locked_under_attack", 0):
        notes.append("many empty turns are budget/defense locks")
    if empty.get("full_reserve_no_immediate_risk", 0):
        notes.append("full reserve happened without immediate fall risk")
    if debug_counts.get("boxed_accept", 0):
        notes.append("boxed_breakout fired in this replay/log")
    if empty.get("all_candidates_filtered", 0) > max(20, log.get("records", 0) * 0.2):
        notes.append("candidate filters rejected many turns")
    if diag.get("opening_high_value_target_left", 0):
        notes.append("opening idled while a nearby high-value target was visible")
    if diag.get("core_source_drained", 0):
        notes.append("core/home source was drained to a low garrison")
    if diag.get("over_reserved_or_budget_locked", 0):
        notes.append("attack budget was repeatedly locked without an immediate fall signal")
    if diag.get("multi_eta_failed", 0) or diag.get("multi_mass_failed", 0):
        notes.append("multi-source attack assembly often failed")
    if diag.get("routing_rejections", 0):
        notes.append("routing rejected candidate shots")
    if diag.get("time_pressure", 0):
        notes.append("planner hit soft time pressure")
    prod = report.get("production") or {}
    if (prod.get("max_drop_10") or {}).get("drop", 0) >= 12:
        notes.append("production gap fell sharply within 10 turns")
    if prod.get("negative_delta_sum", 0) >= 60:
        notes.append("large cumulative production swing against xite")

    for issue in report.get("data_issues", []):
        if issue["kind"] == "initial_planets_owner_mismatch":
            notes.append("episode initial_planets owner does not mark homes; use step-0 planets for home ownership")
        if issue["kind"] == "slot_observation_missing_step":
            notes.append("some slot observations miss step; use slot-0 observation plus player override for replay simulation")
    return notes


def aggregate_reports(reports):
    result_counts = Counter(report["result"] for report in reports)
    note_counts = Counter()
    empty_counts = Counter()
    diag_counts = Counter()
    stage_counts = Counter()
    profile_counts = Counter()
    total_records = 0
    loss_reports = []
    total_negative_prod_delta = 0
    worst_drop_10 = 0
    worst_drop_20 = 0
    worst_drop_episode = None
    max_avoidable_prod_swing = 0

    for report in reports:
        log = report.get("log") or {}
        prod = report.get("production") or {}
        total_records += int(log.get("records") or 0)
        total_negative_prod_delta += int(prod.get("negative_delta_sum") or 0)
        drop10 = int((prod.get("max_drop_10") or {}).get("drop") or 0)
        drop20 = int((prod.get("max_drop_20") or {}).get("drop") or 0)
        if drop10 > worst_drop_10:
            worst_drop_10 = drop10
            worst_drop_episode = report["episode"]
        worst_drop_20 = max(worst_drop_20, drop20)
        max_avoidable_prod_swing = max(
            max_avoidable_prod_swing,
            int(log.get("max_avoidable_prod_swing") or 0),
        )
        for note in report.get("notes") or []:
            note_counts[note] += 1
        for key, value in log.get("empty") or []:
            empty_counts[key] += int(value)
        for key, value in log.get("diag") or []:
            diag_counts[key] += int(value)
        for key, value in log.get("stages") or []:
            stage_counts[key] += int(value)
        for key, value in log.get("profiles") or []:
            profile_counts[key] += int(value)
        if report["result"] == "LOSS":
            loss_reports.append(
                {
                    "episode": report["episode"],
                    "rank": report["rank"],
                    "steps": report["steps"],
                    "first_action_step": log.get("first_action_step"),
                    "max_prod_deficit": log.get("max_prod_deficit"),
                    "max_total_deficit": log.get("max_total_deficit"),
                    "negative_prod_delta": prod.get("negative_delta_sum"),
                    "max_drop_10": prod.get("max_drop_10"),
                    "top_notes": report.get("notes", [])[:4],
                }
            )

    return {
        "episodes": len(reports),
        "results": compact_counter(result_counts),
        "records": total_records,
        "empty": compact_counter(empty_counts, limit=10),
        "diag": compact_counter(diag_counts, limit=14),
        "stages": compact_counter(stage_counts, limit=10),
        "profiles": compact_counter(profile_counts, limit=8),
        "notes": compact_counter(note_counts, limit=10),
        "prod": {
            "negative_delta_sum": total_negative_prod_delta,
            "worst_drop_10": worst_drop_10,
            "worst_drop_20": worst_drop_20,
            "worst_drop_episode": worst_drop_episode,
            "max_avoidable_prod_swing": max_avoidable_prod_swing,
        },
        "losses": loss_reports[:12],
    }


def analyze_episode(
    episode_path,
    replay_dir,
    player_name,
    event_limit,
    resample_current=False,
    resample_limit=8,
    resample_reasons=None,
):
    episode = load_json(episode_path)
    episode_id = episode_id_from_path(episode_path)
    match_manifest = load_match_manifest(replay_dir, episode_id)
    names = team_names(episode)
    num_players = len(episode.get("steps", [[]])[0])
    slot, slot_source = resolve_player_slot(
        episode,
        replay_dir,
        episode_id,
        player_name,
        match_manifest,
    )
    rewards = episode.get("rewards") or []
    final_obs = canonical_observation(episode, len(episode["steps"]) - 1)
    final_scores = score_table(final_obs, num_players)["scores"]
    xite_score = final_scores[slot] if slot is not None else 0
    rank = None if slot is None else 1 + sum(score > xite_score for score in final_scores)
    log_path = replay_dir / f"{episode_id}-{slot}.json" if episode_id is not None and slot is not None else None
    records = load_log_records(log_path)

    report = {
        "episode": episode_id,
        "episode_path": str(episode_path),
        "match": match_manifest,
        "map": (match_manifest or {}).get("map") or map_signature_from_episode(episode),
        "log_path": str(log_path) if log_path and log_path.exists() else None,
        "player_name": player_name,
        "slot": slot,
        "slot_source": slot_source,
        "teams": names,
        "steps": len(episode.get("steps", [])),
        "rewards": rewards,
        "result": player_result(slot, rewards),
        "rank": rank,
        "final_scores": final_scores,
        "final_planets": score_table(final_obs, num_players)["planets"],
        "final_production": score_table(final_obs, num_players)["production"],
        "checkpoints": checkpoints(episode, num_players),
        "ownership_events": ownership_events(episode, slot, limit=event_limit) if slot is not None else [],
        "production": production_drop_summary(episode, slot, num_players),
        "production_swing_events": production_swing_events(episode, slot, limit=event_limit),
        "elimination_turn": first_elimination_turn(episode, slot, num_players),
        "data_issues": detect_replay_data_issues(episode, slot),
        "log": analyze_log_records(records),
    }
    if resample_current and slot is not None:
        report["current_resample"] = resample_current_agent(
            episode,
            slot,
            records,
            limit_per_reason=resample_limit,
            reasons=resample_reasons,
        )
    report["notes"] = build_notes(report)
    return report


def discover_episode_paths(replay_dir, episode_ids):
    if episode_ids:
        paths = []
        for episode_id in episode_ids:
            path = replay_dir / f"episode-{episode_id}.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing episode file: {path}")
            paths.append(path)
        return paths
    return sorted(replay_dir.glob("episode-*.json"))


def format_vector(values):
    return "[" + ",".join(str(value) for value in values) + "]"


def print_report(report, event_limit, checkpoint_limit):
    slot_label = "?" if report["slot"] is None else f"s{report['slot']}"
    print(
        f"{report['episode']} | {report['player_name']}={slot_label} "
        f"slot_source={report.get('slot_source', 'unknown')} | "
        f"{report['result']} | rank={report['rank']} | steps={report['steps']} | "
        f"final={format_vector(report['final_scores'])}"
    )
    print("  teams: " + " | ".join(f"s{idx}={name}" for idx, name in enumerate(report["teams"])))
    match = report.get("match") or {}
    if match:
        opponents = ", ".join(
            f"s{slot}:{Path(path).name}"
            for slot, path in sorted(
                ((int(slot), path) for slot, path in (match.get("opponents") or {}).items()),
                key=lambda item: item[0],
            )
        )
        match_map = match.get("map") or {}
        print(
            "  match: "
            f"seed={match.get('seed')} "
            f"my=s{match.get('my_slot')} "
            f"match_key={match.get('match_key')} "
            f"match={match.get('match_hash')} "
            f"map={match_map.get('hash')} "
            f"opp=[{opponents}]"
        )
    else:
        match_map = report.get("map") or {}
        if match_map.get("hash"):
            print(f"  match: manifest=missing map={match_map.get('hash')}")
    if report["log_path"]:
        print(f"  log: {report['log_path']} records={report['log']['records']} sent={report['log']['sent_total']}")
    else:
        print("  log: missing")
    print(
        "  final territory: planets="
        + format_vector(report["final_planets"])
        + " prod="
        + format_vector(report["final_production"])
    )
    prod = report.get("production") or {}
    if prod:
        print(
            "  prod drops: "
            f"neg_sum={prod.get('negative_delta_sum')} "
            f"max_delta={prod.get('max_negative_delta')} "
            f"drop5={prod.get('max_drop_5', {}).get('drop')} "
            f"drop10={prod.get('max_drop_10', {}).get('drop')} "
            f"drop20={prod.get('max_drop_20', {}).get('drop')} "
            f"final_gap={prod.get('final_gap')}"
        )

    if report["data_issues"]:
        issue_text = "; ".join(issue["kind"] for issue in report["data_issues"])
        print(f"  data issues: {issue_text}")

    log = report["log"]
    print(f"  empty: {log['empty']}")
    print(f"  stages: {log['stages']}")
    print(f"  diag: {log['diag']}")
    print(f"  debug: {log['debug_counts']}")
    if (
        log.get("negative_logged_prod_delta")
        or log.get("max_avoidable_prod_swing")
        or log.get("max_threatened_prod_sum")
    ):
        print(
            "  log prod: "
            f"neg_delta={log.get('negative_logged_prod_delta')} "
            f"max_drop={log.get('max_logged_prod_drop')} "
            f"max_threatened_prod={log.get('max_threatened_prod_sum')} "
            f"max_avoidable_swing={log.get('max_avoidable_prod_swing')}"
        )
    if log["issue_turns"]:
        print(f"  issue turns: {log['issue_turns'][:event_limit]}")
    if log.get("avoidable_swing_turns"):
        print(f"  avoidable swing turns: {log['avoidable_swing_turns'][:event_limit]}")
    if log["route_trace_turns"]:
        print(f"  route traces: {log['route_trace_turns'][:event_limit]}")
    if log["mission_trace_turns"]:
        print(f"  mission traces: {log['mission_trace_turns'][:event_limit]}")
    if log["timeout_trace_turns"]:
        print(f"  timeout traces: {log['timeout_trace_turns'][:event_limit]}")
    if log["boxed_accepts"]:
        print(f"  boxed accepts: {log['boxed_accepts'][:event_limit]}")
    if log["first_actions"]:
        print(f"  first actions: {log['first_actions'][:event_limit]}")

    current_resample = report.get("current_resample")
    if current_resample:
        if current_resample.get("error"):
            print(f"  current resample: {current_resample['error']}")
        else:
            print(f"  current resample: sampled={current_resample['sampled']} summary={current_resample['summary']}")
            if current_resample.get("examples"):
                print(f"  current examples: {current_resample['examples'][:event_limit]}")

    if checkpoint_limit:
        print("  checkpoints:")
        for row in report["checkpoints"][:checkpoint_limit]:
            print(
                f"    t={row['turn']:>3} planets={format_vector(row['planets'])} "
                f"prod={format_vector(row['production'])} scores={format_vector(row['scores'])}"
            )

    events = report["ownership_events"]
    if events:
        print("  xite ownership swings:")
        for event in events[:event_limit]:
            sign = "+" if event["change"] == "gain" else "-"
            print(
                f"    {sign} t={event['turn']:>3} p{event['planet']} "
                f"{event['from']}->{event['to']} prod={event['production']} "
                f"ships={event['ships']} pos=({event['x']},{event['y']})"
            )
    swing_events = report.get("production_swing_events") or []
    if swing_events:
        print("  production swing events:")
        for event in swing_events[:event_limit]:
            print(
                f"    t={event['turn']:>3} p{event['planet']} "
                f"{event['from']}->{event['to']} prod={event['production']} "
                f"gap_delta={event['prod_gap_delta']}"
            )

    if report["notes"]:
        print("  notes: " + "; ".join(report["notes"]))
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze Orbit Wars episode replays and OWLOG files.")
    parser.add_argument(
        "--replay-dir",
        default="replay",
        help="Saved experiment directory containing episode-*.json and *-slot.json logs.",
    )
    parser.add_argument("--episode", nargs="*", help="Episode ids to analyze. Defaults to all episodes in replay-dir.")
    parser.add_argument(
        "--player-name",
        default="xite",
        help="Player/team name to analyze when no manifest-*.json my_slot is available.",
    )
    parser.add_argument("--events", type=int, default=12, help="Maximum ownership/action events to print per episode.")
    parser.add_argument("--checkpoints", type=int, default=8, help="Maximum checkpoint rows to print per episode.")
    parser.add_argument(
        "--resample-current",
        action="store_true",
        help="Run the current submission.py on sampled historical empty turns.",
    )
    parser.add_argument(
        "--resample-limit",
        type=int,
        default=8,
        help="Maximum old empty turns to resample per reason per episode.",
    )
    parser.add_argument(
        "--resample-reason",
        nargs="*",
        default=None,
        help="Old empty reasons to resample. Defaults to common strategic failure reasons.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text.")
    args = parser.parse_args()

    replay_dir = Path(args.replay_dir)
    reports = [
        analyze_episode(
            path,
            replay_dir,
            args.player_name,
            args.events,
            resample_current=args.resample_current,
            resample_limit=args.resample_limit,
            resample_reasons=args.resample_reason,
        )
        for path in discover_episode_paths(replay_dir, args.episode)
    ]

    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return

    summary = aggregate_reports(reports)
    print(f"Replay dir: {replay_dir}")
    print(f"Episodes: {len(reports)}")
    print(f"Aggregate results: {summary['results']}")
    print(f"Aggregate empty: {summary['empty']}")
    print(f"Aggregate diag: {summary['diag']}")
    print(f"Aggregate prod: {summary['prod']}")
    if summary["notes"]:
        print(f"Aggregate notes: {summary['notes']}")
    if summary["losses"]:
        print(f"Loss focus: {summary['losses']}")
    print()
    for report in reports:
        print_report(report, args.events, args.checkpoints)


if __name__ == "__main__":
    main()
