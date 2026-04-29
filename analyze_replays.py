import argparse
import contextlib
import copy
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


def load_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def episode_id_from_path(path):
    match = re.search(r"episode-(\d+)\.json$", Path(path).name)
    return match.group(1) if match else None


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


def player_result(slot, rewards):
    if slot is None or slot >= len(rewards):
        return "UNKNOWN"
    winners = [idx for idx, reward in enumerate(rewards) if reward == 1]
    if slot not in winners:
        return "LOSS"
    return "WIN" if len(winners) == 1 else "TIE_WIN"


def compact_counter(counter, limit=6):
    return [[key, value] for key, value in counter.most_common(limit)]


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
    profiles = Counter()
    flags = Counter()
    debug_counts = Counter()
    first_actions = []
    boxed_samples = []
    pressure_turns = 0
    max_ms = 0.0
    sent_total = 0

    for record in records:
        if record.get("empty"):
            empty[record.get("empty")] += 1
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
        if record.get("pressure"):
            pressure_turns += 1
        try:
            max_ms = max(max_ms, float(record.get("ms") or 0.0))
        except (TypeError, ValueError):
            pass
        sent_total += int(record.get("sent") or 0)
        if record.get("act") and len(first_actions) < 6:
            first_actions.append(
                {
                    "step": record.get("step", record.get("_log_turn")),
                    "act": record.get("act"),
                    "actx": record.get("actx"),
                    "stage": debug.get("stage"),
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
        "profiles": compact_counter(profiles),
        "flags": compact_counter(flags),
        "debug_counts": compact_counter(debug_counts, limit=10),
        "first_actions": first_actions,
        "boxed_accepts": boxed_samples,
        "pressure_turns": pressure_turns,
        "sent_total": sent_total,
        "max_ms": round(max_ms, 2),
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
            if len(examples) < 12 and (actions or current_reason != old_reason):
                examples.append(
                    {
                        "turn": turn,
                        "old": old_reason,
                        "current": current_reason,
                        "actions": actions,
                        "stage": debug.get("stage"),
                        "top_counts": compact_counter(Counter(debug.get("counts") or {}), limit=6),
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
    if empty.get("no_attack_budget", 0) or empty.get("locked_under_attack", 0):
        notes.append("many empty turns are budget/defense locks")
    if empty.get("full_reserve_no_immediate_risk", 0):
        notes.append("full reserve happened without immediate fall risk")
    if debug_counts.get("boxed_accept", 0):
        notes.append("boxed_breakout fired in this replay/log")
    if empty.get("all_candidates_filtered", 0) > max(20, log.get("records", 0) * 0.2):
        notes.append("candidate filters rejected many turns")

    for issue in report.get("data_issues", []):
        if issue["kind"] == "initial_planets_owner_mismatch":
            notes.append("episode initial_planets owner does not mark homes; use step-0 planets for home ownership")
        if issue["kind"] == "slot_observation_missing_step":
            notes.append("some slot observations miss step; use slot-0 observation plus player override for replay simulation")
    return notes


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
    names = team_names(episode)
    num_players = len(episode.get("steps", [[]])[0])
    slot = find_player_slot(episode, player_name)
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
        "log_path": str(log_path) if log_path and log_path.exists() else None,
        "player_name": player_name,
        "slot": slot,
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
        f"{report['episode']} | {report['player_name']}={slot_label} | "
        f"{report['result']} | rank={report['rank']} | steps={report['steps']} | "
        f"final={format_vector(report['final_scores'])}"
    )
    print("  teams: " + " | ".join(f"s{idx}={name}" for idx, name in enumerate(report["teams"])))
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

    if report["data_issues"]:
        issue_text = "; ".join(issue["kind"] for issue in report["data_issues"])
        print(f"  data issues: {issue_text}")

    log = report["log"]
    print(f"  empty: {log['empty']}")
    print(f"  stages: {log['stages']}")
    print(f"  debug: {log['debug_counts']}")
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

    if report["notes"]:
        print("  notes: " + "; ".join(report["notes"]))
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze Orbit Wars episode replays and OWLOG files.")
    parser.add_argument("--replay-dir", default="replay", help="Directory containing episode-*.json and *-slot.json logs.")
    parser.add_argument("--episode", nargs="*", help="Episode ids to analyze. Defaults to all episodes in replay-dir.")
    parser.add_argument("--player-name", default="xite", help="Player/team name to analyze.")
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

    print(f"Replay dir: {replay_dir}")
    print(f"Episodes: {len(reports)}")
    print()
    for report in reports:
        print_report(report, args.events, args.checkpoints)


if __name__ == "__main__":
    main()
