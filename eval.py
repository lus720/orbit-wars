import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import html as html_lib
import inspect
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import time

from kaggle_environments import make


PLAYER_COLORS = (
    ("blue", "#0072B2"),
    ("orange", "#E69F00"),
    ("green/teal", "#009E73"),
    ("yellow", "#F0E442"),
)
NUM_PLAYERS = 2
DEFAULT_SLOT_BASE_SEED = 42
OWLOG_PREFIX = "OWLOG "


def load_agent(path, module_name):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import agent from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    agent = getattr(module, "agent", None)
    if agent is None:
        raise RuntimeError(f"{path} does not define an agent function")
    return agent, path


class AgentRecorder:
    def __init__(self, agent, log_enabled=False, log_every=1, random_seed=None):
        self.agent = agent
        self.log_enabled = log_enabled
        self.log_every = max(1, int(log_every))
        self.records = []
        self.random_state = None
        if random_seed is not None:
            rng = random.Random(int(random_seed))
            self.random_state = rng.getstate()
        try:
            parameters = list(inspect.signature(agent).parameters.values())
            positional = [
                parameter
                for parameter in parameters
                if parameter.kind
                in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            ]
            self.accepts_config = len(positional) >= 2 or any(
                parameter.kind == parameter.VAR_POSITIONAL for parameter in parameters
            )
        except (TypeError, ValueError):
            self.accepts_config = True

    def __call__(self, obs, config=None):
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        old_log = os.environ.get("ORBIT_LOG")
        old_every = os.environ.get("ORBIT_LOG_EVERY")
        if self.log_enabled:
            os.environ["ORBIT_LOG"] = "1"
            os.environ["ORBIT_LOG_EVERY"] = str(self.log_every)
        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                old_random_state = None
                if self.random_state is not None:
                    old_random_state = random.getstate()
                    random.setstate(self.random_state)
                try:
                    if self.accepts_config:
                        return self.agent(obs, config)
                    return self.agent(obs)
                finally:
                    if self.random_state is not None:
                        self.random_state = random.getstate()
                        random.setstate(old_random_state)
        finally:
            self.records.append(
                {
                    "stdout": stdout_buffer.getvalue(),
                    "stderr": stderr_buffer.getvalue(),
                }
            )
            if self.log_enabled:
                if old_log is None:
                    os.environ.pop("ORBIT_LOG", None)
                else:
                    os.environ["ORBIT_LOG"] = old_log
                if old_every is None:
                    os.environ.pop("ORBIT_LOG_EVERY", None)
                else:
                    os.environ["ORBIT_LOG_EVERY"] = old_every


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def parse_owlog_payload(raw_payload, record_index, stream_name):
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        payload = {
            "_parse_error": str(exc),
            "_raw": raw_payload,
        }
    if isinstance(payload, dict):
        payload = dict(payload)
        payload["_record"] = record_index
        payload["_stream"] = stream_name
        return payload
    return {
        "payload": payload,
        "_record": record_index,
        "_stream": stream_name,
    }


def split_stream_capture(text, record_index, stream_name):
    ordinary_parts = []
    owlog_rows = []
    for line in (text or "").splitlines(keepends=True):
        line_text = line.rstrip("\r\n")
        if line_text.startswith(OWLOG_PREFIX):
            owlog_rows.append(
                parse_owlog_payload(
                    line_text[len(OWLOG_PREFIX) :],
                    record_index,
                    stream_name,
                )
            )
        else:
            ordinary_parts.append(line)
    return "".join(ordinary_parts), owlog_rows


def build_agent_log(records):
    rows = []
    for record_index, record in enumerate(records):
        stdout, stdout_owlog = split_stream_capture(
            record.get("stdout") or "",
            record_index,
            "stdout",
        )
        stderr, stderr_owlog = split_stream_capture(
            record.get("stderr") or "",
            record_index,
            "stderr",
        )
        rows.append(
            {
                "record": record_index,
                "stdout": stdout,
                "stderr": stderr,
                "owlog": stdout_owlog + stderr_owlog,
            }
        )
    return rows


def path_is_relative_to(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def resolve_save_dir(save_root, run_name):
    root = Path(save_root)
    if run_name in (None, "", "."):
        return root
    save_dir = root / Path(run_name)
    if not path_is_relative_to(save_dir, root):
        raise ValueError("--run-name must stay under --save-root")
    return save_dir


def resolve_summary_json(summary_json, save_dir, save_root):
    if not summary_json:
        return None
    path = Path(summary_json)
    if not path.is_absolute() and (
        not path.parts or path.parts[0] not in ("output", "replay")
    ):
        path = Path(save_dir) / path
    if not path_is_relative_to(path, save_root):
        raise ValueError(
            f"--summary-json must be saved under selected --save-root={save_root}: {path}"
        )
    return path


def relative_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def stable_json_hash(payload):
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]


def deterministic_seed_token(*parts):
    return stable_json_hash(list(parts))


def deterministic_seed_value(*parts):
    return int(deterministic_seed_token(*parts), 16)


def round_map_value(value):
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, list):
        return [round_map_value(item) for item in value]
    if isinstance(value, dict):
        return {key: round_map_value(value[key]) for key in sorted(value)}
    return value


def stable_sorted(values):
    return sorted(
        values,
        key=lambda value: json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def map_signature_from_episode(episode):
    resolved_seed = resolved_map_seed_from_episode(episode)
    for step in episode.get("steps", []):
        if not step:
            continue
        state = step[0]
        observation = state.get("observation", {})
        planets = observation.get("planets") or []
        if not planets:
            continue
        payload = {
            "seed": resolved_seed,
            "planets": stable_sorted(round_map_value(planet) for planet in planets),
            "initial_planets": stable_sorted(
                round_map_value(planet)
                for planet in observation.get("initial_planets", [])
            ),
            "angular_velocity": round_map_value(observation.get("angular_velocity")),
            "comets": stable_sorted(
                round_map_value(comet) for comet in observation.get("comets", [])
            ),
            "comet_planet_ids": sorted(observation.get("comet_planet_ids", [])),
        }
        return {"hash": stable_json_hash(payload), "step": observation.get("step")}
    return {"hash": None, "step": None}


def resolved_map_seed_from_episode(episode, fallback=None):
    info_seed = episode.get("info", {}).get("seed")
    if info_seed is not None:
        return int(info_seed)
    configuration = episode.get("configuration", {})
    for key in ("seed", "randomSeed"):
        value = configuration.get(key)
        if value is not None:
            return int(value)
    if fallback is not None:
        return int(fallback)
    return None


def scrub_unsupported_seed_config(env, map_seed):
    if "seed" in env.specification.configuration:
        return
    if env.configuration.get("seed") == map_seed:
        env.configuration.seed = None


def seed_stable_slot(seed, slot_base_seed):
    return (int(seed) - int(slot_base_seed)) % NUM_PLAYERS


def choose_my_slot(game_index, seed, match_key, slot_base_seed, fixed_my_slot):
    if fixed_my_slot is not None:
        return int(fixed_my_slot)
    if match_key == "seed":
        return seed_stable_slot(seed, slot_base_seed)
    return int(game_index) % NUM_PLAYERS


def player_color(slot):
    if 0 <= int(slot) < len(PLAYER_COLORS):
        return PLAYER_COLORS[int(slot)]
    return ("unknown", "#888888")


def swatch_html(color):
    escaped = html_lib.escape(color)
    return (
        '<span style="display:inline-block;width:12px;height:12px;'
        f'border:1px solid #222;background:{escaped};vertical-align:-1px"></span>'
    )


def player_token_html(slot):
    _color_name, color_hex = player_color(slot)
    return f"{swatch_html(color_hex)} player {int(slot)}"


def inject_replay_banner(rendered_html, banner):
    script = f"""
<script>
(function() {{
  const bannerHtml = {json.dumps(banner)};
  function installCodexBanner() {{
    const existing = document.getElementById("codex-my-agent-banner");
    if (existing) return;
    const container = document.createElement("div");
    container.innerHTML = bannerHtml;
    document.body.appendChild(container.firstElementChild);
  }}
  installCodexBanner();
  window.addEventListener("load", installCodexBanner);
  setTimeout(installCodexBanner, 0);
  setTimeout(installCodexBanner, 250);
  setTimeout(installCodexBanner, 1000);
}})();
</script>
""".strip()
    body_end = rendered_html.lower().rfind("</body>")
    if body_end >= 0:
        return rendered_html[:body_end] + "\n" + script + "\n" + rendered_html[body_end:]
    return rendered_html + "\n" + script


def replay_banner_html(my_slot, my_agent_path, result, scores, rewards, seed, game):
    _color_name, color_hex = player_color(my_slot)
    winners = [slot for slot, reward in enumerate(rewards) if reward == 1]
    winner_html = ", ".join(player_token_html(slot) for slot in winners) or "none"
    score_html = ", ".join(
        f"{player_token_html(slot)}={float(score):.0f}" for slot, score in enumerate(scores)
    )
    agent_text = html_lib.escape(relative_path(my_agent_path))
    banner = f"""
<div id="codex-my-agent-banner" style="position:fixed;top:0;left:0;bottom:44px;width:min(340px,28vw);z-index:2147483647;pointer-events:none;background:rgba(17,17,17,.94);color:#fff;padding:12px 14px;font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;border-right:3px solid {html_lib.escape(color_hex)};box-shadow:2px 0 10px rgba(0,0,0,.35);box-sizing:border-box;overflow:hidden;overflow-wrap:anywhere">
  <div><strong>MY METHOD</strong>: {swatch_html(color_hex)} player {int(my_slot)} / <code>{agent_text}</code></div>
  <div>seed={int(seed)} game={int(game)} result={html_lib.escape(str(result))} winners={winner_html}</div>
  <div>scores: {score_html}</div>
</div>
""".strip()
    return banner


def render_replay_with_my_banner(env, my_slot, my_agent_path, result, scores, rewards, seed, game):
    rendered = env.render(mode="html")
    banner = replay_banner_html(my_slot, my_agent_path, result, scores, rewards, seed, game)
    return inject_replay_banner(rendered, banner)


def result_label(my_reward, baseline_reward):
    # Use the environment's official outcome directly:
    # reward == 1 means this player is one of the winner(s), reward == -1 otherwise.
    if my_reward == 1 and baseline_reward != 1:
        return "WIN"
    if my_reward != 1 and baseline_reward == 1:
        return "LOSS"
    return "TIE"


def final_ship_scores(observation, num_agents):
    scores = [0.0] * num_agents
    for planet in observation.get("planets", []):
        owner = planet[1]
        if owner != -1 and 0 <= owner < num_agents:
            scores[owner] += planet[5]
    for fleet in observation.get("fleets", []):
        owner = fleet[1]
        if 0 <= owner < num_agents:
            scores[owner] += fleet[6]
    return scores


def run_game(
    my_agent,
    baseline_agent,
    my_agent_path,
    baseline_agent_path,
    game_index,
    seed,
    my_slot,
    episode_steps,
    act_timeout,
    save_replays,
    save_artifacts,
    save_dir,
    log_every,
):
    my_slot = int(my_slot)
    baseline_slot = 1 - my_slot
    map_seed = int(seed)
    agent_paths_by_slot = [None, None]
    agent_paths_by_slot[my_slot] = relative_path(my_agent_path)
    agent_paths_by_slot[baseline_slot] = relative_path(baseline_agent_path)
    agent_seed_tokens_by_slot = [
        deterministic_seed_token("agent", map_seed, slot, path)
        for slot, path in enumerate(agent_paths_by_slot)
    ]

    capture_logs = save_artifacts != "none"
    my_recorder = AgentRecorder(
        my_agent,
        log_enabled=capture_logs,
        log_every=log_every,
        random_seed=int(agent_seed_tokens_by_slot[my_slot], 16),
    )
    baseline_recorder = AgentRecorder(
        baseline_agent,
        log_enabled=False,
        log_every=log_every,
        random_seed=int(agent_seed_tokens_by_slot[baseline_slot], 16),
    )

    agents = [None, None]
    agents[my_slot] = my_recorder
    agents[baseline_slot] = baseline_recorder

    random.seed(map_seed)
    start_time = time.perf_counter()
    config = {
        "episodeSteps": episode_steps,
        "seed": map_seed,
        "randomSeed": map_seed,
    }
    if act_timeout is not None:
        config["actTimeout"] = act_timeout
    env = make("orbit_wars", config, debug=True)
    scrub_unsupported_seed_config(env, map_seed)
    random.seed(map_seed)
    env.run(agents)
    elapsed = time.perf_counter() - start_time
    episode_json = env.toJSON()
    resolved_map_seed = resolved_map_seed_from_episode(episode_json, map_seed)
    map_signature = map_signature_from_episode(episode_json)
    match_metadata = {
        "seed": map_seed,
        "map_seed": resolved_map_seed,
        "my_slot": my_slot,
        "baseline_slot": baseline_slot,
        "agents": agent_paths_by_slot,
        "agent_seed_tokens": agent_seed_tokens_by_slot,
        "map": map_signature,
    }
    match_metadata["match_hash"] = stable_json_hash(match_metadata)
    match_metadata["agent_hash"] = stable_json_hash(agent_paths_by_slot)

    final = env.steps[-1]
    my_state = final[my_slot]
    baseline_state = final[baseline_slot]
    my_reward = my_state.reward
    baseline_reward = baseline_state.reward
    slot_rewards = [state.reward for state in final]
    result = result_label(my_reward, baseline_reward)
    scores = final_ship_scores(final[0].observation, len(final))
    my_score = scores[my_slot]
    baseline_score = scores[baseline_slot]
    replay_path = None
    should_save_replay = save_replays == "all" or (
        save_replays == "loss" and result == "LOSS"
    )
    if should_save_replay:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        replay_path = save_dir / f"seed{seed}.html"
        replay_path.write_text(
            render_replay_with_my_banner(
                env,
                my_slot,
                my_agent_path,
                result,
                scores,
                slot_rewards,
                seed,
                game_index + 1,
            ),
            encoding="utf-8",
        )

    artifact_episode_path = None
    artifact_log_path = None
    artifact_manifest_path = None
    should_save_artifacts = save_artifacts == "all" or (
        save_artifacts == "loss" and result == "LOSS"
    )
    if should_save_artifacts:
        artifact_root = Path(save_dir)
        artifact_episode_path = artifact_root / f"seed{seed}-replay.json"
        artifact_log_path = artifact_root / f"seed{seed}-log.json"
        artifact_manifest_path = artifact_root / f"seed{seed}-manifest.json"
        write_json(artifact_episode_path, episode_json)
        write_json(artifact_log_path, build_agent_log(my_recorder.records))
        write_json(artifact_manifest_path, match_metadata)

    return {
        "game": game_index + 1,
        "seed": seed,
        "map_seed": resolved_map_seed,
        "map_hash": map_signature["hash"],
        "agent_hash": match_metadata["agent_hash"],
        "match_hash": match_metadata["match_hash"],
        "agents": agent_paths_by_slot,
        "agent_seed_tokens": agent_seed_tokens_by_slot,
        "my_slot": my_slot,
        "baseline_slot": baseline_slot,
        "my_reward": my_reward,
        "baseline_reward": baseline_reward,
        "my_score": my_score,
        "baseline_score": baseline_score,
        "my_status": my_state.status,
        "baseline_status": baseline_state.status,
        "result": result,
        "replay_path": str(replay_path) if replay_path is not None else None,
        "artifact_episode_path": str(artifact_episode_path) if artifact_episode_path is not None else None,
        "artifact_log_path": str(artifact_log_path) if artifact_log_path is not None else None,
        "artifact_manifest_path": str(artifact_manifest_path) if artifact_manifest_path is not None else None,
        "steps": len(env.steps),
        "elapsed": elapsed,
    }


def run_game_worker(
    my_agent_path,
    baseline_agent_path,
    game_index,
    seed,
    my_slot,
    episode_steps,
    act_timeout,
    save_replays,
    save_artifacts,
    save_dir,
    log_every,
):
    my_slot = int(my_slot)
    baseline_slot = 1 - my_slot
    random.seed(
        deterministic_seed_value(
            "agent-import",
            int(seed),
            my_slot,
            relative_path(my_agent_path),
        )
    )
    my_agent, _ = load_agent(my_agent_path, f"my_submission_agent_{game_index}")
    random.seed(
        deterministic_seed_value(
            "agent-import",
            int(seed),
            baseline_slot,
            relative_path(baseline_agent_path),
        )
    )
    baseline_agent, _ = load_agent(
        baseline_agent_path,
        f"baseline_old_version_agent_{game_index}",
    )
    return run_game(
        my_agent,
        baseline_agent,
        my_agent_path,
        baseline_agent_path,
        game_index,
        seed,
        my_slot,
        episode_steps,
        act_timeout,
        save_replays,
        save_artifacts,
        save_dir,
        log_every,
    )


def parse_seeds(seed_args, seed_start):
    if seed_args:
        seeds = []
        for item in seed_args:
            for part in item.split(","):
                part = part.strip()
                if part:
                    seeds.append(int(part))
        return seeds
    return [int(seed_start)]


def main():
    parser = argparse.ArgumentParser(
        description="Run Orbit Wars matches between submission.py and a baseline."
    )
    parser.add_argument("--my-agent", default="submission.py")
    parser.add_argument("--baseline-agent", default="baselines/mine_old_version.py")
    parser.add_argument(
        "--seeds",
        nargs="*",
        help="Explicit seeds, e.g. --seeds 42 43 44 or --seeds 42,43,44.",
    )
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument(
        "--act-timeout",
        type=float,
        default=2.0,
        help="Kaggle environment actTimeout seconds. Use a large value for no practical local timeout.",
    )
    parser.add_argument(
        "--match-key",
        choices=("seed", "order"),
        default="seed",
        help=(
            "Use seed-stable player slot by default. Use order to reproduce "
            "the older behavior where list order determined the slot; order "
            "mode intentionally breaks the seed-fixed agent-slot guarantee."
        ),
    )
    parser.add_argument(
        "--slot-base-seed",
        type=int,
        default=DEFAULT_SLOT_BASE_SEED,
        help="Base seed for seed-stable slot rotation: slot=(seed-base) mod 2.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker count. Defaults to min(number of seeds, 8, CPU count).",
    )
    parser.add_argument(
        "--my-slot",
        type=int,
        choices=range(NUM_PLAYERS),
        default=None,
        help="Fix my agent to one slot. By default it rotates by seed.",
    )
    parser.add_argument(
        "--no-alternate-sides",
        action="store_true",
        help="Deprecated alias for --my-slot 0.",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Save replay HTML, replay JSON, and structured agent log JSON for every game.",
    )
    parser.add_argument(
        "--save-artifacts",
        choices=("none", "loss", "all"),
        default="none",
        help="Save replay JSON plus structured my-agent log JSON for no games, losses, or all games.",
    )
    parser.add_argument(
        "--save-replays",
        choices=("none", "loss", "all"),
        default="loss",
        help="Save HTML replays for no games, losses, or all games.",
    )
    parser.add_argument(
        "--save-root",
        choices=("output", "replay"),
        default="replay",
        help="Top-level save root. HTML, replay JSON, logs, and summary must all use this root.",
    )
    parser.add_argument(
        "--run-name",
        default="local_eval",
        help="Experiment subdirectory under --save-root. Use '.' to write directly into the root.",
    )
    parser.add_argument("--log-every", type=int, default=1, help="OWLOG sampling interval for saved artifacts.")
    parser.add_argument("--summary-json", default=None, help="Optional path for machine-readable experiment results.")
    args = parser.parse_args()
    try:
        save_dir = resolve_save_dir(args.save_root, args.run_name)
        summary_json = resolve_summary_json(args.summary_json, save_dir, args.save_root)
    except ValueError as exc:
        parser.error(str(exc))

    my_agent, my_path = load_agent(args.my_agent, "my_submission_agent")
    baseline_agent, baseline_path = load_agent(args.baseline_agent, "baseline_old_version_agent")
    seeds = parse_seeds(args.seeds, args.seed_start)
    game_count = len(seeds)
    save_replays = "all" if args.log else args.save_replays
    save_artifacts = "all" if args.log else args.save_artifacts
    if args.no_alternate_sides and args.my_slot not in (None, 0):
        raise ValueError("--no-alternate-sides conflicts with --my-slot other than 0")
    fixed_my_slot = 0 if args.no_alternate_sides else args.my_slot
    workers = args.workers or min(game_count, 8, os.cpu_count() or 1)
    workers = max(1, min(workers, game_count))

    print(f"My agent:       {my_path}")
    print(f"Baseline agent: {baseline_path}")
    print(f"Games: {game_count}")
    print(f"Seeds: {seeds}")
    print(f"Match key: {args.match_key}")
    print("Map seed: configuration.seed=seed")
    if fixed_my_slot is None:
        if args.match_key == "seed":
            print(f"My slot: seed-stable slot=(seed-{args.slot_base_seed}) mod 2")
        else:
            print("My slot: order rotation 0/1")
    else:
        print(f"My slot: fixed player {fixed_my_slot}")
    print(f"Episode steps: {args.episode_steps}")
    print(f"Act timeout: {args.act_timeout if args.act_timeout is not None else 'environment default'}")
    print(f"Workers: {workers}")
    print(f"Save replays: {save_replays}")
    print(f"Save artifacts: {save_artifacts}")
    print(f"Log shortcut: {'on' if args.log else 'off'}")
    print(f"Save root: {args.save_root}")
    print(f"Save dir: {save_dir}")
    print()

    wins = 0
    losses = 0
    ties = 0
    my_total = 0.0
    baseline_total = 0.0

    del my_agent
    del baseline_agent

    futures = []
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for game_index, seed in enumerate(seeds):
            my_slot = choose_my_slot(
                game_index,
                seed,
                args.match_key,
                args.slot_base_seed,
                fixed_my_slot,
            )
            futures.append(
                executor.submit(
                    run_game_worker,
                    str(my_path),
                    str(baseline_path),
                    game_index,
                    seed,
                    my_slot,
                    args.episode_steps,
                    args.act_timeout,
                    save_replays,
                    save_artifacts,
                    str(save_dir),
                    args.log_every,
                )
            )

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            print(
                f"Game {result['game']:02d} | seed={result['seed']} | "
                f"map_seed={result['map_seed']} | "
                f"{result['result']} | "
                f"steps={result['steps']} | "
                f"time={result['elapsed']:.2f}s",
                flush=True,
            )

    for result in sorted(results, key=lambda item: item["game"]):
        if result["result"] == "WIN":
            wins += 1
        elif result["result"] == "LOSS":
            losses += 1
        else:
            ties += 1

        my_total += float(result["my_reward"] or 0.0)
        baseline_total += float(result["baseline_reward"] or 0.0)

    decided = wins + losses
    win_rate = wins / game_count if game_count else 0.0
    non_tie_win_rate = wins / decided if decided else 0.0

    print()
    print("Summary")
    print(f"  wins/losses/ties: {wins}/{losses}/{ties}")
    print(f"  win rate: {win_rate:.1%}")
    print(f"  non-tie win rate: {non_tie_win_rate:.1%}")
    print(f"  average reward: my={my_total / game_count:.2f}, baseline={baseline_total / game_count:.2f}")

    if summary_json:
        ordered_results = sorted(results, key=lambda item: item["game"])
        payload = {
            "kind": "2p",
            "my_agent": str(my_path),
            "baseline_agent": str(baseline_path),
            "games": game_count,
            "seeds": seeds,
            "map_seed_policy": "configuration.seed=seed",
            "match_key": args.match_key,
            "slot_base_seed": args.slot_base_seed,
            "my_slot": fixed_my_slot,
            "log": args.log,
            "save_replays": save_replays,
            "save_artifacts": save_artifacts,
            "save_root": args.save_root,
            "save_dir": str(save_dir),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": win_rate,
            "non_tie_win_rate": non_tie_win_rate,
            "avg_my_reward": my_total / game_count if game_count else 0.0,
            "avg_baseline_reward": baseline_total / game_count if game_count else 0.0,
            "results": ordered_results,
        }
        write_json(summary_json, payload)
        print(f"  summary json: {summary_json}")


if __name__ == "__main__":
    main()
