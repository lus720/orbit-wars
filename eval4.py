import argparse
import contextlib
from collections import Counter, defaultdict
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
import sys
import time
import types

from kaggle_environments import make


NUM_PLAYERS = 4
DEFAULT_SLOT_BASE_SEED = 42
PLAYER_COLORS = (
    ("blue", "#0072B2"),
    ("orange", "#E69F00"),
    ("green/teal", "#009E73"),
    ("yellow", "#F0E442"),
)


class NullWriter(io.StringIO):
    def write(self, text):
        return len(text)


class AgentRecorder:
    def __init__(self, agent, log_enabled=False, log_every=1, mirror_output=False):
        self.agent = agent
        self.log_enabled = log_enabled
        self.log_every = max(1, int(log_every))
        self.mirror_output = mirror_output
        self.records = []
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
                if self.accepts_config:
                    return self.agent(obs, config)
                return self.agent(obs)
        finally:
            stdout_text = stdout_buffer.getvalue()
            stderr_text = stderr_buffer.getvalue()
            self.records.append({"stdout": stdout_text, "stderr": stderr_text})
            if self.mirror_output:
                if stdout_text:
                    print(stdout_text, end="", file=sys.stdout)
                if stderr_text:
                    print(stderr_text, end="", file=sys.stderr)
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


def load_agent(path, module_name):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import agent from {path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SyntaxError as exc:
        module = load_agent_from_notebook_export(path, module_name, exc)

    agent = getattr(module, "agent", None)
    if agent is None:
        raise RuntimeError(f"{path} does not define an agent function")
    return agent, path


def load_agent_from_notebook_export(path, module_name, original_error):
    source = path.read_text(encoding="utf-8")
    if not any(line.lstrip().startswith("%%") for line in source.splitlines()):
        raise original_error

    cleaned_lines = [
        "" if line.lstrip().startswith("%%") else line
        for line in source.splitlines()
    ]
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__name__ = module_name
    code = compile("\n".join(cleaned_lines), str(path), "exec")
    exec(code, module.__dict__)
    return module


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


def parse_seeds(seed_args, seed_start, games):
    if seed_args:
        seeds = []
        for item in seed_args:
            for part in item.split(","):
                part = part.strip()
                if part:
                    seeds.append(int(part))
        return seeds
    return list(range(seed_start, seed_start + games))


def relative_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


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


def replay_banner_html(my_slot, my_agent_path, result, rank, scores, winner_slots, seed, game):
    _color_name, color_hex = player_color(my_slot)
    winner_html = ", ".join(player_token_html(slot) for slot in winner_slots) or "none"
    score_html = ", ".join(
        f"{player_token_html(slot)}={float(score):.0f}" for slot, score in enumerate(scores)
    )
    agent_text = html_lib.escape(str(my_agent_path))
    return f"""
<div id="codex-my-agent-banner" style="position:fixed;top:0;left:0;bottom:44px;width:min(340px,28vw);z-index:2147483647;pointer-events:none;background:rgba(17,17,17,.94);color:#fff;padding:12px 14px;font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;border-right:3px solid {html_lib.escape(color_hex)};box-shadow:2px 0 10px rgba(0,0,0,.35);box-sizing:border-box;overflow:hidden;overflow-wrap:anywhere">
  <div><strong>MY METHOD</strong>: {swatch_html(color_hex)} player {int(my_slot)} / <code>{agent_text}</code></div>
  <div>seed={int(seed)} game={int(game)} result={html_lib.escape(str(result))} rank={int(rank)} winners={winner_html}</div>
  <div>scores: {score_html}</div>
</div>
""".strip()


def render_replay_with_my_banner(env, my_slot, my_agent_path, result, rank, scores, winner_slots, seed, game):
    rendered = env.render(mode="html")
    banner = replay_banner_html(my_slot, my_agent_path, result, rank, scores, winner_slots, seed, game)
    return inject_replay_banner(rendered, banner)


def discover_baselines(baseline_dir, my_agent_path):
    baseline_dir = Path(baseline_dir)
    my_agent_path = Path(my_agent_path).resolve()
    paths = sorted(path.resolve() for path in baseline_dir.glob("*.py"))
    return [path for path in paths if path != my_agent_path]


def stable_json_hash(payload):
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]


def stable_seed_rng(selection_seed, seed):
    return random.Random(int(selection_seed) * 1_000_003 + int(seed))


def seed_stable_slot(seed, slot_base_seed):
    return (int(seed) - int(slot_base_seed)) % NUM_PLAYERS


def map_signature_from_episode(episode):
    for step in episode.get("steps", []):
        for state in step:
            obs = state.get("observation") if isinstance(state, dict) else None
            if not isinstance(obs, dict):
                continue
            planets = obs.get("initial_planets") or obs.get("planets") or []
            if not planets:
                continue
            canonical_planets = sorted(
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
            )
            payload = {
                "planets": canonical_planets,
                "angular_velocity": round(float(obs.get("angular_velocity", 0.0)), 8),
                "comets": obs.get("comets") or [],
                "comet_planet_ids": obs.get("comet_planet_ids") or [],
            }
            home_slots = defaultdict(list)
            for planet in obs.get("planets") or []:
                owner = int(planet[1])
                if 0 <= owner < NUM_PLAYERS:
                    home_slots[owner].append(int(planet[0]))
            return {
                "hash": stable_json_hash(payload),
                "planet_count": len(canonical_planets),
                "home_planets": {
                    str(owner): sorted(ids)
                    for owner, ids in sorted(home_slots.items())
                },
            }
    return {"hash": "unknown", "planet_count": 0, "home_planets": {}}


def build_game_specs(
    seeds,
    my_agent_path,
    baseline_paths,
    selection_seed,
    fixed_my_slot,
    match_key,
    slot_base_seed,
):
    order_rng = random.Random(selection_seed)
    specs = []
    for game_index, seed in enumerate(seeds):
        if fixed_my_slot is not None:
            my_slot = fixed_my_slot
        elif match_key == "seed":
            my_slot = seed_stable_slot(seed, slot_base_seed)
        else:
            my_slot = game_index % NUM_PLAYERS

        rng = stable_seed_rng(selection_seed, seed) if match_key == "seed" else order_rng
        opponents = rng.sample(baseline_paths, NUM_PLAYERS - 1)
        opponent_iter = iter(opponents)
        agent_paths = []
        for slot in range(NUM_PLAYERS):
            if slot == my_slot:
                agent_paths.append(my_agent_path)
            else:
                agent_paths.append(next(opponent_iter))

        specs.append(
            {
                "game_index": game_index,
                "seed": seed,
                "my_slot": my_slot,
                "agent_paths": [str(path) for path in agent_paths],
                "match_key": match_key,
                "slot_base_seed": slot_base_seed,
            }
        )
    return specs


def result_label(my_slot, rewards):
    winner_slots = [slot for slot, reward in enumerate(rewards) if reward == 1]
    if my_slot not in winner_slots:
        return "LOSS", winner_slots
    if len(winner_slots) == 1:
        return "WIN", winner_slots
    return "TIE_WIN", winner_slots


def rank_from_scores(my_slot, scores):
    my_score = scores[my_slot]
    return 1 + sum(score > my_score for score in scores)


def run_game_worker(
    agent_paths,
    my_slot,
    game_index,
    seed,
    episode_steps,
    save_replays,
    save_artifacts,
    save_dir,
    log_every,
    agent_random_seed,
    selection_seed,
    match_key,
    slot_base_seed,
    show_agent_output,
):
    agents = []
    recorders = []
    loaded_paths = []
    for slot, agent_path in enumerate(agent_paths):
        agent, loaded_path = load_agent(
            agent_path,
            f"orbit_wars_4p_game_{game_index}_slot_{slot}",
        )
        recorder = AgentRecorder(
            agent,
            log_enabled=(slot == my_slot and save_artifacts != "none"),
            log_every=log_every,
            mirror_output=show_agent_output,
        )
        agents.append(recorder)
        recorders.append(recorder)
        loaded_paths.append(str(loaded_path))

    random.seed(agent_random_seed)
    start_time = time.perf_counter()
    env = make(
        "orbit_wars",
        {
            "episodeSteps": episode_steps,
            "randomSeed": seed,
        },
        debug=True,
    )
    random.seed(agent_random_seed)
    if show_agent_output:
        env.run(agents)
    else:
        sink = NullWriter()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            env.run(agents)
    elapsed = time.perf_counter() - start_time

    final = env.steps[-1]
    rewards = [state.reward for state in final]
    statuses = [state.status for state in final]
    scores = final_ship_scores(final[0].observation, len(final))
    result, winner_slots = result_label(my_slot, rewards)
    rank = rank_from_scores(my_slot, scores)
    episode_json = env.toJSON()
    map_signature = map_signature_from_episode(episode_json)
    agent_names = [relative_path(path) for path in loaded_paths]
    match_metadata = {
        "kind": "eval4_match",
        "version": 1,
        "seed": seed,
        "game": game_index + 1,
        "match_key": match_key,
        "selection_seed": selection_seed,
        "slot_base_seed": slot_base_seed,
        "my_slot": my_slot,
        "agent_paths": agent_names,
        "opponents": {
            str(slot): agent_names[slot]
            for slot in range(NUM_PLAYERS)
            if slot != my_slot
        },
        "map": map_signature,
    }
    match_metadata["match_hash"] = stable_json_hash(
        {
            "seed": seed,
            "my_slot": my_slot,
            "agent_paths": agent_names,
            "map": map_signature.get("hash"),
        }
    )

    replay_path = None
    should_save_replay = save_replays == "all" or (
        save_replays == "loss" and result == "LOSS"
    )
    if should_save_replay:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        replay_path = save_path / f"4p_seed{seed}_game{game_index + 1:02d}.html"
        replay_path.write_text(
            render_replay_with_my_banner(
                env,
                my_slot,
                agent_names[my_slot],
                result,
                rank,
                scores,
                winner_slots,
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
        prefix = f"4p-seed{seed}-game{game_index + 1:02d}"
        artifact_root = Path(save_dir)
        artifact_episode_path = artifact_root / f"episode-{prefix}.json"
        artifact_log_path = artifact_root / f"{prefix}-{my_slot}.json"
        artifact_manifest_path = artifact_root / f"manifest-{prefix}.json"
        write_json(artifact_episode_path, episode_json)
        write_json(artifact_log_path, recorders[my_slot].records)
        write_json(artifact_manifest_path, match_metadata)

    return {
        "game": game_index + 1,
        "seed": seed,
        "my_slot": my_slot,
        "agent_paths": loaded_paths,
        "match": match_metadata,
        "rewards": rewards,
        "scores": scores,
        "statuses": statuses,
        "result": result,
        "winner_slots": winner_slots,
        "rank": rank,
        "replay_path": str(replay_path) if replay_path is not None else None,
        "artifact_episode_path": str(artifact_episode_path) if artifact_episode_path is not None else None,
        "artifact_log_path": str(artifact_log_path) if artifact_log_path is not None else None,
        "artifact_manifest_path": str(artifact_manifest_path) if artifact_manifest_path is not None else None,
        "steps": len(env.steps),
        "elapsed": elapsed,
    }


def summarize(results, my_path):
    result_counts = Counter(result["result"] for result in results)
    rank_counts = Counter(result["rank"] for result in results)
    my_rewards = []
    my_scores = []
    baseline_stats = defaultdict(
        lambda: {"games": 0, "wins": 0, "reward": 0.0, "score": 0.0}
    )

    for result in results:
        my_slot = result["my_slot"]
        my_rewards.append(float(result["rewards"][my_slot] or 0.0))
        my_scores.append(float(result["scores"][my_slot]))

        for slot, path in enumerate(result["agent_paths"]):
            if slot == my_slot:
                continue
            stats = baseline_stats[path]
            stats["games"] += 1
            stats["wins"] += 1 if result["rewards"][slot] == 1 else 0
            stats["reward"] += float(result["rewards"][slot] or 0.0)
            stats["score"] += float(result["scores"][slot])

    games = len(results)
    winner_rate = (result_counts["WIN"] + result_counts["TIE_WIN"]) / games if games else 0.0
    solo_win_rate = result_counts["WIN"] / games if games else 0.0
    avg_rank = sum(result["rank"] for result in results) / games if games else 0.0
    avg_reward = sum(my_rewards) / games if games else 0.0
    avg_score = sum(my_scores) / games if games else 0.0

    print()
    print("Summary")
    print(f"  my agent: {relative_path(my_path)}")
    print(
        "  results: "
        f"win={result_counts['WIN']}, "
        f"tie_win={result_counts['TIE_WIN']}, "
        f"loss={result_counts['LOSS']}"
    )
    print(f"  winner rate: {winner_rate:.1%}")
    print(f"  solo win rate: {solo_win_rate:.1%}")
    print(f"  average rank: {avg_rank:.2f}")
    print(f"  rank distribution: {dict(sorted(rank_counts.items()))}")
    print(f"  average reward: {avg_reward:.2f}")
    print(f"  average score: {avg_score:.1f}")

    print()
    print("Baseline pool stats")
    for path, stats in sorted(baseline_stats.items(), key=lambda item: relative_path(item[0])):
        games_for_baseline = stats["games"]
        avg_baseline_reward = stats["reward"] / games_for_baseline
        avg_baseline_score = stats["score"] / games_for_baseline
        print(
            f"  {relative_path(path)} | "
            f"games={games_for_baseline} | "
            f"wins={stats['wins']} | "
            f"avg_reward={avg_baseline_reward:.2f} | "
            f"avg_score={avg_baseline_score:.1f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run reproducible 4-player Orbit Wars matches: submission.py plus "
            "three randomly sampled baselines."
        )
    )
    parser.add_argument("--my-agent", default="submission.py")
    parser.add_argument("--baseline-dir", default="baselines")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument(
        "--seeds",
        nargs="*",
        help="Explicit environment seeds, e.g. --seeds 42 43 44 or --seeds 42,43,44.",
    )
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=20260429,
        help="Seed used to choose the three baseline opponents for each game.",
    )
    parser.add_argument(
        "--match-key",
        choices=("seed", "order"),
        default="seed",
        help=(
            "Use seed-stable opponents and slot by default. Use order to reproduce "
            "the older behavior where list order determined slot/opponents."
        ),
    )
    parser.add_argument(
        "--slot-base-seed",
        type=int,
        default=DEFAULT_SLOT_BASE_SEED,
        help="Base seed for seed-stable slot rotation: slot=(seed-base) mod 4.",
    )
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker count. Defaults to min(games, 8, CPU count).",
    )
    parser.add_argument(
        "--my-slot",
        type=int,
        choices=range(NUM_PLAYERS),
        default=None,
        help="Fix my agent to one slot. By default it rotates through slots 0-3.",
    )
    parser.add_argument(
        "--save-replays",
        choices=("none", "loss", "all"),
        default="loss",
        help="Save HTML replays for no games, losses, or all games.",
    )
    parser.add_argument(
        "--save-artifacts",
        choices=("none", "loss", "all"),
        default="none",
        help="Save episode JSON plus my-agent OWLOG capture for no games, losses, or all games.",
    )
    parser.add_argument(
        "--save-root",
        choices=("output", "replay"),
        default="replay",
        help="Top-level save root. HTML, episode JSON, OWLOG, manifest, and summary must all use this root.",
    )
    parser.add_argument(
        "--run-name",
        default="local_eval4",
        help="Experiment subdirectory under --save-root. Use '.' to write directly into the root.",
    )
    parser.add_argument("--log-every", type=int, default=1, help="OWLOG sampling interval for saved artifacts.")
    parser.add_argument("--summary-json", default=None, help="Optional path for machine-readable experiment results.")
    parser.add_argument(
        "--show-agent-output",
        action="store_true",
        help="Do not suppress stdout/stderr printed by agents during a match.",
    )
    args = parser.parse_args()
    try:
        save_dir = resolve_save_dir(args.save_root, args.run_name)
        summary_json = resolve_summary_json(args.summary_json, save_dir, args.save_root)
    except ValueError as exc:
        parser.error(str(exc))

    my_agent, my_path = load_agent(args.my_agent, "my_submission_agent_4p")
    baseline_paths = discover_baselines(args.baseline_dir, my_path)
    if len(baseline_paths) < NUM_PLAYERS - 1:
        raise ValueError(
            f"Need at least {NUM_PLAYERS - 1} baseline .py files in {args.baseline_dir}; "
            f"found {len(baseline_paths)}"
        )

    seeds = parse_seeds(args.seeds, args.seed_start, args.games)
    if len(seeds) != args.games:
        raise ValueError(f"Expected {args.games} seeds, got {len(seeds)}: {seeds}")

    specs = build_game_specs(
        seeds,
        my_path,
        baseline_paths,
        args.selection_seed,
        args.my_slot,
        args.match_key,
        args.slot_base_seed,
    )
    workers = args.workers or min(args.games, 8, os.cpu_count() or 1)
    workers = max(1, min(workers, args.games))

    print(f"My agent: {relative_path(my_path)}")
    print("Baseline pool:")
    for path in baseline_paths:
        print(f"  {relative_path(path)}")
    print(f"Games: {args.games}")
    print(f"Seeds: {seeds}")
    print(f"Baseline selection seed: {args.selection_seed}")
    print(f"Match key: {args.match_key}")
    if args.my_slot is None:
        if args.match_key == "seed":
            print(f"My slot: seed-stable slot=(seed-{args.slot_base_seed}) mod 4")
        else:
            print("My slot: order rotation 0-3")
    else:
        print(f"My slot: fixed s{args.my_slot}")
    print(f"Episode steps: {args.episode_steps}")
    print(f"Workers: {workers}")
    print(f"Save replays: {args.save_replays}")
    print(f"Save artifacts: {args.save_artifacts}")
    print(f"Save root: {args.save_root}")
    print(f"Save dir: {save_dir}")
    print()

    del my_agent

    futures = []
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for spec in specs:
            agent_random_seed = args.selection_seed * 1_000_003 + spec["seed"]
            futures.append(
                executor.submit(
                    run_game_worker,
                    spec["agent_paths"],
                    spec["my_slot"],
                    spec["game_index"],
                    spec["seed"],
                    args.episode_steps,
                    args.save_replays,
                    args.save_artifacts,
                    str(save_dir),
                    args.log_every,
                    agent_random_seed,
                    args.selection_seed,
                    args.match_key,
                    args.slot_base_seed,
                    args.show_agent_output,
                )
            )

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            my_slot = result["my_slot"]
            opponents = ", ".join(
                f"s{slot}:{Path(path).name}"
                for slot, path in enumerate(result["agent_paths"])
                if slot != my_slot
            )
            winners = ",".join(f"s{slot}" for slot in result["winner_slots"]) or "-"
            print(
                f"Game {result['game']:02d} | seed={result['seed']} | "
                f"my=s{my_slot} | {result['result']} | "
                f"rank={result['rank']} | "
                f"score={result['scores'][my_slot]:.0f} | "
                f"winners={winners} | "
                f"steps={result['steps']} | "
                f"time={result['elapsed']:.2f}s | "
                f"match={result['match']['match_hash']} | "
                f"map={result['match']['map']['hash']} | "
                f"opp=[{opponents}]"
                + (f" | replay={result['replay_path']}" if result["replay_path"] else "")
                + (f" | log={result['artifact_log_path']}" if result["artifact_log_path"] else ""),
                flush=True,
            )

    ordered_results = sorted(results, key=lambda item: item["game"])
    summarize(ordered_results, my_path)

    if summary_json:
        result_counts = Counter(result["result"] for result in ordered_results)
        rank_counts = Counter(result["rank"] for result in ordered_results)
        games = len(ordered_results)
        winner_rate = (result_counts["WIN"] + result_counts["TIE_WIN"]) / games if games else 0.0
        solo_win_rate = result_counts["WIN"] / games if games else 0.0
        payload = {
            "kind": "4p",
            "my_agent": str(my_path),
            "baseline_dir": args.baseline_dir,
            "baseline_pool": [str(path) for path in baseline_paths],
            "games": args.games,
            "seeds": seeds,
            "selection_seed": args.selection_seed,
            "match_key": args.match_key,
            "slot_base_seed": args.slot_base_seed,
            "my_slot": args.my_slot,
            "save_root": args.save_root,
            "save_dir": str(save_dir),
            "results_count": dict(result_counts),
            "rank_count": dict(rank_counts),
            "winner_rate": winner_rate,
            "solo_win_rate": solo_win_rate,
            "results": ordered_results,
        }
        write_json(summary_json, payload)
        print(f"  summary json: {summary_json}")


if __name__ == "__main__":
    main()
