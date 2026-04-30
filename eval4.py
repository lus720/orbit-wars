import argparse
import contextlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import io
import os
from pathlib import Path
import random
import time
import types

from kaggle_environments import make


NUM_PLAYERS = 4


class NullWriter(io.StringIO):
    def write(self, text):
        return len(text)


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


def discover_baselines(baseline_dir, my_agent_path):
    baseline_dir = Path(baseline_dir)
    my_agent_path = Path(my_agent_path).resolve()
    paths = sorted(path.resolve() for path in baseline_dir.glob("*.py"))
    return [path for path in paths if path != my_agent_path]


def build_game_specs(seeds, my_agent_path, baseline_paths, selection_seed, fixed_my_slot):
    rng = random.Random(selection_seed)
    specs = []
    for game_index, seed in enumerate(seeds):
        my_slot = fixed_my_slot if fixed_my_slot is not None else game_index % NUM_PLAYERS
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
    output_dir,
    agent_random_seed,
    show_agent_output,
):
    agents = []
    loaded_paths = []
    for slot, agent_path in enumerate(agent_paths):
        agent, loaded_path = load_agent(
            agent_path,
            f"orbit_wars_4p_game_{game_index}_slot_{slot}",
        )
        agents.append(agent)
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

    replay_path = None
    should_save_replay = save_replays == "all" or (
        save_replays == "loss" and result == "LOSS"
    )
    if should_save_replay:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        replay_path = output_path / f"4p_seed{seed}_game{game_index + 1:02d}.html"
        replay_path.write_text(env.render(mode="html"), encoding="utf-8")

    return {
        "game": game_index + 1,
        "seed": seed,
        "my_slot": my_slot,
        "agent_paths": loaded_paths,
        "rewards": rewards,
        "scores": scores,
        "statuses": statuses,
        "result": result,
        "winner_slots": winner_slots,
        "rank": rank,
        "replay_path": str(replay_path) if replay_path is not None else None,
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
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker count. Defaults to min(games, 4, CPU count).",
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
    parser.add_argument("--output-dir", default="output")
    parser.add_argument(
        "--show-agent-output",
        action="store_true",
        help="Do not suppress stdout/stderr printed by agents during a match.",
    )
    args = parser.parse_args()

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
    )
    workers = args.workers or min(args.games, 4, os.cpu_count() or 1)
    workers = max(1, min(workers, args.games))

    print(f"My agent: {relative_path(my_path)}")
    print("Baseline pool:")
    for path in baseline_paths:
        print(f"  {relative_path(path)}")
    print(f"Games: {args.games}")
    print(f"Seeds: {seeds}")
    print(f"Baseline selection seed: {args.selection_seed}")
    print(f"My slot: {'rotate 0-3' if args.my_slot is None else args.my_slot}")
    print(f"Episode steps: {args.episode_steps}")
    print(f"Workers: {workers}")
    print(f"Save replays: {args.save_replays}")
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
                    args.output_dir,
                    agent_random_seed,
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
                f"opp=[{opponents}]"
                + (f" | replay={result['replay_path']}" if result["replay_path"] else ""),
                flush=True,
            )

    summarize(sorted(results, key=lambda item: item["game"]), my_path)


if __name__ == "__main__":
    main()
