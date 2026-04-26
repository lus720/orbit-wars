import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import os
from pathlib import Path
import random
import time

from kaggle_environments import make


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
    game_index,
    seed,
    alternate_sides,
    episode_steps,
):
    my_slot = 1 if alternate_sides and game_index % 2 == 1 else 0
    baseline_slot = 1 - my_slot

    agents = [None, None]
    agents[my_slot] = my_agent
    agents[baseline_slot] = baseline_agent

    random.seed(seed)
    start_time = time.perf_counter()
    env = make(
        "orbit_wars",
        {
            "episodeSteps": episode_steps,
            "randomSeed": seed,
        },
        debug=True,
    )
    env.run(agents)
    elapsed = time.perf_counter() - start_time

    final = env.steps[-1]
    my_state = final[my_slot]
    baseline_state = final[baseline_slot]
    my_reward = my_state.reward
    baseline_reward = baseline_state.reward
    scores = final_ship_scores(final[0].observation, len(final))
    my_score = scores[my_slot]
    baseline_score = scores[baseline_slot]

    return {
        "game": game_index + 1,
        "seed": seed,
        "my_slot": my_slot,
        "baseline_slot": baseline_slot,
        "my_reward": my_reward,
        "baseline_reward": baseline_reward,
        "my_score": my_score,
        "baseline_score": baseline_score,
        "my_status": my_state.status,
        "baseline_status": baseline_state.status,
        "result": result_label(my_reward, baseline_reward),
        "steps": len(env.steps),
        "elapsed": elapsed,
    }


def run_game_worker(
    my_agent_path,
    baseline_agent_path,
    game_index,
    seed,
    alternate_sides,
    episode_steps,
):
    my_agent, _ = load_agent(my_agent_path, f"my_submission_agent_{game_index}")
    baseline_agent, _ = load_agent(
        baseline_agent_path,
        f"baseline_submission_agent_{game_index}",
    )
    return run_game(
        my_agent,
        baseline_agent,
        game_index,
        seed,
        alternate_sides,
        episode_steps,
    )


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


def main():
    parser = argparse.ArgumentParser(
        description="Run Orbit Wars matches between submission.py and a baseline."
    )
    parser.add_argument("--my-agent", default="submission.py")
    parser.add_argument("--baseline-agent", default="baselines/submission.py")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument(
        "--seeds",
        nargs="*",
        help="Explicit seeds, e.g. --seeds 42 43 44 or --seeds 42,43,44.",
    )
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker count. Defaults to min(games, CPU count).",
    )
    parser.add_argument(
        "--no-alternate-sides",
        action="store_true",
        help="Keep my agent in slot 0 for every game.",
    )
    args = parser.parse_args()

    my_agent, my_path = load_agent(args.my_agent, "my_submission_agent")
    baseline_agent, baseline_path = load_agent(args.baseline_agent, "baseline_submission_agent")
    seeds = parse_seeds(args.seeds, args.seed_start, args.games)
    if len(seeds) != args.games:
        raise ValueError(f"Expected {args.games} seeds, got {len(seeds)}: {seeds}")
    workers = args.workers or min(args.games, os.cpu_count() or 1)
    workers = max(1, min(workers, args.games))

    print(f"My agent:       {my_path}")
    print(f"Baseline agent: {baseline_path}")
    print(f"Games: {args.games}")
    print(f"Seeds: {seeds}")
    print(f"Alternate sides: {not args.no_alternate_sides}")
    print(f"Episode steps: {args.episode_steps}")
    print(f"Workers: {workers}")
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
            futures.append(
                executor.submit(
                    run_game_worker,
                    str(my_path),
                    str(baseline_path),
                    game_index,
                    seed,
                    not args.no_alternate_sides,
                    args.episode_steps,
                )
            )

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            print(
                f"Game {result['game']:02d} | seed={result['seed']} | "
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
    win_rate = wins / args.games if args.games else 0.0
    non_tie_win_rate = wins / decided if decided else 0.0

    print()
    print("Summary")
    print(f"  wins/losses/ties: {wins}/{losses}/{ties}")
    print(f"  win rate: {win_rate:.1%}")
    print(f"  non-tie win rate: {non_tie_win_rate:.1%}")
    print(f"  average reward: my={my_total / args.games:.2f}, baseline={baseline_total / args.games:.2f}")


if __name__ == "__main__":
    main()
