import ast
import importlib.util
import json
import random
import re
from pathlib import Path

from kaggle_environments import make


ROOT = Path(__file__).resolve().parent
METHODS_DIR = ROOT / "methods"
CACHE_DIR = ROOT / ".generated_methods"
SEEDS = [7, 19, 43]


def notebook_to_python(notebook_path):
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    chunks = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        lines = source.splitlines()
        if not lines:
            continue
        first = lines[0].strip()
        if first.startswith("%%writefile") and ("submission.py" in first or "main.py" in first):
            chunks.append("\n".join(lines[1:]))
    return "\n\n".join(chunk for chunk in chunks if chunk.strip()) + "\n"


def choose_agent_name(source, module):
    preferred = ["agent", "smart_agent", "nearest_planet_sniper"]
    for name in preferred:
        if callable(getattr(module, name, None)):
            return name

    tree = ast.parse(source)
    function_names = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.args.args:
            first_arg = node.args.args[0].arg
            if first_arg == "obs":
                function_names.append(node.name)
    for name in function_names:
        if callable(getattr(module, name, None)):
            return name
    raise RuntimeError("No callable agent found")


def load_module_from_path(path):
    source = notebook_to_python(path)
    if not source.strip():
        raise RuntimeError(f"No writefile agent code found in {path.name}")
    CACHE_DIR.mkdir(exist_ok=True)
    py_name = re.sub(r"[^a-zA-Z0-9_]+", "_", path.stem) + ".py"
    py_path = CACHE_DIR / py_name
    py_path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(py_path.stem, py_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    agent_name = choose_agent_name(source, module)
    return module, getattr(module, agent_name), agent_name


def load_main_agent():
    main_path = ROOT / "main.py"
    spec = importlib.util.spec_from_file_location("orbit_main_agent", main_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    for name in ["agent", "smart_agent"]:
        if callable(getattr(module, name, None)):
            return getattr(module, name)
    raise RuntimeError("main.py does not expose agent/smart_agent")


def run_game(our_agent, their_agent, seed):
    random.seed(seed)
    env = make("orbit_wars", debug=True)
    env.run([our_agent, their_agent])
    final = env.steps[-1]
    return {
        "our_reward": final[0].reward,
        "their_reward": final[1].reward,
        "our_status": final[0].status,
        "their_status": final[1].status,
        "steps": len(env.steps),
    }


def summarize(opponent_name, results):
    wins = sum(1 for item in results if item["our_reward"] > item["their_reward"])
    losses = sum(1 for item in results if item["our_reward"] < item["their_reward"])
    draws = len(results) - wins - losses
    total_margin = sum(item["our_reward"] - item["their_reward"] for item in results)
    print(f"{opponent_name}: {wins}-{losses}-{draws}, margin={total_margin}")
    for seed, item in zip(SEEDS, results):
        print(
            f"  seed={seed} reward={item['our_reward']}:{item['their_reward']} "
            f"status={item['our_status']}/{item['their_status']} steps={item['steps']}"
        )
    return wins == len(results)


def main():
    overall = True
    for notebook_path in sorted(METHODS_DIR.glob("*.ipynb")):
        if notebook_path.name == "getting-started.ipynb":
            continue
        our_agent = load_main_agent()
        _module, their_agent, agent_name = load_module_from_path(notebook_path)
        results = [run_game(our_agent, their_agent, seed) for seed in SEEDS]
        all_won = summarize(f"{notebook_path.name}::{agent_name}", results)
        overall = overall and all_won
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
