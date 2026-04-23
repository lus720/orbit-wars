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
POOL = sorted(METHODS_DIR.glob("*.ipynb"))
STREAK_TARGET = 10
MASTER_SEED = 20260423


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
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.args.args and node.args.args[0].arg == "obs":
            if callable(getattr(module, node.name, None)):
                return node.name
    raise RuntimeError("No callable agent found")


def load_module_from_path(path):
    source = notebook_to_python(path)
    if not source.strip():
        raise RuntimeError(f"No writefile agent code found in {path.name}")
    CACHE_DIR.mkdir(exist_ok=True)
    py_name = re.sub(r"[^a-zA-Z0-9_]+", "_", path.stem) + "_4p.py"
    py_path = CACHE_DIR / py_name
    py_path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(py_path.stem, py_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    agent_name = choose_agent_name(source, module)
    return getattr(module, agent_name)


def load_main_agent():
    spec = importlib.util.spec_from_file_location("orbit_main_agent_4p", ROOT / "main.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    for name in ["agent", "smart_agent"]:
        if callable(getattr(module, name, None)):
            return getattr(module, name)
    raise RuntimeError("main.py does not expose agent/smart_agent")


def run_game(our_agent, opponents, seed):
    random.seed(seed)
    env = make("orbit_wars", debug=True)
    env.run([our_agent] + opponents)
    final = env.steps[-1]
    rewards = [item.reward for item in final]
    return rewards


def main():
    our_agent = load_main_agent()
    rng = random.Random(MASTER_SEED)
    cache = {path: load_module_from_path(path) for path in POOL}

    streak = 0
    for game_idx in range(1, STREAK_TARGET + 1):
        sample = rng.sample(POOL, 3)
        opponents = [cache[path] for path in sample]
        seed = rng.randint(1, 10**9)
        rewards = run_game(our_agent, opponents, seed)
        won = all(rewards[0] > reward for reward in rewards[1:])
        streak = streak + 1 if won else 0
        print(
            f"game={game_idx} seed={seed} opponents={[path.name for path in sample]} "
            f"rewards={rewards} result={'WIN' if won else 'LOSS'} streak={streak}"
        )
        if not won:
            raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()

