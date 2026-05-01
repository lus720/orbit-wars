# Optimization Log

This document records local evaluation results while optimizing root `submission.py`
against `baselines/mine_old_version.py`.

Evaluation command template:

```bash
python eval.py --games 20 --workers 10
```

Evaluation uses the local Kaggle `orbit_wars` environment. Results are local
regression signals, not official leaderboard results.

## 2026-04-26 Baseline Check

### Change

No strategy change.

Root `submission.py` and `baselines/mine_old_version.py` were byte-identical before
this run:

```text
cmp_exit=0
submission.py md5:          a70a052e298404be04de384230f950ab
baselines/mine_old_version.py md5: a70a052e298404be04de384230f950ab
```

### Command

```bash
python eval.py --games 20 --workers 10
```

### Settings

```text
Seeds: 42-61
Alternate sides: True
Episode steps: 500
Workers: 10
Baseline agent: baselines/mine_old_version.py
Candidate agent: submission.py
```

### Per-Game Results

```text
Game 01 | seed=42 | LOSS | steps=219 | time=60.71s
Game 02 | seed=43 | WIN  | steps=287 | time=125.60s
Game 03 | seed=44 | WIN  | steps=195 | time=109.27s
Game 04 | seed=45 | WIN  | steps=247 | time=158.81s
Game 05 | seed=46 | WIN  | steps=252 | time=142.61s
Game 06 | seed=47 | LOSS | steps=238 | time=108.51s
Game 07 | seed=48 | WIN  | steps=199 | time=57.32s
Game 08 | seed=49 | LOSS | steps=182 | time=63.72s
Game 09 | seed=50 | WIN  | steps=210 | time=112.47s
Game 10 | seed=51 | WIN  | steps=207 | time=141.01s
Game 11 | seed=52 | WIN  | steps=301 | time=107.42s
Game 12 | seed=53 | LOSS | steps=171 | time=58.12s
Game 13 | seed=54 | LOSS | steps=203 | time=67.33s
Game 14 | seed=55 | WIN  | steps=156 | time=57.67s
Game 15 | seed=56 | WIN  | steps=245 | time=99.37s
Game 16 | seed=57 | LOSS | steps=180 | time=88.42s
Game 17 | seed=58 | WIN  | steps=269 | time=99.07s
Game 18 | seed=59 | WIN  | steps=194 | time=102.74s
Game 19 | seed=60 | LOSS | steps=230 | time=118.04s
Game 20 | seed=61 | WIN  | steps=165 | time=68.15s
```

### Summary

```text
wins/losses/ties: 13/7/0
win rate: 65.0%
non-tie win rate: 65.0%
average reward: my=0.30, baseline=-0.30
```

### Interpretation

Because the two files were identical, this result should not be interpreted as a
real strategy improvement. It is the starting reference for the current local
evaluation setup and includes side/seed variance.

Future entries should record:

- exact strategy or parameter change,
- evaluation command,
- seeds and game count,
- win/loss/tie summary,
- absolute win-rate change versus this baseline and versus the previous run,
- short interpretation of whether the change is worth keeping.
