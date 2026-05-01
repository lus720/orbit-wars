# Logging and Experiment Workflow

This project uses structured `OWLOG` records to turn replay review into a
repeatable experiment loop.

## Save Local 2-Player Artifacts

```bash
python eval.py \
  --games 20 \
  --workers 8 \
  --save-root replay \
  --run-name exp-current-2p \
  --save-artifacts loss \
  --summary-json summary.json
```

Saved files use the same shape as downloaded competition data:

- `episode-*.json`: full episode replay data.
- `*-<slot>.json`: captured stdout/stderr for our agent slot.

For 2-player sweeps, `eval.py` defaults to seed-stable slots:
`player_slot=(seed-42) mod 2`. This keeps a focused rerun like `--seeds 52`
on the same player slot as the full `seed42-61` sweep. Use
`--match-key order` only when intentionally reproducing the older list-order
rotation.

Use `--save-artifacts all` for focused debugging sets and `loss` for larger
regression runs.

Use `--workers 8` for the standard seed sweep. If a run shows unusual
`time_pressure`, rerun only those seeds with a smaller worker count to separate
strategy issues from local CPU contention.

## Save Local 4-Player Artifacts

```bash
python eval4.py \
  --games 50 \
  --workers 8 \
  --save-root replay \
  --run-name exp-current-4p \
  --save-artifacts loss \
  --summary-json summary.json
```

`eval4.py` captures only our agent logs by default, even when opponents are
also Python agents.

By default, 4-player matches are now seed-stable:

- opponent selection is derived from `selection_seed + seed`, not from the
  order of `--seeds`;
- when `--my-slot` is omitted, our slot is `slot=(seed-42) mod 4`;
- saved artifacts include `manifest-*.json` with `seed`, `my_slot`, opponent
  paths, `match_hash`, and `map.hash`;
- `analyze_replays.py` uses manifest `my_slot` before falling back to
  `--player-name` or log-file inference, then prints the manifest row so a
  single-seed run and a batch run can be checked for the same
  `match_hash`/`map.hash`.

Use `--match-key order` only when intentionally reproducing older experiment
files where list order determined slot and opponent sampling.

`--save-root` is deliberately restricted to `replay` or `output`; all saved
HTML, episode JSON, OWLOG, manifest, and summary files for a run go under
`<save-root>/<run-name>/`. Do not split one experiment across both directories.

## Analyze Replays And Logs

```bash
python analyze_replays.py --replay-dir replay/exp-current-2p
python analyze_replays.py --replay-dir replay/exp-current-4p --events 8 --checkpoints 6
```

Important fields:

- `prod_gap_delta`: this turn's production-gap change from our logged point of
  view. Negative values mean the production race moved against us.
- `threatened_prod_sum`: total production on our planets expected to fall
  within the next 1-5 turns if no extra action is taken.
- `avoidable_prod_swing`: threatened production-gap swing that still appears
  practically defendable or recapturable by nearby sources.
- `dbg.opening_prize_reserve`: opening reserve rows for generic high-production
  neutral prizes that are not quite ready this turn but are expected to be
  feasible within the short lookahead. Row shape:
  `[target_id, prod, ships, joint_turn, desired, current_total, future_total,
  shortage, source_ids]`.
- `debug_counts.opening_prize_reserve_block`: number of lower-production
  neutral candidate shots skipped because their source is being held for the
  short-lookahead high-production prize. Use this to verify the behavior is a
  general reserve rule, not a seed or planet-id hardcode.
- `production` / `production_swing_events`: replay-derived production-gap
  summaries. These are computed for every episode, including wins, so a change
  is judged by the whole seed set instead of only the losses.
- `Aggregate diag`: repeated strategy signals, such as idle openings, route
  rejections, source draining, or failed multi-source assembly.
- `issue turns`: concrete turns with `src`, `targets`, `actf`, and planner
  stage context.
- `route traces`: exact route-blocker decisions. A `reject` row records
  source, intended target, a blocking planet before the intended target, hit
  distances, opening scores, and whether rejection came from global guard,
  profile guard, poor-target logic, or a 4-player detour relaxation. Use this
  before changing any opening route parameter.
  Row shape:
  `[decision, reason, src_id, src_prod, src_ships, target_id, target_owner,
  target_prod, target_ships, blocker_id, blocker_owner, blocker_prod,
  blocker_ships, target_hit_dist, blocker_hit_dist, target_opening_score,
  blocker_opening_score, guard_enabled, guard_always, profile_guard,
  poor_blocker, relaxed_detour]`.
- `samples.route_guard_reject`, `samples.route_blocker_pass`,
  `samples.aim_no_solution`, and `samples.route_target_miss`: compact reasons
  behind `no_route` so losses can distinguish bad geometry from over-strict
  filtering.
- `samples.opening_wait_relaxed`, `samples.opening_soften_accept`,
  `samples.opening_soften_no_followup`, `samples.route_guard_relaxed`, and
  `samples.reserve_relief`: diagnostics for the current 4-player opening
  experiments. Treat these as evidence, not proof that a strategy change is
  good.
- `notes`: human-readable hints derived from replay and log patterns.

## Check Whether An Experiment Helped

Use three seed groups:

- `target`: losses the change is supposed to fix.
- `regression`: historically fragile seeds.
- `holdout`: seeds not used while designing the change.

Keep a change only if it improves `target`, does not materially degrade
`regression`, and remains stable on `holdout`.

For 2-player seeds `42-61`, also keep a production stability gate:

- analyze all 20 games, not only losses;
- accept strategy changes only when aggregate rapid production drops improve
  across multiple games, especially `negative_delta_sum`, `worst_drop_10`, and
  `worst_drop_20`;
- use `avoidable_prod_swing` and `threatened_prod_sum` to explain the decision
  before changing strategy constants;
- prefer generic defense and recapture scoring for future 1-5 turn high
  production swings. Do not hardcode home planets, seed numbers, or planet IDs.

Current accepted 2-player gate:

- `output/exp-2p-42-61-prod-swing-v1.json`: 20/20 wins,
  `negative_delta_sum=1780`, `worst_drop_10=24`, `worst_drop_20=33`.
- `output/exp-2p-42-61-prize-reserve-v1.json`: 20/20 wins,
  `negative_delta_sum=1432`, `worst_drop_10=21`, `worst_drop_20=19`.
- Accepted reason: the short-lookahead high-production prize reserve reduced
  multi-game rapid production-gap drops while fixing the seed51 regression.

For every kept change, record:

- command and seeds,
- changed strategy/flag,
- win/loss/tie or rank summary,
- aggregate `diag` changes,
- remaining failure examples.

Current 4-player diagnostic notes:

- `output/exp-4p-focused-losses-opening-soften-reserve-v1.json`: old
  order-based focused run fixed seed61 but still lost 48/50/51/58/59
  (`3/8` wins). Do not accept this as a strategy fix without re-running under
  seed-stable matches.
- `output/exp-4p-focused-losses-opening-soften-reserve-wait1-v1.json`: moving
  opening wait relief to turn 1 was rejected (`1/8` wins) because it broke
  seeds that were previously fixed.
- `output/exp-4p-focused-losses-meta-mode-v1.json`: global `ORBIT_STRATEGY_MODE=meta`
  repaired seed48 but broke seed44, so unified opening scoring needs a gated
  trigger rather than global enablement.
- `output/smoke-seed-stable-single.json` and
  `output/smoke-seed-stable-list.json`: seed50 has the same
  `match_hash=f3cd9be09f3afabc` and `map=5848824f288d2568` whether run alone
  or inside a seed list, verifying the new evaluator determinism.
- `output/exp-4p-focused-seedstable-current-losslogs.json`: seed-stable
  focused baseline for seeds `44,48,50,51,55,58,59,61` is `3/8` wins.
  Wins: `44,50,55`. Losses: `48,51,58,59,61`.
- `output/exp-4p-focused-seedstable-current-losslogs-analysis.txt`: current
  seed-stable losses are dominated by multi-source assembly and routing
  failures, not just one bad opening wait:
  `multi_eta_filtered=485`, `partial_options_unassembled=182`,
  `all_candidates_filtered=101`, `negative_delta_sum=1096`,
  `worst_drop_10=28`, `worst_drop_20=28`.
