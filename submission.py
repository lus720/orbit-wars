import math
import os
import json
import sys
import time
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from itertools import combinations

# ============================================================
# Shared Configuration
# ============================================================

BOARD = 100.0
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_R = 10.0
MAX_SPEED = 6.0
SUN_SAFETY = 1.5
ROTATION_LIMIT = 50.0
TOTAL_STEPS = 500
SIM_HORIZON = 110
ROUTE_SEARCH_HORIZON = 60
HORIZON = 110
LAUNCH_CLEARANCE = 0.1
PATH_BLOCKER_EPSILON = 0.05
ROUTE_AIM_OFFSETS = (0.0, -0.35, 0.35, -0.7, 0.7, -0.95, 0.95)
EDGE_AIM_ENABLED = False
DELAYED_SNIPE_ENABLED = False
LOCAL_OPENING_ENABLED = False
LEAN_OPENING_ENABLED = False
AGGRESSIVE_DEFENSE_ENABLED = False
OPENING_ROUTE_GUARD_ENABLED = False
OPENING_ROUTE_GUARD_ALWAYS = False
OPENING_META_ENABLED = False
ORBIT_LOG_PREFIX = "OWLOG "
ORBIT_LOG_SAMPLE_LIMIT = 4
ORBIT_TIMING_LIMIT = 24
PROFILE_SIGNATURE = None
PROFILE_EDGE_AIM = False
PROFILE_DELAYED_SNIPE = False
PROFILE_LOCAL_OPENING = False
PROFILE_LEAN_OPENING = False
PROFILE_AGGRESSIVE_DEFENSE = False
PROFILE_OPENING_ROUTE_GUARD = False
PROFILE_ARCHETYPE = "baseline"
PROFILE_HOME_IDS = ()
PROFILE_LAST_OWNERS = {}
PROFILE_CAPTURED_AT = {}


def _truthy_env(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in ("0", "false", "off", "no")


def timing_add(world, name, elapsed_ms):
    if world is None or not getattr(world, "timing_enabled", False):
        return
    records = getattr(world, "timing_records", None)
    if records is None:
        records = {}
        world.timing_records = records
    total_ms, count = records.get(name, (0.0, 0))
    records[name] = (total_ms + float(elapsed_ms), count + 1)


def timed_call(world, name, fn, *args, **kwargs):
    if world is None or not getattr(world, "timing_enabled", False):
        return fn(*args, **kwargs)
    started = time.perf_counter()
    try:
        return fn(*args, **kwargs)
    finally:
        timing_add(world, name, (time.perf_counter() - started) * 1000.0)


def timing_rows(world, limit=ORBIT_TIMING_LIMIT):
    records = getattr(world, "timing_records", None) or {}
    rows = []
    for name, (total_ms, count) in records.items():
        if count <= 0:
            continue
        rows.append(
            [
                name,
                round(float(total_ms), 3),
                int(count),
                round(float(total_ms) / max(1, int(count)), 4),
            ]
        )
    rows.sort(key=lambda row: (-row[1], row[0]))
    return rows[:limit]

EARLY_TURN_LIMIT = 40
OPENING_TURN_LIMIT = 80
LATE_REMAINING_TURNS = 60
VERY_LATE_REMAINING_TURNS = 25

SAFE_NEUTRAL_MARGIN = 2
CONTESTED_NEUTRAL_MARGIN = 2
INTERCEPT_TOLERANCE = 1

SAFE_OPENING_PROD_THRESHOLD = 4
SAFE_OPENING_TURN_LIMIT = 10
ROTATING_OPENING_MAX_TURNS = 13
ROTATING_OPENING_LOW_PROD = 2
FOUR_PLAYER_ROTATING_REACTION_GAP = 3
FOUR_PLAYER_ROTATING_SEND_RATIO = 0.62
FOUR_PLAYER_ROTATING_TURN_LIMIT = 10

COMET_MAX_CHASE_TURNS = 10
COMET_EVAC_LIFE_TURNS = 2
COMET_EVAC_MIN_SHIPS = 1

ATTACK_COST_TURN_WEIGHT = 0.55
SNIPE_COST_TURN_WEIGHT = 0.45
INDIRECT_VALUE_SCALE = 0.15
INDIRECT_FRIENDLY_WEIGHT = 0.35
INDIRECT_NEUTRAL_WEIGHT = 0.9
INDIRECT_ENEMY_WEIGHT = 1.25

STATIC_NEUTRAL_VALUE_MULT = 1.4
STATIC_HOSTILE_VALUE_MULT = 1.55
ROTATING_OPENING_VALUE_MULT = 0.9
HOSTILE_TARGET_VALUE_MULT = 1.85
OPENING_HOSTILE_TARGET_VALUE_MULT = 1.45
SAFE_NEUTRAL_VALUE_MULT = 1.2
CONTESTED_NEUTRAL_VALUE_MULT = 0.7
EARLY_NEUTRAL_VALUE_MULT = 1.2
COMET_VALUE_MULT = 0.65
SNIPE_VALUE_MULT = 1.12
SWARM_VALUE_MULT = 1.05
REINFORCE_VALUE_MULT = 1.35
CRASH_EXPLOIT_VALUE_MULT = 1.18
FINISHING_HOSTILE_VALUE_MULT = 1.15
BEHIND_ROTATING_NEUTRAL_VALUE_MULT = 0.92
LOCAL_NEUTRAL_CORE_DIST = 18.0
LOCAL_NEUTRAL_OUTER_DIST = 30.0
LOCAL_NEUTRAL_MAX_BONUS = 0.28
LOCAL_NEUTRAL_SOFT_SHIPS = 18
FAR_NEUTRAL_OPPORTUNITY_DIST = 32.0
FAR_NEUTRAL_OPPORTUNITY_PENALTY = 0.92
LOCAL_PRIZE_RESERVE_TURNS = 8
LOCAL_PRIZE_MAX_DIST = 22.0
LOCAL_PRIZE_ENEMY_LEAD_ALLOWANCE = 4.0
LOCAL_PRIZE_MIN_PRODUCTION = 3
LOCAL_PRIZE_PROD_GAP = 2
LOCAL_PRIZE_FAR_EXTRA_DIST = 8.0

NEUTRAL_MARGIN_BASE = 2
NEUTRAL_MARGIN_PROD_WEIGHT = 2
NEUTRAL_MARGIN_CAP = 8
HOSTILE_MARGIN_BASE = 3
HOSTILE_MARGIN_PROD_WEIGHT = 2
HOSTILE_MARGIN_CAP = 12
STATIC_TARGET_MARGIN = 4
CONTESTED_TARGET_MARGIN = 5
FOUR_PLAYER_TARGET_MARGIN = 3
LONG_TRAVEL_MARGIN_START = 18
LONG_TRAVEL_MARGIN_DIVISOR = 3
LONG_TRAVEL_MARGIN_CAP = 8
COMET_MARGIN_RELIEF = 6
FINISHING_HOSTILE_SEND_BONUS = 3
POST_CAPTURE_HOLD_WINDOW = 2
POST_CAPTURE_HIGH_PROD_HOLD_WINDOW = 6
POST_CAPTURE_HOLD_RATIO = 0.65
POST_CAPTURE_HOLD_MARGIN_CAP = 18

STATIC_TARGET_SCORE_MULT = 1.18
EARLY_STATIC_NEUTRAL_SCORE_MULT = 1.25
FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT = 0.84
DENSE_STATIC_NEUTRAL_COUNT = 4
DENSE_ROTATING_NEUTRAL_SCORE_MULT = 0.86
SNIPE_SCORE_MULT = 1.12
SWARM_SCORE_MULT = 1.06
CRASH_EXPLOIT_SCORE_MULT = 1.05

FOLLOWUP_MIN_SHIPS = 8
LOW_VALUE_COMET_PRODUCTION = 1
LATE_CAPTURE_BUFFER = 5
VERY_LATE_CAPTURE_BUFFER = 3

DEFENSE_LOOKAHEAD_TURNS = 28
DEFENSE_COST_TURN_WEIGHT = 0.4
DEFENSE_FRONTIER_SCORE_MULT = 1.12
URGENT_DEFENSE_TURN = 12
URGENT_DEFENSE_SCORE_MULT = 1.45
EARLY_DEFENSE_SCORE_MULT = 1.15
DEFENSE_SEND_MARGIN_BASE = 1
DEFENSE_SEND_MARGIN_PROD_WEIGHT = 1
DEFENSE_SHIP_VALUE = 0.55
DEFENSE_CORE_PRODUCTION = 3
CORE_PRODUCTION = 4
FRESH_CORE_SOURCE_LOCK_TURN_LIMIT = 90
FRESH_CORE_SOURCE_LOCK_TURNS = 22
FRESH_CORE_SOURCE_LOCK_PROD_MIN = 4
FRESH_CORE_SOURCE_LOCK_KEEP = 24
FRESH_CORE_SOURCE_LOCK_FRACTION = 0.15
FRESH_CORE_SOURCE_LOCK_THREAT_HORIZON = 14
FRESH_CORE_SOURCE_LOCK_THREAT_TURN = 10
FRESH_CORE_SOURCE_LOCK_STACK_FLOOR = 10
RESERVE_RELIEF_START_STEP = 60
RESERVE_RELIEF_PROD_GAP = 10
RESERVE_RELIEF_TOTAL_RATIO = 0.72
RESERVE_RELIEF_KEEP_FRACTION = 0.68
RESERVE_RELIEF_KEEP_BASE = 8
RESERVE_RELIEF_PROD_WEIGHT = 4
LOW_HOME_RICH_CORE_DIST = 27.0
LOW_HOME_RICH_CORE_MAX_SHIPS = 30
LOW_HOME_RICH_CORE_MIN_COUNT = 2
LOW_HOME_RICH_CORE_ATTACK_FRACTION = 0.15
MID_HOME_ANCHOR_ATTACK_FRACTION = 0.25
CORE_THREAT_HORIZON = 24
CORE_THREAT_RATIO = 0.52
CORE_URGENT_THREAT_TURN = 8
CORE_URGENT_THREAT_RATIO = 0.70
CORE_VISIBLE_THREAT_TURN = 20
CORE_DEFENSE_SCORE_MULT = 2.2
CORE_URGENT_DEFENSE_SCORE_MULT = 1.35
CORE_RECAPTURE_SCORE_MULT = 1.25
PREDICTED_FLEET_THREAT_HORIZON = 14
SOURCE_LOCK_FALL_TURNS = 14
RECLAIM_HOME_VALUE_MULT = 2.35
THREATENED_CORE_VALUE_MULT = 1.18
ENEMY_OCCUPATION_TURN_TWO_PLAYER = 40
ENEMY_OCCUPATION_TURN_FOUR_PLAYER = 20
ENEMY_OCCUPATION_PREP_TURNS = 10
MIDGAME_NEUTRAL_FIRST_TURN_LIMIT = 120
MIDGAME_NEUTRAL_FIRST_MIN_PROD = 2
MIDGAME_HOSTILE_WITH_NEUTRALS_MULT = 0.62
MIDGAME_CORE_HOSTILE_WITH_NEUTRALS_MULT = 0.82

REINFORCE_ENABLED = True
REINFORCE_MIN_PRODUCTION = 2
REINFORCE_MAX_TRAVEL_TURNS = 22
REINFORCE_SAFETY_MARGIN = 2
REINFORCE_MAX_SOURCE_FRACTION = 0.75
REINFORCE_MIN_FUTURE_TURNS = 40
REINFORCE_HOLD_LOOKAHEAD = 20
REINFORCE_COST_TURN_WEIGHT = 0.35

RECAPTURE_LOOKAHEAD_TURNS = 10
RECAPTURE_COST_TURN_WEIGHT = 0.52
RECAPTURE_VALUE_MULT = 0.88
RECAPTURE_FRONTIER_MULT = 1.08
RECAPTURE_PRODUCTION_WEIGHT = 0.6
RECAPTURE_IMMEDIATE_WEIGHT = 0.4

REAR_SOURCE_MIN_SHIPS = 16
REAR_DISTANCE_RATIO = 1.25
REAR_STAGE_PROGRESS = 0.78
REAR_SEND_RATIO_TWO_PLAYER = 0.62
REAR_SEND_RATIO_FOUR_PLAYER = 0.7
REAR_SEND_MIN_SHIPS = 10
REAR_MAX_TRAVEL_TURNS = 40

PARTIAL_SOURCE_MIN_SHIPS = 6
MULTI_SOURCE_TOP_K = 5
MULTI_SOURCE_ETA_TOLERANCE = 3
MULTI_SOURCE_PLAN_PENALTY = 0.97
HOSTILE_SWARM_ETA_TOLERANCE = 2
ANCHORED_SWARM_ENABLED = True
ANCHORED_SWARM_MAX_SOURCES = 5
ANCHORED_SWARM_MAX_ANCHORS = 4
ANCHORED_SWARM_EXTRA_TOLERANCE = 2
ANCHORED_SWARM_SCORE_MULT = 0.88
ANCHORED_SWARM_MAX_OVERCOMMIT_RATIO = 1.75
ANCHORED_SWARM_MAX_OVERCOMMIT_FLAT = 24
THREE_SOURCE_SWARM_ENABLED = True
THREE_SOURCE_MIN_TARGET_SHIPS = 20
THREE_SOURCE_ETA_TOLERANCE = 2
THREE_SOURCE_PLAN_PENALTY = 0.93
HEAVY_ASSAULT_ENABLED = True
HEAVY_ASSAULT_START_STEP = 45
HEAVY_ASSAULT_MIN_TOTAL = 60
HEAVY_ASSAULT_MIN_SOURCE_SHIPS = 18
HEAVY_ASSAULT_MAX_SOURCES = 5
HEAVY_ASSAULT_MAX_OPTIONS = 3
HEAVY_ASSAULT_ETA_TOLERANCE = 4
HEAVY_ASSAULT_CLUSTER_RADIUS = 20.0
HEAVY_ASSAULT_CLUSTER_PROD_WEIGHT = 0.45
HEAVY_ASSAULT_OVERKILL = 18
HEAVY_ASSAULT_SCORE_MULT = 2.05
DEFICIT_HEAVY_ASSAULT_PROD_GAP = 10
DEFICIT_HEAVY_ASSAULT_MIN_TOTAL_SHIPS = 350
DEFICIT_HEAVY_ASSAULT_SCORE_MULT = 3.0
EARLY_HEAVY_P2_CORE_DIST = 20.0
EARLY_HEAVY_P2_CORE_MAX_SHIPS = 30
EARLY_HEAVY_P2_CHEAP_CORE_DIST = 14.0
EARLY_HEAVY_P2_CHEAP_CORE_SHIPS = 12
EARLY_PROFILE_HEAVY_ASSAULT_START_STEP = 35
EARLY_PROFILE_HEAVY_ASSAULT_MIN_TOTAL = 45
EARLY_PROFILE_HEAVY_ASSAULT_MIN_SOURCE = 14
TURTLE_BREAKOUT_START_STEP = 120
TURTLE_BREAKOUT_MAX_PLANETS = 6
TURTLE_BREAKOUT_MIN_TOTAL = 700
TURTLE_BREAKOUT_PROD_GAP = 10
TURTLE_BREAKOUT_SOURCE_FRACTION = 0.52
TURTLE_BREAKOUT_MIN_SOURCE = 45
TURTLE_BREAKOUT_MAX_SOURCES = 4
TURTLE_BREAKOUT_MIN_TOTAL_SEND = 110
TURTLE_BREAKOUT_OVERKILL = 45
MIDGAME_HOSTILE_MIN_SEND = 20
MIDGAME_CORE_SOURCE_MIN_SEND = 24
TRANSITION_HOSTILE_TURN_LIMIT = 82
TRANSITION_HOSTILE_MIN_SEND = 32
TRANSITION_CORE_HOSTILE_MIN_SEND = 40
TRANSITION_HOSTILE_PROD_LEAD = 6
BEHIND_HOSTILE_BREAK_TURN_LIMIT = 170
BEHIND_HOSTILE_BREAK_MIN_PROD = 3
BEHIND_HOSTILE_BREAK_MIN_SEND = 12
BEHIND_HOSTILE_BREAK_CORE_MIN_SEND = 16
BEHIND_HOSTILE_BREAK_OWNER_PROD_GAP = 8
BEHIND_HOSTILE_BREAK_TOTAL_RATIO = 0.82
BEHIND_SWARM_ETA_BONUS = 1
BEHIND_HOSTILE_SWARM_ETA_BONUS = 1
LEADER_SUPPRESSION_START_STEP = 20
LEADER_SUPPRESSION_PROD_GAP = 2
LEADER_SUPPRESSION_STRENGTH_RATIO = 0.95
LEADER_SUPPRESSION_SCORE_MULT = 4.0
LEADER_SUPPRESSION_CORE_SCORE_MULT = 1.5
FOUR_PLAYER_NON_LEADER_HOSTILE_MULT = 0.35

WAIT_STRIKE_ENABLED = True
WAIT_STRIKE_DELAYS = (0, 2, 4, 6)
WAIT_STRIKE_MAX_TARGETS = 6

FOUR_SOURCE_SWARM_ENABLED = True
FOUR_SOURCE_ETA_TOLERANCE = 2
FOUR_SOURCE_MIN_TARGET_SHIPS = 40
FOUR_SOURCE_PLAN_PENALTY = 0.91

PROACTIVE_DEFENSE_HORIZON = 12
PROACTIVE_DEFENSE_RATIO = 0.18
AGGRESSIVE_DEFENSE_HORIZON = 18
AGGRESSIVE_DEFENSE_RATIO = 0.28
MULTI_ENEMY_STACK_WINDOW = 3
REACTION_SOURCE_TOP_K_MY = 4
REACTION_SOURCE_TOP_K_ENEMY = 4

CRASH_EXPLOIT_ENABLED = True
CRASH_EXPLOIT_MIN_TOTAL_SHIPS = 10
CRASH_EXPLOIT_ETA_WINDOW = 2
CRASH_EXPLOIT_POST_CRASH_DELAY = 1

LATE_IMMEDIATE_SHIP_VALUE = 0.6
WEAK_ENEMY_THRESHOLD = 45
ELIMINATION_BONUS = 18.0

BEHIND_DOMINATION = -0.20
AHEAD_DOMINATION = 0.18
FINISHING_DOMINATION = 0.35
FINISHING_PROD_RATIO = 1.25
AHEAD_ATTACK_MARGIN_BONUS = 0.08
BEHIND_ATTACK_MARGIN_PENALTY = 0.05
FINISHING_ATTACK_MARGIN_BONUS = 0.08

DOOMED_EVAC_TURN_LIMIT = 24
DOOMED_MIN_SHIPS = 8
DOOMED_CORE_THREAT_HORIZON = 5

SOFT_ACT_DEADLINE = 2.0
HEAVY_PHASE_MIN_TIME = 0.16
OPTIONAL_PHASE_MIN_TIME = 0.08
HEAVY_ROUTE_PLANET_LIMIT = 32
TIMEOUT_FALLBACK_START_STEP = 60
TIMEOUT_FALLBACK_MIN_SOURCE = 24
TIMEOUT_FALLBACK_MAX_MOVES = 2
TIMEOUT_FALLBACK_MAX_TARGETS = 8
TIMEOUT_FALLBACK_MAX_TURNS = 36
TIMEOUT_FALLBACK_SEND_RATIO = 0.62
TIMEOUT_FALLBACK_NEUTRAL_MIN_PROD = 3
TIMEOUT_FALLBACK_NEAR_DEADLINE = 0.03
BOXED_BREAKOUT_START_STEP = 60
BOXED_BREAKOUT_MAX_PLANETS = 12
BOXED_BREAKOUT_MIN_SOURCE = 28
BOXED_BREAKOUT_MAX_MOVES = 2
BOXED_BREAKOUT_MAX_TARGETS = 10
BOXED_BREAKOUT_PROD_DEFICIT = 8
BOXED_BREAKOUT_PROD_RATIO = 0.74
BOXED_BREAKOUT_TOTAL_RATIO = 0.72
BOXED_BREAKOUT_KEEP_BASE = 8
BOXED_BREAKOUT_KEEP_PROD_WEIGHT = 3
BOXED_BREAKOUT_KEEP_FRACTION = 0.30
BOXED_BREAKOUT_EVAC_KEEP_FRACTION = 0.05
BOXED_BREAKOUT_SEND_RATIO = 0.65
BOXED_BREAKOUT_MARGIN = 4
BOXED_BREAKOUT_MAX_TURNS = 36
BOXED_BREAKOUT_NEUTRAL_MIN_PROD = 2
BOXED_BREAKOUT_HOSTILE_MIN_PROD = 3
BOXED_BREAKOUT_NEAR_MISS_SHIPS = 8
BOXED_BREAKOUT_NEAR_MISS_RATIO = 0.90
BOXED_BREAKOUT_NEAR_MISS_SCORE_MULT = 0.72
SOFTEN_FALLBACK_ENABLED = True
SOFTEN_FALLBACK_MIN_STEP = 40
SOFTEN_FALLBACK_MAX_STEP = 150
SOFTEN_FALLBACK_MAX_PLANETS = 6
SOFTEN_FALLBACK_MIN_SEND = 6
SOFTEN_FALLBACK_SEND_RATIO = 0.78
SOFTEN_FALLBACK_MIN_PROD = 4
SOFTEN_FALLBACK_MIN_SHIPS = 18
SOFTEN_FALLBACK_MAX_SHIPS = 90
SOFTEN_FALLBACK_MAX_TURNS = 36
SOFTEN_FALLBACK_MAX_DIST = 30.0
SOFTEN_FALLBACK_RACE_ALLOWANCE = 6.0
SOFTEN_FALLBACK_MIN_DAMAGE_RATIO = 0.30
SOFTEN_FALLBACK_HIGH_PROD_DAMAGE_RATIO = 0.24
SOFTEN_FALLBACK_FOLLOWUP_WINDOW = 14
SOFTEN_FALLBACK_FOLLOWUP_MAX_TURNS = 42
SOFTEN_FALLBACK_FOLLOWUP_MAX_DIST = 34.0
SOFTEN_FALLBACK_FOLLOWUP_MIN_CAP_RATIO = 1.05
SOFTEN_FALLBACK_CORE_HOLD_WINDOW = 14
SOFTEN_FALLBACK_CORE_MIN_HOLD_SHIPS = 8
MIDGAME_UNINTENDED_BLOCKER_ENABLED = False
MIDGAME_UNINTENDED_BLOCKER_START_STEP = 40
MIDGAME_UNINTENDED_BLOCKER_TURN_LIMIT = 180
MIDGAME_UNINTENDED_BLOCKER_MAX_PROD = 2
MIDGAME_UNINTENDED_BLOCKER_TARGET_PROD_GAP = 2

ENEMY_PREDICTION_ENABLED = False
ENEMY_PREDICTION_MAX_STEP = 220
ENEMY_PREDICTION_SOURCE_LIMIT = 4
ENEMY_PREDICTION_TARGET_LIMIT = 10
ENEMY_PREDICTION_MIN_SCORE = 0.0

OPENING_LOW_ROI_SCORE_MULT = 0.28
OPENING_WEAK_ROI_SCORE_MULT = 0.58
OPENING_LOW_ROI_MAX_PRODUCTION = 1
OPENING_LOW_ROI_MIN_SHIPS = 18
OPENING_ROI_PRODUCTION_LIMIT = 2
OPENING_WEAK_ROI_MIN_SHIPS = 30
OPENING_ROI_MIN_SCORE = 0.055
OPENING_BRIDGE_RELIEF_RADIUS = 18.0
OPENING_BRIDGE_RELIEF_MULT = 0.72
OPENING_QUALITY_GATE_TURN = 75
OPENING_BETTER_TARGET_DIST = 34.0
OPENING_BETTER_TARGET_MAX_SHIPS = 32
OPENING_BETTER_TARGET_MIN_PROD = 2
OPENING_BAD_TARGET_MAX_PROD = 1
OPENING_BAD_TARGET_MIN_SHIPS = 18
OPENING_COMMITTED_ENABLED = False
OPENING_COMMIT_TURN_LIMIT = 55
OPENING_COMMIT_HOME_PROD_MAX = 1
OPENING_COMMIT_MIN_TARGET_PROD = 2
OPENING_COMMIT_MAX_SHIPS = 32
OPENING_COMMIT_MIN_SCORE = 0.045
OPENING_COMMIT_MAX_DIST = 36.0
OPENING_COMMIT_MIN_RESERVE = 1
OPENING_COMMIT_MARGIN = 1
OPENING_SCORE_DISTANCE_WEIGHT = 0.75
OPENING_SCORE_SHIP_WEIGHT = 1.0
OPENING_SCORE_BASE_COST = 4.0
OPENING_SCORE_STATIC_MULT = 1.08
OPENING_SCORE_PROD_POWER = 1.0
OPENING_SCORE_BRIDGE_RADIUS = 24.0
OPENING_SCORE_BRIDGE_MULT = 0.55
OPENING_SCORE_LOW_PROD_BRIDGE_MULT = 0.18
OPENING_SCORE_BRIDGE_MIN_TARGET_PROD = 3
OPENING_SCORE_LOW_VALUE_RATIO = 0.55
OPENING_FAR_NEUTRAL_TURN_LIMIT = 60
OPENING_FAR_NEUTRAL_DIST = 44.0
OPENING_FAR_NEUTRAL_TURNS = 20
OPENING_FAR_NEUTRAL_LOCAL_ADVANTAGE = 1.12
OPENING_FILL_STEP = 35
OPENING_FILL_MIN_PROD = 18
OPENING_FILL_MIN_PLANETS = 5
OPENING_FILL_MAX_TARGET_SHIPS = 35
OPENING_FILL_MAX_DIST = 28.0
OPENING_FILL_LOW_PROD_MAX_NEEDED = 26
OPENING_FILL_LOW_PROD_MAX_OVER_NEUTRAL = 16
OPENING_FILL_WEAK_PROD_MAX_NEEDED = 42
OPENING_LOW_PROD_HEAVY_TARGET_SHIPS = 35
OPENING_HEAVY_PRIZE_MIN_PROD = 4
OPENING_HEAVY_PRIZE_MIN_SHIPS = 36
OPENING_HEAVY_PRIZE_SCORE_MULT = 1.45
OPENING_HEAVY_PRIZE_PLAN_START = 22
OPENING_HEAVY_PRIZE_PLAN_END = 68
OPENING_HEAVY_PRIZE_MAX_SOURCES = 4
OPENING_HEAVY_PRIZE_MIN_SOURCE = 8
OPENING_HEAVY_PRIZE_RESERVE_BASE = 4
OPENING_HEAVY_PRIZE_RESERVE_PROD = 2
OPENING_HEAVY_PRIZE_MARGIN = 6
OPENING_HEAVY_PRIZE_SOURCE_THREAT_HORIZON = 5
OPENING_PRIORITY_ENABLED = True
OPENING_PRIORITY_TURN_LIMIT = 58
OPENING_PRIORITY_WAIT_TURN_LIMIT = 42
OPENING_PRIORITY_MAX_PLANETS = 3
OPENING_PRIORITY_MAX_WAIT = 14
OPENING_PRIORITY_HOME_PROD_MIN = 2
OPENING_PRIORITY_HOME_PROD_MAX = 3
OPENING_PRIORITY_MIN_TARGET_PROD = 2
OPENING_PRIORITY_MIN_SCORE = 0.060
OPENING_PRIORITY_HOME_KEEP = 2
OPENING_PRIORITY_CORE_KEEP = 3
OPENING_PRIORITY_WAIT_ADVANTAGE = 1.08
OPENING_PRIORITY_WAIT_PENALTY = 0.055
OPENING_PRIORITY_MARGIN_CAP = 5
OPENING_PRIORITY_ALT_AFFORDABLE_RATIO = 0.88
OPENING_PRIORITY_TRAP_MAX_DIST = 15.0
OPENING_PRIORITY_TRAP_MIN_SHIPS = 20
OPENING_PRIORITY_PRIZE_MAX_DIST = 24.0
OPENING_PRIORITY_PRIZE_MAX_SHIPS = 20
OPENING_PRIORITY_PRIZE_MIN_PROD = 4
OPENING_ANCHOR_ENABLED = True
OPENING_ANCHOR_TURN_LIMIT = 18
OPENING_ANCHOR_MAX_PLANETS = 1
OPENING_ANCHOR_MAX_WAIT = 10
OPENING_ANCHOR_HOME_KEEP = 1
OPENING_ANCHOR_MIN_HOME_PROD = 3
OPENING_ANCHOR_MIN_PROD = 5
OPENING_ANCHOR_MIN_SHIPS = 20
OPENING_ANCHOR_MAX_SHIPS = 32
OPENING_ANCHOR_MAX_DIST = 13.0
OPENING_ANCHOR_FAST_ALT_PROD = 4
OPENING_ANCHOR_FAST_ALT_MAX_SHIPS = 12
OPENING_ANCHOR_FAST_ALT_DIST = 17.0
OPENING_ANCHOR_MARGIN = 3
OPENING_MAINLINE_ENABLED = True
OPENING_MAINLINE_TURN_LIMIT = 42
OPENING_MAINLINE_WAIT_TURN_LIMIT = 30
OPENING_MAINLINE_MAX_PLANETS = 1
OPENING_MAINLINE_MAX_WAIT = 16
OPENING_MAINLINE_HOME_KEEP = 1
OPENING_MAINLINE_CORE_KEEP = 3
OPENING_MAINLINE_MIN_TARGET_PROD = 2
OPENING_MAINLINE_MARGIN = 2
OPENING_MAINLINE_CORE_EXTRA_MARGIN = 2
OPENING_MAINLINE_MIN_SCORE = 0.040
OPENING_MAINLINE_FALLBACK_LOW_MIN_SCORE = 0.005
OPENING_MAINLINE_WAIT_PENALTY = 0.060
OPENING_MAINLINE_ALT_AFFORDABLE_RATIO = 0.84
OPENING_MAINLINE_PROD_POWER = 1.35
OPENING_MAINLINE_SHIP_WEIGHT = 1.0
OPENING_MAINLINE_TURN_WEIGHT = 0.70
OPENING_MAINLINE_WAIT_WEIGHT = 1.8
OPENING_MAINLINE_BASE_COST = 4.0
OPENING_MAINLINE_LOW_PROD_MULT = 0.22
OPENING_MAINLINE_LOW_HOME_CHEAP_P2_DIST = 24.0
OPENING_MAINLINE_LOW_HOME_CHEAP_P2_SHIPS = 14
OPENING_MAINLINE_LOW_HOME_TWO_STEP_DIST = 23.0
OPENING_MAINLINE_LOW_HOME_TWO_STEP_SHIPS = 12
OPENING_MAINLINE_CLOSE_CORE_DIST = 15.0
OPENING_MAINLINE_CLOSE_CORE_SHIPS = 24
OPENING_MAINLINE_P2_TRAP_DIST = 16.0
OPENING_MAINLINE_P2_TRAP_SHIPS = 10
OPENING_MAINLINE_P2_PRIZE_DIST = 30.0
OPENING_MAINLINE_P2_PRIZE_SHIPS = 10
OPENING_MAINLINE_P2_FAST_P5_DIST = 25.0
OPENING_MAINLINE_P2_FAST_P5_SHIPS = 10
OPENING_MAINLINE_P2_FALLBACK_LOW_DIST = 16.0
OPENING_MAINLINE_P2_FALLBACK_LOW_SHIPS = 10
OPENING_MAINLINE_P2_QUALITY_DIST = 35.0
OPENING_MAINLINE_P2_QUALITY_SHIPS = 30
OPENING_MAINLINE_P3_CORE_DIST = 24.0
OPENING_MAINLINE_P3_CORE_MIN_DIST = 18.0
OPENING_MAINLINE_P3_CORE_SHIPS = 28
OPENING_MAINLINE_LOW_HOME_HEAVY_CORE_DIST = 16.0
OPENING_MAINLINE_LOW_HOME_HEAVY_CORE_SHIPS = 36
OPENING_MAINLINE_LOW_HOME_HEAVY_CORE_PROD = 4
OPENING_MAINLINE_LOW_HOME_HEAVY_MAX_WAIT = 30
OPENING_FAST_EXPAND_ENABLED = True
OPENING_FAST_EXPAND_TURN_LIMIT = 30
OPENING_FAST_EXPAND_MAX_PLANETS = 3
OPENING_FAST_EXPAND_EARLY_HOME_KEEP = 0
OPENING_FAST_EXPAND_HOME_KEEP = 1
OPENING_FAST_EXPAND_SOURCE_KEEP = 0
OPENING_FAST_EXPAND_MIN_SOURCE = 4
OPENING_FAST_EXPAND_MIN_TARGET_PROD = 1
OPENING_FAST_EXPAND_MAX_TARGET_SHIPS = 28
OPENING_FAST_EXPAND_MAX_DIST = 32.0
OPENING_FAST_EXPAND_MAX_TURNS = 18
OPENING_FAST_EXPAND_MARGIN = 1
OPENING_FAST_EXPAND_CORE_MARGIN = 2
OPENING_FAST_EXPAND_RACE_ALLOWANCE = 5.0
OPENING_FAST_EXPAND_LOW_HOME_P2_BONUS = 1.55
OPENING_FAST_EXPAND_LOW_TARGET_BLOCK_CORE_PROD = 4
OPENING_FAST_EXPAND_LOW_TARGET_BLOCK_CORE_SHIPS = 30
OPENING_FAST_EXPAND_LOW_TARGET_BLOCK_CORE_DIST = 24.0
OPENING_FAST_EXPAND_SWARM_ENABLED = True
OPENING_FAST_EXPAND_SWARM_MIN_PLANETS = 2
OPENING_FAST_EXPAND_SWARM_MAX_SOURCES = 3
OPENING_FAST_EXPAND_SWARM_MIN_TARGET_PROD = 3
OPENING_FAST_EXPAND_SWARM_MAX_TARGET_SHIPS = 30
OPENING_FAST_EXPAND_SWARM_MAX_DIST = 36.0
OPENING_FAST_EXPAND_SWARM_MAX_TURNS = 22
OPENING_FAST_EXPAND_SWARM_ETA_TOLERANCE = 4
OPENING_FAST_EXPAND_SWARM_OVERKILL = 2
OPENING_LOCAL_QUALITY_ENABLED = True
OPENING_LOCAL_QUALITY_TURN_LIMIT = 34
OPENING_LOCAL_QUALITY_MAX_PLANETS = 1
OPENING_LOCAL_QUALITY_MAX_WAIT = 14
OPENING_LOCAL_QUALITY_HOME_KEEP = 1
OPENING_LOCAL_QUALITY_CORE_KEEP = 3
OPENING_LOCAL_QUALITY_MARGIN = 2
OPENING_LOCAL_QUALITY_CORE_MARGIN = 3
OPENING_LOCAL_LOW_HOME_CORE_DIST = 22.0
OPENING_LOCAL_LOW_HOME_CORE_SHIPS = 22
OPENING_LOCAL_LOW_HOME_P3_DIST = 16.0
OPENING_LOCAL_LOW_HOME_P3_SHIPS = 22
OPENING_LOCAL_LOW_HOME_CHEAP_P5_DIST = 12.0
OPENING_LOCAL_LOW_HOME_CHEAP_P5_SHIPS = 12
OPENING_LOCAL_P2_CORE_DIST = 20.0
OPENING_LOCAL_P2_CORE_SHIPS = 30
OPENING_LOCAL_P2_CORE_PROD = 4
OPENING_LOCAL_HIGH_HOME_CORE_DIST = 22.0
OPENING_LOCAL_HIGH_HOME_CORE_SHIPS = 18
OPENING_LOCAL_MID_HOME_P3_DIST = 24.0
OPENING_LOCAL_MID_HOME_P3_SHIPS = 16
OPENING_LOCAL_MID_HOME_CORE_BLOCK_DIST = 18.0
OPENING_LOCAL_MID_HOME_CORE_BLOCK_SHIPS = 18
OPENING_LOCAL_MID_HOME_CLOSE_CORE_DIST = 14.0
OPENING_LOCAL_MID_HOME_CLOSE_CORE_SHIPS = 30
OPENING_DIRECT_ENABLED = True
OPENING_DIRECT_TURN_LIMIT = 5
OPENING_DIRECT_MAX_PLANETS = 1
OPENING_DIRECT_TWO_PLAYER_MAX_WAIT = 5
OPENING_DIRECT_FOUR_PLAYER_MAX_WAIT = 3
OPENING_DIRECT_TWO_PLAYER_MAX_TURNS = 30
OPENING_DIRECT_FOUR_PLAYER_MAX_TURNS = 16
OPENING_DIRECT_MAX_DIST = 40.0
OPENING_DIRECT_MOVING_MAX_DIST_TWO_PLAYER = 110.0
OPENING_DIRECT_MAX_TARGET_SHIPS = 34
OPENING_DIRECT_MIN_TARGET_PROD = 2
OPENING_DIRECT_MARGIN = 1
OPENING_DIRECT_CORE_MARGIN = 2
OPENING_DIRECT_AFFORDABLE_RATIO = 0.72
OPENING_DIRECT_RACE_ALLOWANCE_TWO_PLAYER = 12.0
OPENING_DIRECT_RACE_ALLOWANCE_FOUR_PLAYER = 4.0
OPENING_META_WAIT_SCORE = 0.038
OPENING_META_MIN_ACTION_SCORE = 0.018
OPENING_META_LOW_PROD_OVERSPEND = 12
OPENING_META_HOME_EMPTY_PENALTY = 0.72
OPENING_META_CORE_CAPTURE_BONUS = 1.22
OPENING_META_STATIC_CAPTURE_BONUS = 1.06
OPENING_META_UNCAPTURED_PENALTY = 0.42
OPENING_META_ENEMY_RACE_PENALTY = 0.64
OPENING_HOME_HALF_CORE_MULT = 1.28
OPENING_AWAY_HALF_CORE_PENALTY = 0.52
OPENING_HOME_HALF_CORE_SHIP_ALLOWANCE = 4
OPENING_META_STAGE_WEIGHTS = {
    "opening_direct": 1.08,
    "opening_fast_expand": 1.00,
    "opening_anchor": 1.06,
    "opening_priority": 1.05,
    "opening_local_quality": 1.04,
    "opening_mainline": 1.02,
    "opening_heavy_prize": 1.10,
}


# ============================================================
# Shared Types
# ============================================================

Planet = namedtuple(
    "Planet", ["id", "owner", "x", "y", "radius", "ships", "production"]
)
Fleet = namedtuple(
    "Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"]
)


@dataclass(frozen=True)
class ShotOption:
    score: float
    src_id: int
    target_id: int
    angle: float
    turns: int
    needed: int
    send_cap: int
    mission: str = "capture"
    anchor_turn: int | None = None


@dataclass
class Mission:
    kind: str
    score: float
    target_id: int
    turns: int
    options: list[ShotOption] = field(default_factory=list)
    min_total: int = 0

# ============================================================
# Physics
# ============================================================

def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def orbital_radius(planet):
    return dist(planet.x, planet.y, CENTER_X, CENTER_Y)


def is_static_planet(planet):
    return orbital_radius(planet) + planet.radius >= ROTATION_LIMIT


def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio**1.5)


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-9:
        return dist(px, py, x1, y1)
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return dist(px, py, proj_x, proj_y)


def segment_hits_sun(x1, y1, x2, y2, safety=SUN_SAFETY):
    return point_to_segment_distance(CENTER_X, CENTER_Y, x1, y1, x2, y2) < SUN_R + safety


def launch_point(sx, sy, sr, angle):
    clearance = sr + LAUNCH_CLEARANCE
    return sx + math.cos(angle) * clearance, sy + math.sin(angle) * clearance


def ray_circle_hit_distance(start_x, start_y, angle, cx, cy, radius):
    dir_x = math.cos(angle)
    dir_y = math.sin(angle)
    ox = start_x - cx
    oy = start_y - cy
    b = 2.0 * (ox * dir_x + oy * dir_y)
    c = ox * ox + oy * oy - radius * radius
    disc = b * b - 4.0 * c
    if disc < 0:
        return None
    root = math.sqrt(max(0.0, disc))
    hits = [(-b - root) / 2.0, (-b + root) / 2.0]
    forward_hits = [hit for hit in hits if hit >= 0.0]
    if not forward_hits:
        return None
    return min(forward_hits)


def ray_board_exit_distance(start_x, start_y, angle):
    dir_x = math.cos(angle)
    dir_y = math.sin(angle)
    hits = []
    if abs(dir_x) > 1e-9:
        for boundary_x in (0.0, BOARD):
            t = (boundary_x - start_x) / dir_x
            if t >= 0.0:
                y = start_y + dir_y * t
                if -PATH_BLOCKER_EPSILON <= y <= BOARD + PATH_BLOCKER_EPSILON:
                    hits.append(t)
    if abs(dir_y) > 1e-9:
        for boundary_y in (0.0, BOARD):
            t = (boundary_y - start_y) / dir_y
            if t >= 0.0:
                x = start_x + dir_x * t
                if -PATH_BLOCKER_EPSILON <= x <= BOARD + PATH_BLOCKER_EPSILON:
                    hits.append(t)
    if not hits:
        return None
    return min(hits)


def path_geometry_for_angle(sx, sy, sr, tx, ty, tr, angle):
    start_x, start_y = launch_point(sx, sy, sr, angle)
    hit_distance = ray_circle_hit_distance(start_x, start_y, angle, tx, ty, tr)
    if hit_distance is None:
        return None
    end_x = start_x + math.cos(angle) * hit_distance
    end_y = start_y + math.sin(angle) * hit_distance
    return angle, start_x, start_y, end_x, end_y, hit_distance


def actual_path_geometry(sx, sy, sr, tx, ty, tr):
    angle = math.atan2(ty - sy, tx - sx)
    if EDGE_AIM_ENABLED:
        geometry = path_geometry_for_angle(sx, sy, sr, tx, ty, tr, angle)
        if geometry is not None:
            return geometry
    start_x, start_y = launch_point(sx, sy, sr, angle)
    hit_distance = max(0.0, dist(sx, sy, tx, ty) - (sr + LAUNCH_CLEARANCE) - tr)
    end_x = start_x + math.cos(angle) * hit_distance
    end_y = start_y + math.sin(angle) * hit_distance
    return angle, start_x, start_y, end_x, end_y, hit_distance


def candidate_path_geometries(sx, sy, sr, tx, ty, tr):
    base_angle = math.atan2(ty - sy, tx - sx)
    normal_x = -math.sin(base_angle)
    normal_y = math.cos(base_angle)
    aim_span = max(0.0, tr - 0.02)

    for offset in ROUTE_AIM_OFFSETS:
        aim_x = tx + normal_x * aim_span * offset
        aim_y = ty + normal_y * aim_span * offset
        angle = math.atan2(aim_y - sy, aim_x - sx)
        geometry = path_geometry_for_angle(sx, sy, sr, tx, ty, tr, angle)
        if geometry is not None:
            yield offset, geometry


def safe_angle_and_distance(sx, sy, sr, tx, ty, tr):
    # Launch from the source boundary and time the route to the first hit on
    # the target circle.
    angle, start_x, start_y, end_x, end_y, hit_distance = actual_path_geometry(
        sx,
        sy,
        sr,
        tx,
        ty,
        tr,
    )
    if not segment_hits_sun(start_x, start_y, end_x, end_y):
        return angle, hit_distance

    if not EDGE_AIM_ENABLED:
        return None

    best = None
    best_key = None
    for offset, geometry in candidate_path_geometries(sx, sy, sr, tx, ty, tr):
        angle, start_x, start_y, end_x, end_y, hit_distance = geometry
        sun_clearance = point_to_segment_distance(
            CENTER_X,
            CENTER_Y,
            start_x,
            start_y,
            end_x,
            end_y,
        ) - SUN_R
        if sun_clearance < SUN_SAFETY:
            continue
        key = (hit_distance, abs(offset), -sun_clearance)
        if best_key is None or key < best_key:
            best_key = key
            best = (angle, hit_distance)
    if best is None:
        return None
    return best


def predict_planet_position(planet, initial_by_id, angular_velocity, turns):
    init = initial_by_id.get(planet.id)
    if init is None:
        return planet.x, planet.y
    r = dist(init.x, init.y, CENTER_X, CENTER_Y)
    if r + init.radius >= ROTATION_LIMIT:
        return planet.x, planet.y
    cur_ang = math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X)
    new_ang = cur_ang + angular_velocity * turns
    return (
        CENTER_X + r * math.cos(new_ang),
        CENTER_Y + r * math.sin(new_ang),
    )


def predict_comet_position(planet_id, comets, turns):
    for group in comets:
        pids = group.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", [])
        path_index = group.get("path_index", 0)
        if idx >= len(paths):
            return None
        path = paths[idx]
        future_idx = path_index + int(turns)
        if 0 <= future_idx < len(path):
            return path[future_idx][0], path[future_idx][1]
        return None
    return None


def comet_remaining_life(planet_id, comets):
    for group in comets:
        pids = group.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", [])
        path_index = group.get("path_index", 0)
        if idx < len(paths):
            return max(0, len(paths[idx]) - path_index)
    return 0


def estimate_arrival(sx, sy, sr, tx, ty, tr, ships):
    # Use one boundary-aware ETA model for routing, ranking, reserve, and
    # launch decisions.
    safe = safe_angle_and_distance(sx, sy, sr, tx, ty, tr)
    if safe is None:
        return None
    angle, total_d = safe
    turns = max(1, int(math.ceil(total_d / fleet_speed(max(1, ships)))))
    return angle, turns


def travel_time(sx, sy, sr, tx, ty, tr, ships):
    est = estimate_arrival(sx, sy, sr, tx, ty, tr, ships)
    if est is None:
        return 10**9
    return est[1]


def predict_target_position(target, turns, initial_by_id, ang_vel, comets, comet_ids):
    if target.id in comet_ids:
        return predict_comet_position(target.id, comets, turns)
    return predict_planet_position(target, initial_by_id, ang_vel, turns)


def target_can_move(target, initial_by_id, comet_ids):
    if target.id in comet_ids:
        return True
    init = initial_by_id.get(target.id)
    if init is None:
        return False
    r = dist(init.x, init.y, CENTER_X, CENTER_Y)
    return r + init.radius < ROTATION_LIMIT


def search_safe_intercept(src, target, ships, initial_by_id, ang_vel, comets, comet_ids):
    # If the direct line is unsafe, scan future positions and keep the earliest
    # viable intercept window.
    best = None
    best_score = None
    max_turns = min(HORIZON, ROUTE_SEARCH_HORIZON)
    if target.id in comet_ids:
        max_turns = min(max_turns, max(0, comet_remaining_life(target.id, comets) - 1))

    for candidate_turns in range(1, max_turns + 1):
        pos = predict_target_position(
            target,
            candidate_turns,
            initial_by_id,
            ang_vel,
            comets,
            comet_ids,
        )
        if pos is None:
            continue
        est = estimate_arrival(src.x, src.y, src.radius, pos[0], pos[1], target.radius, ships)
        if est is None:
            continue
        _, turns = est
        if abs(turns - candidate_turns) > INTERCEPT_TOLERANCE:
            continue

        actual_turns = max(turns, candidate_turns)
        actual_pos = predict_target_position(
            target,
            actual_turns,
            initial_by_id,
            ang_vel,
            comets,
            comet_ids,
        )
        if actual_pos is None:
            continue

        confirm = estimate_arrival(
            src.x,
            src.y,
            src.radius,
            actual_pos[0],
            actual_pos[1],
            target.radius,
            ships,
        )
        if confirm is None:
            continue

        delta = abs(confirm[1] - actual_turns)
        if delta > INTERCEPT_TOLERANCE:
            continue

        score = (delta, confirm[1], candidate_turns)
        if best is None or score < best_score:
            best_score = score
            best = (confirm[0], confirm[1], actual_pos[0], actual_pos[1])

    return best


def aim_with_prediction(src, target, ships, initial_by_id, ang_vel, comets, comet_ids):
    # Iterate toward a self-consistent moving-target intercept, then fall back
    # to a later safe window if needed.
    est = estimate_arrival(src.x, src.y, src.radius, target.x, target.y, target.radius, ships)
    if est is None:
        if not target_can_move(target, initial_by_id, comet_ids):
            return None
        return search_safe_intercept(
            src,
            target,
            ships,
            initial_by_id,
            ang_vel,
            comets,
            comet_ids,
        )

    tx, ty = target.x, target.y
    for _ in range(5):
        _, turns = est
        pos = predict_target_position(target, turns, initial_by_id, ang_vel, comets, comet_ids)
        if pos is None:
            return None
        ntx, nty = pos
        next_est = estimate_arrival(src.x, src.y, src.radius, ntx, nty, target.radius, ships)
        if next_est is None:
            if not target_can_move(target, initial_by_id, comet_ids):
                return None
            return search_safe_intercept(
                src,
                target,
                ships,
                initial_by_id,
                ang_vel,
                comets,
                comet_ids,
            )
        if (
            abs(ntx - tx) < 0.3
            and abs(nty - ty) < 0.3
            and abs(next_est[1] - turns) <= INTERCEPT_TOLERANCE
        ):
            return next_est[0], next_est[1], ntx, nty
        tx, ty = ntx, nty
        est = next_est

    final_est = estimate_arrival(src.x, src.y, src.radius, tx, ty, target.radius, ships)
    if final_est is None:
        return search_safe_intercept(
            src,
            target,
            ships,
            initial_by_id,
            ang_vel,
            comets,
            comet_ids,
        )
    return final_est[0], final_est[1], tx, ty

# ============================================================
# World Model
# ============================================================

def fleet_target_planet(fleet, planets):
    # Project in-flight fleets by ray-circle hit timing to build a usable
    # arrival ledger.  Keep this lightweight: the ledger is rebuilt every
    # decision, and over-modeling it starves higher-value planning phases.
    best_planet = None
    best_time = 1e9
    dir_x = math.cos(fleet.angle)
    dir_y = math.sin(fleet.angle)
    speed = fleet_speed(fleet.ships)

    for planet in planets:
        dx = planet.x - fleet.x
        dy = planet.y - fleet.y
        proj = dx * dir_x + dy * dir_y
        if proj < 0:
            continue
        perp_sq = dx * dx + dy * dy - proj * proj
        radius_sq = planet.radius * planet.radius
        if perp_sq >= radius_sq:
            continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, radius_sq - perp_sq)))
        turns = hit_d / speed
        if turns <= HORIZON and turns < best_time:
            best_time = turns
            best_planet = planet

    if best_planet is None:
        return None, None
    return best_planet, int(math.ceil(best_time))


def fleet_target_planet_predictive(fleet, planets, initial_by_id, ang_vel, comets, comet_ids):
    dir_x = math.cos(fleet.angle)
    dir_y = math.sin(fleet.angle)
    speed = fleet_speed(fleet.ships)
    prev_x, prev_y = fleet.x, fleet.y
    best = None

    for turn in range(1, ROUTE_SEARCH_HORIZON + 1):
        cur_x = fleet.x + dir_x * speed * turn
        cur_y = fleet.y + dir_y * speed * turn
        if cur_x < 0.0 or cur_x > BOARD or cur_y < 0.0 or cur_y > BOARD:
            break

        for planet in planets:
            pos = predict_target_position(
                planet,
                turn,
                initial_by_id,
                ang_vel,
                comets,
                comet_ids,
            )
            if pos is None:
                continue
            if point_to_segment_distance(
                pos[0],
                pos[1],
                prev_x,
                prev_y,
                cur_x,
                cur_y,
            ) <= planet.radius:
                distance_key = point_to_segment_distance(
                    pos[0],
                    pos[1],
                    fleet.x,
                    fleet.y,
                    cur_x,
                    cur_y,
                )
                key = (turn, distance_key, planet.id)
                if best is None or key < best[0]:
                    best = (key, planet)
        if best is not None and best[0][0] <= turn:
            return best[1], best[0][0]
        prev_x, prev_y = cur_x, cur_y

    return fleet_target_planet(fleet, planets)


def build_arrival_ledger(fleets, planets, initial_by_id, ang_vel, comets, comet_ids):
    arrivals_by_planet = {planet.id: [] for planet in planets}
    for fleet in fleets:
        target, eta = fleet_target_planet_predictive(
            fleet,
            planets,
            initial_by_id,
            ang_vel,
            comets,
            comet_ids,
        )
        if target is None:
            continue
        arrivals_by_planet[target.id].append((eta, fleet.owner, int(fleet.ships)))
    return arrivals_by_planet


def resolve_arrival_event(owner, garrison, arrivals):
    # Match the environment's same-turn combat order: aggregate by owner, let
    # the top two attackers cancel, then resolve the survivor against garrison.
    by_owner = {}
    for _, attacker_owner, ships in arrivals:
        by_owner[attacker_owner] = by_owner.get(attacker_owner, 0) + ships

    if not by_owner:
        return owner, max(0.0, garrison)

    sorted_players = sorted(by_owner.items(), key=lambda item: item[1], reverse=True)
    top_owner, top_ships = sorted_players[0]

    if len(sorted_players) > 1:
        second_ships = sorted_players[1][1]
        if top_ships == second_ships:
            survivor_owner = -1
            survivor_ships = 0
        else:
            survivor_owner = top_owner
            survivor_ships = top_ships - second_ships
    else:
        survivor_owner = top_owner
        survivor_ships = top_ships

    if survivor_ships <= 0:
        return owner, max(0.0, garrison)

    if owner == survivor_owner:
        return owner, garrison + survivor_ships

    garrison -= survivor_ships
    if garrison < 0:
        return survivor_owner, -garrison
    return owner, garrison


def normalize_arrivals(arrivals, horizon):
    events = []
    for turns, owner, ships in arrivals:
        if ships <= 0:
            continue
        eta = max(1, int(math.ceil(turns)))
        if eta > horizon:
            continue
        events.append((eta, owner, int(ships)))
    events.sort(key=lambda item: item[0])
    return events


def simulate_planet_timeline(planet, arrivals, player, horizon):
    # Build one reusable future timeline so defense, capture, and evacuation
    # all query the same state model.
    horizon = max(0, int(math.ceil(horizon)))
    events = normalize_arrivals(arrivals, horizon)
    by_turn = defaultdict(list)
    for item in events:
        by_turn[item[0]].append(item)

    owner = planet.owner
    garrison = float(planet.ships)
    owner_at = {0: owner}
    ships_at = {0: max(0.0, garrison)}
    min_owned = garrison if owner == player else 0.0
    first_enemy = None
    fall_turn = None

    for turn in range(1, horizon + 1):
        if owner != -1:
            garrison += planet.production

        group = by_turn.get(turn, [])
        prev_owner = owner
        if group:
            if prev_owner == player and first_enemy is None:
                if any(item[1] not in (-1, player) for item in group):
                    first_enemy = turn
            owner, garrison = resolve_arrival_event(owner, garrison, group)
            if prev_owner == player and owner != player and fall_turn is None:
                fall_turn = turn

        owner_at[turn] = owner
        ships_at[turn] = max(0.0, garrison)
        if owner == player:
            min_owned = min(min_owned, garrison)

    keep_needed = 0
    holds_full = True

    if planet.owner == player:

        def survives_with_keep(keep):
            sim_owner = planet.owner
            sim_garrison = float(keep)
            for turn in range(1, horizon + 1):
                if sim_owner != -1:
                    sim_garrison += planet.production
                group = by_turn.get(turn, [])
                if group:
                    sim_owner, sim_garrison = resolve_arrival_event(sim_owner, sim_garrison, group)
                    if sim_owner != player:
                        return False
            return sim_owner == player

        if survives_with_keep(int(planet.ships)):
            lo, hi = 0, int(planet.ships)
            while lo < hi:
                mid = (lo + hi) // 2
                if survives_with_keep(mid):
                    hi = mid
                else:
                    lo = mid + 1
            keep_needed = lo
        else:
            holds_full = False
            keep_needed = int(planet.ships)

    return {
        "owner_at": owner_at,
        "ships_at": ships_at,
        "keep_needed": keep_needed,
        "min_owned": max(0, int(math.floor(min_owned))) if planet.owner == player else 0,
        "first_enemy": first_enemy,
        "fall_turn": fall_turn,
        "holds_full": holds_full,
        "horizon": horizon,
    }


def state_at_timeline(timeline, arrival_turn):
    turn = max(0, int(math.ceil(arrival_turn)))
    turn = min(turn, timeline["horizon"])
    owner = timeline["owner_at"].get(turn, timeline["owner_at"][timeline["horizon"]])
    ships = timeline["ships_at"].get(turn, timeline["ships_at"][timeline["horizon"]])
    return owner, max(0.0, ships)


def count_players(planets, fleets):
    owners = set()
    for planet in planets:
        if planet.owner != -1:
            owners.add(planet.owner)
    for fleet in fleets:
        owners.add(fleet.owner)
    return max(2, len(owners))


def nearest_distance_to_set(px, py, planets):
    if not planets:
        return 10**9
    return min(dist(px, py, planet.x, planet.y) for planet in planets)


def indirect_features(planet, planets, player):
    friendly = 0.0
    neutral = 0.0
    enemy = 0.0
    for other in planets:
        if other.id == planet.id:
            continue
        d = dist(planet.x, planet.y, other.x, other.y)
        if d < 1:
            continue
        factor = other.production / (d + 12.0)
        if other.owner == player:
            friendly += factor
        elif other.owner == -1:
            neutral += factor
        else:
            enemy += factor
    return friendly, neutral, enemy


class WorldModel:
    def __init__(self, player, step, planets, fleets, initial_by_id, ang_vel, comets, comet_ids):
        self.player = player
        self.step = step
        self.planets = planets
        self.fleets = fleets
        self.initial_by_id = initial_by_id
        self.ang_vel = ang_vel
        self.comets = comets
        self.comet_ids = set(comet_ids)

        self.planet_by_id = {planet.id: planet for planet in planets}
        self.my_planets = [planet for planet in planets if planet.owner == player]
        self.enemy_planets = [planet for planet in planets if planet.owner not in (-1, player)]
        self.neutral_planets = [planet for planet in planets if planet.owner == -1]
        self.static_neutral_planets = [
            planet for planet in self.neutral_planets if is_static_planet(planet)
        ]

        self.num_players = count_players(planets, fleets)
        self.remaining_steps = max(1, TOTAL_STEPS - step)
        self.is_early = step < EARLY_TURN_LIMIT
        self.is_opening = step < OPENING_TURN_LIMIT
        self.is_late = self.remaining_steps < LATE_REMAINING_TURNS
        self.is_very_late = self.remaining_steps < VERY_LATE_REMAINING_TURNS
        self.is_four_player = self.num_players >= 4
        self.timing_enabled = _truthy_env(
            "ORBIT_TIMING",
            default=_truthy_env("ORBIT_LOG", default=True),
        )
        self.timing_records = {}

        self.owner_strength = defaultdict(int)
        self.owner_production = defaultdict(int)
        for planet in planets:
            if planet.owner != -1:
                self.owner_strength[planet.owner] += int(planet.ships)
                self.owner_production[planet.owner] += int(planet.production)
        for fleet in fleets:
            self.owner_strength[fleet.owner] += int(fleet.ships)

        self.my_total = self.owner_strength.get(player, 0)
        self.enemy_owners = [
            owner
            for owner in self.owner_strength
            if owner not in (-1, player)
        ]
        self.enemy_total = sum(
            strength for owner, strength in self.owner_strength.items() if owner != player
        )
        self.max_enemy_strength = max(
            (strength for owner, strength in self.owner_strength.items() if owner != player),
            default=0,
        )
        self.my_prod = self.owner_production.get(player, 0)
        self.enemy_prod = sum(
            production
            for owner, production in self.owner_production.items()
            if owner != player
        )
        self.leading_enemy_owner = max(
            self.enemy_owners,
            key=lambda owner: (
                self.owner_production.get(owner, 0),
                self.owner_strength.get(owner, 0),
            ),
            default=None,
        )
        self.leading_enemy_prod = (
            self.owner_production.get(self.leading_enemy_owner, 0)
            if self.leading_enemy_owner is not None
            else 0
        )
        self.leading_enemy_strength = (
            self.owner_strength.get(self.leading_enemy_owner, 0)
            if self.leading_enemy_owner is not None
            else 0
        )

        self.arrivals_by_planet = build_arrival_ledger(
            fleets,
            planets,
            initial_by_id,
            ang_vel,
            comets,
            self.comet_ids,
        )
        self.base_timeline = {
            planet.id: simulate_planet_timeline(
                planet,
                self.arrivals_by_planet[planet.id],
                player,
                HORIZON,
            )
            for planet in planets
        }
        self.keep_needed_map = {
            planet.id: self.base_timeline[planet.id]["keep_needed"] for planet in planets
        }
        self.min_owned_map = {
            planet.id: self.base_timeline[planet.id]["min_owned"] for planet in planets
        }
        self.first_enemy_map = {
            planet.id: self.base_timeline[planet.id]["first_enemy"] for planet in planets
        }
        self.fall_turn_map = {
            planet.id: self.base_timeline[planet.id]["fall_turn"] for planet in planets
        }
        self.holds_full_map = {
            planet.id: self.base_timeline[planet.id]["holds_full"] for planet in planets
        }
        self.indirect_feature_map = {
            planet.id: indirect_features(planet, planets, player) for planet in planets
        }
        self.shot_cache = {}
        self.probe_candidate_cache = {}
        self.best_probe_cache = {}
        self.reaction_cache = {}
        self.exact_need_cache = {}

        self.total_visible_ships = sum(int(planet.ships) for planet in planets) + sum(
            int(fleet.ships) for fleet in fleets
        )
        self.total_production = sum(int(planet.production) for planet in planets)

    def is_static(self, planet_id):
        return is_static_planet(self.planet_by_id[planet_id])

    def comet_life(self, planet_id):
        return comet_remaining_life(planet_id, self.comets)

    def source_inventory_left(self, source_id, spent_total):
        return max(0, int(self.planet_by_id[source_id].ships) - spent_total[source_id])

    def route_hits_target_first(self, src, target, angle, path_target_x, path_target_y):
        started = time.perf_counter() if self.timing_enabled else None
        try:
            start_x, start_y = launch_point(src.x, src.y, src.radius, angle)
            target_hit = ray_circle_hit_distance(
                start_x,
                start_y,
                angle,
                path_target_x,
                path_target_y,
                target.radius,
            )
            if target_hit is None:
                return False

            for other in self.planets:
                if other.id in (src.id, target.id):
                    continue
                hit = ray_circle_hit_distance(
                    start_x,
                    start_y,
                    angle,
                    other.x,
                    other.y,
                    other.radius,
                )
                if (
                    hit is not None
                    and hit + PATH_BLOCKER_EPSILON < target_hit
                    and self.is_opening
                    and (OPENING_ROUTE_GUARD_ENABLED or poor_opening_target(other, self))
                ):
                    return False
            return True
        finally:
            if started is not None:
                timing_add(self, "fn.route_hits_target_first", (time.perf_counter() - started) * 1000.0)

    def plan_shot(self, src_id, target_id, ships):
        started = time.perf_counter() if self.timing_enabled else None
        try:
            ships = int(ships)
            key = (src_id, target_id, ships)
            cached = self.shot_cache.get(key)
            if key in self.shot_cache:
                return cached
            src = self.planet_by_id[src_id]
            target = self.planet_by_id[target_id]
            result = aim_with_prediction(
                src,
                target,
                ships,
                self.initial_by_id,
                self.ang_vel,
                self.comets,
                self.comet_ids,
            )
            if result is not None:
                angle, _, path_target_x, path_target_y = result
                if not self.route_hits_target_first(src, target, angle, path_target_x, path_target_y):
                    result = None
            self.shot_cache[key] = result
            return result
        finally:
            if started is not None:
                timing_add(self, "fn.plan_shot", (time.perf_counter() - started) * 1000.0)

    def probe_ship_candidates(self, src_id, target_id, source_cap, hints=()):
        cache = getattr(self, "probe_candidate_cache", None)
        if cache is None:
            cache = {}
            self.probe_candidate_cache = cache
        source_cap = max(1, int(source_cap))
        normalized_hints = tuple(
            int(math.ceil(hint))
            for hint in hints
            if hint is not None
        )
        cache_key = (src_id, target_id, source_cap, normalized_hints)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        target = self.planet_by_id[target_id]
        target_ships = max(1, int(math.ceil(target.ships)))

        values = set(range(1, min(6, source_cap) + 1))
        values.update(
            {
                source_cap,
                max(1, source_cap // 2),
                max(1, source_cap // 3),
                min(source_cap, PARTIAL_SOURCE_MIN_SHIPS),
                min(source_cap, target_ships + 1),
                min(source_cap, target_ships + 2),
                min(source_cap, target_ships + 4),
                min(source_cap, target_ships + 8),
            }
        )

        for hint in normalized_hints:
            base = max(1, min(source_cap, hint))
            for delta in (-2, -1, 0, 1, 2):
                candidate = base + delta
                if 1 <= candidate <= source_cap:
                    values.add(candidate)

        result = sorted(values)
        cache[cache_key] = result
        return result

    def best_probe_aim(
        self,
        src_id,
        target_id,
        source_cap,
        hints=(),
        min_turn=None,
        max_turn=None,
        anchor_turn=None,
        max_anchor_diff=None,
    ):
        started = time.perf_counter() if self.timing_enabled else None
        try:
            cache_key = (
                src_id,
                target_id,
                max(1, int(source_cap)),
                tuple(hints),
                min_turn,
                max_turn,
                anchor_turn,
                max_anchor_diff,
            )
            cache = getattr(self, "best_probe_cache", None)
            if cache is None:
                cache = {}
                self.best_probe_cache = cache
            if cache_key in cache:
                return cache[cache_key]

            best = None
            best_key = None

            for ships in self.probe_ship_candidates(src_id, target_id, source_cap, hints=hints):
                aim = self.plan_shot(src_id, target_id, ships)
                if aim is None:
                    continue

                angle, turns, dist_to_target, path_target = aim
                if min_turn is not None and turns < min_turn:
                    continue
                if max_turn is not None and turns > max_turn:
                    continue
                if (
                    anchor_turn is not None
                    and max_anchor_diff is not None
                    and abs(turns - anchor_turn) > max_anchor_diff
                ):
                    continue

                if anchor_turn is None:
                    key = (turns, ships)
                else:
                    key = (abs(turns - anchor_turn), turns, ships)

                if best_key is None or key < best_key:
                    best_key = key
                    best = (ships, (angle, turns, dist_to_target, path_target))

            cache[cache_key] = best
            return best
        finally:
            if started is not None:
                timing_add(self, "fn.best_probe_aim", (time.perf_counter() - started) * 1000.0)

    def reaction_times(self, target_id):
        cached = self.reaction_cache.get(target_id)
        if cached is not None:
            return cached

        target = self.planet_by_id[target_id]
        my_t = 10**9
        for planet in self.my_planets:
            seeded = self.best_probe_aim(planet.id, target.id, max(1, int(planet.ships)))
            if seeded is None:
                continue
            _, aim = seeded
            my_t = min(my_t, aim[1])

        enemy_t = 10**9
        for planet in self.enemy_planets:
            seeded = self.best_probe_aim(planet.id, target.id, max(1, int(planet.ships)))
            if seeded is None:
                continue
            _, aim = seeded
            enemy_t = min(enemy_t, aim[1])

        cached = (my_t, enemy_t)
        self.reaction_cache[target_id] = cached
        return cached

    def projected_state(self, target_id, arrival_turn, planned_commitments=None, extra_arrivals=()):
        planned_commitments = planned_commitments or {}
        cutoff = max(1, int(math.ceil(arrival_turn)))
        if not planned_commitments.get(target_id) and not extra_arrivals:
            return state_at_timeline(self.base_timeline[target_id], cutoff)

        arrivals = [
            item
            for item in self.arrivals_by_planet.get(target_id, [])
            if item[0] <= cutoff
        ]
        arrivals.extend(
            item
            for item in planned_commitments.get(target_id, [])
            if item[0] <= cutoff
        )
        arrivals.extend(item for item in extra_arrivals if item[0] <= cutoff)

        target = self.planet_by_id[target_id]
        dyn = simulate_planet_timeline(target, arrivals, self.player, cutoff)
        return state_at_timeline(dyn, cutoff)

    def projected_timeline(self, target_id, horizon, planned_commitments=None, extra_arrivals=()):
        planned_commitments = planned_commitments or {}
        horizon = max(1, int(math.ceil(horizon)))
        arrivals = [
            item for item in self.arrivals_by_planet.get(target_id, []) if item[0] <= horizon
        ]
        arrivals.extend(
            item for item in planned_commitments.get(target_id, []) if item[0] <= horizon
        )
        arrivals.extend(item for item in extra_arrivals if item[0] <= horizon)
        target = self.planet_by_id[target_id]
        return simulate_planet_timeline(target, arrivals, self.player, horizon)

    def hold_status(self, target_id, planned_commitments=None, horizon=HORIZON):
        planned_commitments = planned_commitments or {}
        if planned_commitments.get(target_id):
            tl = self.projected_timeline(
                target_id,
                horizon,
                planned_commitments=planned_commitments,
            )
        else:
            tl = self.base_timeline[target_id]
        return {
            "keep_needed": tl["keep_needed"],
            "min_owned": tl["min_owned"],
            "first_enemy": tl["first_enemy"],
            "fall_turn": tl["fall_turn"],
            "holds_full": tl["holds_full"],
        }

    def _ownership_search_cap(self, eval_turn):
        productive_cap = self.total_production * max(2, eval_turn + 2)
        return max(32, int(self.total_visible_ships + productive_cap + 32))

    def min_ships_to_own_by(
        self,
        target_id,
        eval_turn,
        attacker_owner,
        arrival_turn=None,
        planned_commitments=None,
        extra_arrivals=(),
        upper_bound=None,
    ):
        planned_commitments = planned_commitments or {}
        eval_turn = max(1, int(math.ceil(eval_turn)))
        arrival_turn = eval_turn if arrival_turn is None else max(1, int(math.ceil(arrival_turn)))
        if arrival_turn > eval_turn:
            if upper_bound is not None:
                return max(1, int(upper_bound)) + 1
            return self._ownership_search_cap(eval_turn) + 1

        normalized_extra = tuple(
            (
                max(1, int(math.ceil(turns))),
                owner,
                int(ships),
            )
            for turns, owner, ships in extra_arrivals
            if ships > 0 and max(1, int(math.ceil(turns))) <= eval_turn
        )

        cache_key = None
        if (
            arrival_turn == eval_turn
            and not planned_commitments.get(target_id)
            and not normalized_extra
        ):
            cache_key = (target_id, eval_turn, attacker_owner)
            cached = self.exact_need_cache.get(cache_key)
            if cached is not None:
                return cached

        owner_before, ships_before = self.projected_state(
            target_id,
            eval_turn,
            planned_commitments=planned_commitments,
            extra_arrivals=normalized_extra,
        )
        if owner_before == attacker_owner:
            if cache_key is not None:
                self.exact_need_cache[cache_key] = 0
            return 0

        def owns_at(ships):
            owner_after, _ = self.projected_state(
                target_id,
                eval_turn,
                planned_commitments=planned_commitments,
                extra_arrivals=normalized_extra + ((arrival_turn, attacker_owner, int(ships)),),
            )
            return owner_after == attacker_owner

        if upper_bound is not None:
            hi = max(1, int(upper_bound))
            if not owns_at(hi):
                return hi + 1
        else:
            hi = max(1, int(math.ceil(ships_before)) + 1)
            search_cap = self._ownership_search_cap(eval_turn)
            while hi <= search_cap and not owns_at(hi):
                hi *= 2
            if hi > search_cap:
                hi = search_cap
                if not owns_at(hi):
                    return hi + 1

        lo = 1
        while lo < hi:
            mid = (lo + hi) // 2
            if owns_at(mid):
                hi = mid
            else:
                lo = mid + 1

        if cache_key is not None:
            self.exact_need_cache[cache_key] = lo
        return lo

    def min_ships_to_own_at(
        self,
        target_id,
        arrival_turn,
        attacker_owner,
        planned_commitments=None,
        extra_arrivals=(),
        upper_bound=None,
    ):
        return self.min_ships_to_own_by(
            target_id,
            arrival_turn,
            attacker_owner,
            arrival_turn=arrival_turn,
            planned_commitments=planned_commitments,
            extra_arrivals=extra_arrivals,
            upper_bound=upper_bound,
        )

    def reinforcement_needed_to_hold_until(
        self,
        planet_id,
        arrival_turn,
        hold_until,
        planned_commitments=None,
        upper_bound=None,
    ):
        planned_commitments = planned_commitments or {}
        target = self.planet_by_id[planet_id]
        arrival_turn = max(1, int(math.ceil(arrival_turn)))
        hold_until = max(arrival_turn, int(math.ceil(hold_until)))

        if target.owner != self.player:
            return self.min_ships_to_own_by(
                planet_id,
                hold_until,
                self.player,
                arrival_turn=arrival_turn,
                planned_commitments=planned_commitments,
                upper_bound=upper_bound,
            )

        def holds_with_reinforcement(ships):
            timeline = self.projected_timeline(
                planet_id,
                hold_until,
                planned_commitments=planned_commitments,
                extra_arrivals=((arrival_turn, self.player, int(ships)),),
            )
            for turn in range(arrival_turn, hold_until + 1):
                if timeline["owner_at"].get(turn) != self.player:
                    return False
            return True

        if upper_bound is not None:
            hi = max(1, int(upper_bound))
            if not holds_with_reinforcement(hi):
                return hi + 1
        else:
            hi = 1
            search_cap = self._ownership_search_cap(hold_until)
            while hi <= search_cap and not holds_with_reinforcement(hi):
                hi *= 2
            if hi > search_cap:
                hi = search_cap
                if not holds_with_reinforcement(hi):
                    return hi + 1

        lo = 1
        while lo < hi:
            mid = (lo + hi) // 2
            if holds_with_reinforcement(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def ships_needed_to_capture(
        self,
        target_id,
        arrival_turn,
        planned_commitments=None,
        extra_arrivals=(),
    ):
        return self.min_ships_to_own_at(
            target_id,
            arrival_turn,
            self.player,
            planned_commitments=planned_commitments,
            extra_arrivals=extra_arrivals,
        )

    def ships_needed_to_capture_and_hold(
        self,
        target_id,
        arrival_turn,
        hold_until,
        planned_commitments=None,
        extra_arrivals=(),
        upper_bound=None,
    ):
        planned_commitments = planned_commitments or {}
        extra_arrivals = tuple(extra_arrivals)
        arrival_turn = max(1, int(math.ceil(arrival_turn)))
        hold_until = max(arrival_turn, int(math.ceil(hold_until)))

        base_need = self.min_ships_to_own_at(
            target_id,
            arrival_turn,
            self.player,
            planned_commitments=planned_commitments,
            extra_arrivals=extra_arrivals,
            upper_bound=upper_bound,
        )
        if base_need <= 0:
            return base_need

        relevant_enemy_arrival = any(
            arrival_turn < int(math.ceil(eta)) <= hold_until
            and owner not in (-1, self.player)
            and ships > 0
            for eta, owner, ships in self.arrivals_by_planet.get(target_id, [])
        )
        if (
            not relevant_enemy_arrival
            and not planned_commitments.get(target_id)
            and not extra_arrivals
        ):
            return base_need

        def holds_with(ships):
            timeline = self.projected_timeline(
                target_id,
                hold_until,
                planned_commitments=planned_commitments,
                extra_arrivals=extra_arrivals
                + ((arrival_turn, self.player, int(ships)),),
            )
            for turn in range(arrival_turn, hold_until + 1):
                if timeline["owner_at"].get(turn) != self.player:
                    return False
            return True

        if upper_bound is not None:
            hi = max(1, int(upper_bound))
            if not holds_with(hi):
                return hi + 1
        else:
            hi = max(1, base_need)
            search_cap = self._ownership_search_cap(hold_until)
            while hi <= search_cap and not holds_with(hi):
                hi *= 2
            if hi > search_cap:
                hi = search_cap
                if not holds_with(hi):
                    return hi + 1

        lo = max(1, base_need)
        while lo < hi:
            mid = (lo + hi) // 2
            if holds_with(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

# ============================================================
# Strategy
# ============================================================

def planet_distance(first, second):
    return math.hypot(first.x - second.x, first.y - second.y)


def nearest_sources_to_target(target, sources, top_k):
    if top_k <= 0 or len(sources) <= top_k:
        return sources
    return sorted(
        sources,
        key=lambda src: (planet_distance(src, target), -int(src.ships), src.id),
    )[:top_k]


def min_legal_reaction_time(target, sources, world):
    best = 10**9
    for src in sources:
        seeded = world.best_probe_aim(src.id, target.id, max(1, int(src.ships)))
        if seeded is None:
            continue
        _, aim = seeded
        best = min(best, aim[1])
    return best


def policy_reaction_times(target_id, policy):
    return policy["reaction_time_map"].get(target_id, (10**9, 10**9))


def candidate_time_valid(target, turns, world, remaining_buffer):
    if turns > world.remaining_steps - remaining_buffer:
        return False
    if target.id in world.comet_ids:
        life = world.comet_life(target.id)
        if turns >= life or turns > COMET_MAX_CHASE_TURNS:
            return False
    return True


def initial_owner_of(planet, world):
    if planet.id in PROFILE_HOME_IDS:
        return world.player
    initial = world.initial_by_id.get(planet.id)
    return initial.owner if initial is not None else planet.owner


def is_original_home(planet, world):
    return initial_owner_of(planet, world) == world.player


def is_profile_home(planet):
    return planet.id in PROFILE_HOME_IDS


def captured_age(planet, world):
    captured_at = PROFILE_CAPTURED_AT.get(planet.id)
    if captured_at is None:
        return None
    return max(0, int(world.step) - int(captured_at))


def fresh_neutral_core_source_lock(planet, world):
    if world.step > FRESH_CORE_SOURCE_LOCK_TURN_LIMIT:
        return False
    if planet.production < FRESH_CORE_SOURCE_LOCK_PROD_MIN:
        return False
    if initial_owner_of(planet, world) != -1:
        return False
    age = captured_age(planet, world)
    if age is None or age > FRESH_CORE_SOURCE_LOCK_TURNS:
        return False

    fleet_eta, fleet_stack = enemy_fleet_pressure_to_planet(
        planet,
        world,
        min(PREDICTED_FLEET_THREAT_HORIZON, FRESH_CORE_SOURCE_LOCK_THREAT_HORIZON),
    )
    return (
        fleet_eta is not None
        and fleet_eta <= FRESH_CORE_SOURCE_LOCK_THREAT_TURN
        and fleet_stack >= max(FRESH_CORE_SOURCE_LOCK_STACK_FLOOR, int(planet.production) * 3)
    )


def low_home_rich_core_profile(world):
    if not low_production_home_profile(world):
        return False
    homes = [
        world.initial_by_id.get(home_id) or world.planet_by_id.get(home_id)
        for home_id in PROFILE_HOME_IDS
    ]
    homes = [home for home in homes if home is not None and home.production <= 1]
    for home in homes:
        nearby_cores = 0
        for target in world.initial_by_id.values():
            if target.id == home.id or target.owner != -1:
                continue
            if target.production < CORE_PRODUCTION:
                continue
            if int(target.ships) > LOW_HOME_RICH_CORE_MAX_SHIPS:
                continue
            if planet_distance(home, target) <= LOW_HOME_RICH_CORE_DIST:
                nearby_cores += 1
        if nearby_cores >= LOW_HOME_RICH_CORE_MIN_COUNT:
            return True
    return False


def is_profile_defense_home(planet):
    return is_profile_home(planet) and (
        planet.production <= 1 or planet.production >= CORE_PRODUCTION
    )


def profile_home_mid_anchor(planet, world):
    if not is_profile_home(planet) or planet.production != 2:
        return False
    features = opening_mainline_profile_features(planet, world)
    return features["low_trap"] and features["p2_prize"] and not features["fast_p5"]


def low_production_home_profile(world):
    for home_id in PROFILE_HOME_IDS:
        home = world.initial_by_id.get(home_id) or world.planet_by_id.get(home_id)
        if home is not None and home.production <= 1:
            return True
    return False


def is_core_planet(planet, world):
    return is_original_home(planet, world) or planet.production >= CORE_PRODUCTION


def is_defense_core_planet(planet, world):
    return (
        is_core_planet(planet, world)
        or is_profile_defense_home(planet)
        or (
            planet.production >= DEFENSE_CORE_PRODUCTION
            and low_production_home_profile(world)
        )
    )


def fleet_predicted_hit_turn(fleet, planet, world, horizon):
    dir_x = math.cos(fleet.angle)
    dir_y = math.sin(fleet.angle)
    speed = fleet_speed(fleet.ships)
    prev_x, prev_y = fleet.x, fleet.y

    for turn in range(1, horizon + 1):
        cur_x = fleet.x + dir_x * speed * turn
        cur_y = fleet.y + dir_y * speed * turn
        if cur_x < 0.0 or cur_x > BOARD or cur_y < 0.0 or cur_y > BOARD:
            break

        pos = predict_target_position(
            planet,
            turn,
            world.initial_by_id,
            world.ang_vel,
            world.comets,
            world.comet_ids,
        )
        if pos is not None and point_to_segment_distance(
            pos[0],
            pos[1],
            prev_x,
            prev_y,
            cur_x,
            cur_y,
        ) <= planet.radius:
            return turn
        prev_x, prev_y = cur_x, cur_y

    return None


def enemy_fleet_pressure_to_planet(planet, world, horizon):
    threats = []
    for fleet in world.fleets:
        if fleet.owner == world.player:
            continue
        eta = fleet_predicted_hit_turn(fleet, planet, world, horizon)
        if eta is None:
            continue
        threats.append((eta, int(fleet.ships)))

    if not threats:
        return None, 0

    threats.sort()
    best_eta = threats[0][0]
    stacked = 0
    for eta, _ in threats:
        stacked = max(
            stacked,
            sum(ships for other_eta, ships in threats if abs(other_eta - eta) <= MULTI_ENEMY_STACK_WINDOW),
        )
    return best_eta, stacked


def enemy_occupation_window_open(world):
    threshold = (
        ENEMY_OCCUPATION_TURN_FOUR_PLAYER
        if world.is_four_player
        else ENEMY_OCCUPATION_TURN_TWO_PLAYER
    )
    return world.step >= threshold


def enemy_occupation_prep_window_open(world):
    threshold = (
        ENEMY_OCCUPATION_TURN_FOUR_PLAYER
        if world.is_four_player
        else ENEMY_OCCUPATION_TURN_TWO_PLAYER
    )
    return world.step >= max(0, threshold - ENEMY_OCCUPATION_PREP_TURNS)


def behind_hostile_break_allowed(src, target, send, world, modes):
    if target.owner in (-1, world.player) or world.is_late:
        return False
    if world.step > BEHIND_HOSTILE_BREAK_TURN_LIMIT:
        return False
    if target.production < BEHIND_HOSTILE_BREAK_MIN_PROD:
        return False

    owner_prod = int(world.owner_production.get(target.owner, 0))
    owner_strength = int(world.owner_strength.get(target.owner, 0))
    prod_behind = world.my_prod + BEHIND_HOSTILE_BREAK_OWNER_PROD_GAP < owner_prod
    total_behind = world.my_total < owner_strength * BEHIND_HOSTILE_BREAK_TOTAL_RATIO
    if not (modes.get("is_behind") or prod_behind or total_behind):
        return False

    min_send = BEHIND_HOSTILE_BREAK_MIN_SEND
    if target.production >= CORE_PRODUCTION or is_core_planet(src, world):
        min_send = BEHIND_HOSTILE_BREAK_CORE_MIN_SEND
    return int(send) >= min_send


def midgame_attack_allowed(src, target, send, turns, world, modes):
    if not enemy_occupation_window_open(world) or world.is_late:
        return True
    if target.owner == -1:
        return True
    if modes["is_finishing"]:
        return True
    if initial_owner_of(target, world) == world.player:
        return True
    if behind_hostile_break_allowed(src, target, send, world, modes):
        return True

    min_send = MIDGAME_CORE_SOURCE_MIN_SEND if is_core_planet(src, world) else MIDGAME_HOSTILE_MIN_SEND
    if target.production >= CORE_PRODUCTION:
        min_send = min(min_send, MIDGAME_HOSTILE_MIN_SEND)
    if world.step <= TRANSITION_HOSTILE_TURN_LIMIT and not modes["is_finishing"]:
        transition_min = (
            TRANSITION_CORE_HOSTILE_MIN_SEND
            if is_core_planet(src, world)
            else TRANSITION_HOSTILE_MIN_SEND
        )
        min_send = max(min_send, transition_min)
        if world.my_prod <= world.enemy_prod + TRANSITION_HOSTILE_PROD_LEAD:
            return int(send) >= min_send and target.production >= CORE_PRODUCTION
    return int(send) >= min_send


def opening_target_score(src, target, world, arrival_turns=None):
    distance = planet_distance(src, target)
    if arrival_turns is None:
        arrival_turns = distance / max(1.0, fleet_speed(max(1, int(target.ships) + 1)))

    production_value = float(target.production) ** OPENING_SCORE_PROD_POWER
    if world.is_static(target.id):
        production_value *= OPENING_SCORE_STATIC_MULT

    bridge_value = 0.0
    for other in world.planets:
        if other.id == target.id or other.owner == world.player:
            continue
        if other.production < max(OPENING_SCORE_BRIDGE_MIN_TARGET_PROD, target.production + 1):
            continue
        if planet_distance(target, other) <= OPENING_BRIDGE_RELIEF_RADIUS:
            bridge_value = max(bridge_value, other.production - target.production)

    cost = (
        OPENING_SCORE_BASE_COST
        + int(target.ships) * OPENING_SCORE_SHIP_WEIGHT
        + arrival_turns * OPENING_SCORE_DISTANCE_WEIGHT
    )
    bridge_mult = OPENING_SCORE_BRIDGE_MULT
    if target.production <= 1:
        bridge_mult *= OPENING_SCORE_LOW_PROD_BRIDGE_MULT
    return (production_value + bridge_value * bridge_mult) / max(1.0, cost)


def opening_best_available_score(world, blocked_target=None):
    best = 0.0
    for source in world.my_planets:
        for target in world.neutral_planets:
            if blocked_target is not None and target.id == blocked_target.id:
                continue
            if target.id in world.comet_ids:
                continue
            best = max(best, opening_target_score(source, target, world))
    return best


def opening_fill_unlocked(world):
    return (
        world.is_opening
        and world.step >= OPENING_FILL_STEP
        and (world.my_prod >= OPENING_FILL_MIN_PROD or len(world.my_planets) >= OPENING_FILL_MIN_PLANETS)
    )


def opening_fill_target(target, world):
    if (
        not opening_fill_unlocked(world)
        or target.owner != -1
        or target.id in world.comet_ids
        or int(target.ships) > OPENING_FILL_MAX_TARGET_SHIPS
    ):
        return False
    my_dist = nearest_distance_to_set(target.x, target.y, world.my_planets)
    if my_dist > OPENING_FILL_MAX_DIST:
        return False
    enemy_dist = nearest_distance_to_set(target.x, target.y, world.enemy_planets)
    return enemy_dist >= my_dist - 4.0


def opening_roi_multiplier(target, arrival_turns, world):
    if not world.is_opening or target.owner != -1 or target.id in world.comet_ids:
        return 1.0
    if opening_fill_target(target, world):
        return 1.0

    my_sources = nearest_sources_to_target(target, world.my_planets, 1)
    if not my_sources:
        return 1.0
    score = opening_target_score(my_sources[0], target, world, arrival_turns)
    best = opening_best_available_score(world, blocked_target=target)
    if best <= 0 or score >= best * OPENING_SCORE_LOW_VALUE_RATIO:
        return 1.0
    if target.production <= OPENING_LOW_ROI_MAX_PRODUCTION:
        return OPENING_LOW_ROI_SCORE_MULT
    if (
        target.production <= OPENING_ROI_PRODUCTION_LIMIT
        and int(target.ships) >= OPENING_WEAK_ROI_MIN_SHIPS
    ):
        return OPENING_WEAK_ROI_SCORE_MULT
    return 1.0


def opening_far_neutral_detour(target, arrival_turns, world):
    if (
        not world.is_opening
        or world.step > OPENING_FAR_NEUTRAL_TURN_LIMIT
        or target.owner != -1
        or target.id in world.comet_ids
    ):
        return False

    nearest_sources = nearest_sources_to_target(target, world.my_planets, 1)
    if not nearest_sources:
        return False
    src = nearest_sources[0]
    target_dist = planet_distance(src, target)
    if target_dist <= OPENING_FAR_NEUTRAL_DIST and arrival_turns <= OPENING_FAR_NEUTRAL_TURNS:
        return False

    target_score = opening_target_score(src, target, world, arrival_turns)
    local_best = opening_best_available_score(world, blocked_target=target)
    return local_best > target_score * OPENING_FAR_NEUTRAL_LOCAL_ADVANTAGE


def has_better_opening_target(world, blocked_target):
    for source in world.my_planets:
        for target in world.neutral_planets:
            if target.id == blocked_target.id or target.id in world.comet_ids:
                continue
            if opening_target_score(source, target, world) > opening_target_score(source, blocked_target, world) / max(0.01, OPENING_SCORE_LOW_VALUE_RATIO):
                return True
    return False


def poor_opening_target(target, world):
    if (
        not world.is_opening
        or world.step > OPENING_QUALITY_GATE_TURN
        or target.owner != -1
        or target.id in world.comet_ids
    ):
        return False
    if opening_fill_target(target, world):
        return False
    if (
        target.production <= OPENING_BAD_TARGET_MAX_PROD
        and has_better_opening_target(world, target)
    ):
        return True
    if (
        target.production <= OPENING_ROI_PRODUCTION_LIMIT
        and int(target.ships) >= OPENING_WEAK_ROI_MIN_SHIPS
        and opening_roi_multiplier(target, 1, world) <= OPENING_WEAK_ROI_SCORE_MULT
    ):
        return True
    return opening_roi_multiplier(target, 1, world) <= OPENING_LOW_ROI_SCORE_MULT


def opening_committed_target(src, world):
    if (
        not OPENING_COMMITTED_ENABLED
        or not world.is_opening
        or world.step > OPENING_COMMIT_TURN_LIMIT
        or src.production > OPENING_COMMIT_HOME_PROD_MAX
    ):
        return None

    best = None
    best_key = None
    for target in world.neutral_planets:
        if target.id in world.comet_ids:
            continue
        if target.production < OPENING_COMMIT_MIN_TARGET_PROD:
            continue
        if int(target.ships) > OPENING_COMMIT_MAX_SHIPS:
            continue

        distance = planet_distance(src, target)
        if distance > OPENING_COMMIT_MAX_DIST:
            continue
        score = opening_target_score(src, target, world)
        if score < OPENING_COMMIT_MIN_SCORE:
            continue
        key = (-score, -target.production, int(target.ships), distance, target.id)
        if best_key is None or key < best_key:
            best_key = key
            best = target
    return best


def opening_priority_source_keep(src, world):
    keep = OPENING_PRIORITY_CORE_KEEP if is_core_planet(src, world) else OPENING_PRIORITY_HOME_KEEP
    if is_profile_home(src):
        keep = max(keep, OPENING_PRIORITY_HOME_KEEP)
    return min(int(src.ships), keep)


def opening_priority_profile_active(src, world):
    if not is_profile_home(src) or int(src.production) != 3:
        return False

    has_low_trap = False
    has_quality_prize = False
    for target in world.neutral_planets:
        if target.id in world.comet_ids:
            continue
        distance = planet_distance(src, target)
        if (
            target.production <= 1
            and int(target.ships) >= OPENING_PRIORITY_TRAP_MIN_SHIPS
            and distance <= OPENING_PRIORITY_TRAP_MAX_DIST
        ):
            has_low_trap = True
        if (
            target.production >= OPENING_PRIORITY_PRIZE_MIN_PROD
            and int(target.ships) <= OPENING_PRIORITY_PRIZE_MAX_SHIPS
            and distance <= OPENING_PRIORITY_PRIZE_MAX_DIST
        ):
            has_quality_prize = True
    return has_low_trap and has_quality_prize


def opening_trap_signature_active(world):
    for home_id in PROFILE_HOME_IDS:
        home = world.initial_by_id.get(home_id) or world.planet_by_id.get(home_id)
        if home is None or int(home.production) != 3:
            continue

        has_low_trap = False
        has_quality_prize = False
        for target in world.initial_by_id.values():
            if target.id == home.id:
                continue
            distance = planet_distance(home, target)
            if (
                target.production <= 1
                and int(target.ships) >= OPENING_PRIORITY_TRAP_MIN_SHIPS
                and distance <= OPENING_PRIORITY_TRAP_MAX_DIST
            ):
                has_low_trap = True
            if (
                target.production >= OPENING_PRIORITY_PRIZE_MIN_PROD
                and int(target.ships) <= OPENING_PRIORITY_PRIZE_MAX_SHIPS
                and distance <= OPENING_PRIORITY_PRIZE_MAX_DIST
            ):
                has_quality_prize = True
        if has_low_trap and has_quality_prize:
            return True
    return False


def opening_priority_plan_shot(world, src, target, ships):
    if target_can_move(target, world.initial_by_id, world.comet_ids):
        aim = search_safe_intercept(
            src,
            target,
            ships,
            world.initial_by_id,
            world.ang_vel,
            world.comets,
            world.comet_ids,
        )
        if aim is not None:
            angle, _, path_target_x, path_target_y = aim
            if world.route_hits_target_first(src, target, angle, path_target_x, path_target_y):
                return aim
    return world.plan_shot(src.id, target.id, ships)


def opening_fast_expand_source_keep(src, world):
    if is_profile_home(src):
        if world.step <= OPENING_FAST_EXPAND_TURN_LIMIT // 2:
            return min(int(src.ships), OPENING_FAST_EXPAND_EARLY_HOME_KEEP)
        return min(int(src.ships), OPENING_FAST_EXPAND_HOME_KEEP)
    return min(int(src.ships), OPENING_FAST_EXPAND_SOURCE_KEEP)


def opening_fast_expand_available(src, world):
    return max(0, int(src.ships) - opening_fast_expand_source_keep(src, world))


def opening_fast_expand_base_ready(world):
    return (
        OPENING_FAST_EXPAND_ENABLED
        and world.is_opening
        and world.num_players >= 4
        and world.step <= OPENING_FAST_EXPAND_TURN_LIMIT
        and len(world.my_planets) <= OPENING_FAST_EXPAND_MAX_PLANETS
    )


def opening_fast_expand_close_core_prize(src, world):
    for target in world.initial_by_id.values():
        if target.id == src.id or target.owner != -1 or target.id in world.comet_ids:
            continue
        if int(target.production) < OPENING_FAST_EXPAND_LOW_TARGET_BLOCK_CORE_PROD:
            continue
        if int(target.ships) > OPENING_FAST_EXPAND_LOW_TARGET_BLOCK_CORE_SHIPS:
            continue
        if planet_distance(src, target) <= OPENING_FAST_EXPAND_LOW_TARGET_BLOCK_CORE_DIST:
            return True
    return False


def opening_fast_expand_target_ok(src, target, world):
    if target.owner != -1 or target.id in world.comet_ids:
        return False
    if my_incoming_ships_to(target, world) > 0:
        return False
    if int(target.production) < OPENING_FAST_EXPAND_MIN_TARGET_PROD:
        return False
    if int(target.ships) > OPENING_FAST_EXPAND_MAX_TARGET_SHIPS:
        return False
    if (
        int(target.production) <= 1
        and is_profile_home(src)
        and int(src.production) <= 1
        and opening_fast_expand_close_core_prize(src, world)
    ):
        return False
    if planet_distance(src, target) > OPENING_FAST_EXPAND_MAX_DIST:
        return False
    enemy_dist = nearest_distance_to_set(target.x, target.y, world.enemy_planets)
    my_dist = planet_distance(src, target)
    return enemy_dist + OPENING_FAST_EXPAND_RACE_ALLOWANCE >= my_dist


def opening_fast_expand_target_score(src, target, turns, send, world, source_count=1):
    production_value = float(max(1, target.production)) ** 1.35
    if world.is_static(target.id):
        production_value *= OPENING_SCORE_STATIC_MULT
    if (
        is_profile_home(src)
        and int(src.production) <= 1
        and int(target.production) == 2
        and int(target.ships) <= OPENING_MAINLINE_LOW_HOME_CHEAP_P2_SHIPS
    ):
        production_value *= OPENING_FAST_EXPAND_LOW_HOME_P2_BONUS
    if source_count > 1 and int(target.production) >= CORE_PRODUCTION:
        production_value *= 1.2
    distance = planet_distance(src, target)
    cost = 1.0 + int(send) * 0.75 + float(turns) * 0.85 + distance * 0.12
    return production_value / max(1.0, cost)


def opening_fast_expand_single_moves(world):
    candidates = []
    for src in world.my_planets:
        if opening_mainline_source_inflight(src, world):
            continue
        available = opening_fast_expand_available(src, world)
        if available < OPENING_FAST_EXPAND_MIN_SOURCE:
            continue
        for target in world.neutral_planets:
            if not opening_fast_expand_target_ok(src, target, world):
                continue

            probe = min(available, max(1, int(target.ships) + OPENING_FAST_EXPAND_MARGIN))
            aim = opening_priority_plan_shot(world, src, target, probe)
            if aim is None:
                continue
            _, turns, _, _ = aim
            if turns > OPENING_FAST_EXPAND_MAX_TURNS:
                continue
            if not candidate_time_valid(target, turns, world, LATE_CAPTURE_BUFFER):
                continue

            needed = world.min_ships_to_own_at(
                target.id,
                turns,
                world.player,
                upper_bound=available,
            )
            if needed <= 0 or needed > available:
                continue
            margin = OPENING_FAST_EXPAND_MARGIN
            if int(target.production) >= CORE_PRODUCTION:
                margin += OPENING_FAST_EXPAND_CORE_MARGIN
            send = min(available, max(int(needed), int(target.ships) + margin))
            if send < needed:
                continue
            aim = opening_priority_plan_shot(world, src, target, send)
            if aim is None:
                continue
            angle, turns, _, _ = aim
            if turns > OPENING_FAST_EXPAND_MAX_TURNS:
                continue
            score = opening_fast_expand_target_score(src, target, turns, send, world)
            candidates.append((score, -int(target.production), turns, int(target.ships), src.id, src, target, angle, send))

    if not candidates:
        return None
    candidates.sort()
    _, _, _, _, _, src, target, angle, send = candidates[-1]
    return [[src.id, float(angle), int(send)]]


def opening_fast_expand_swarm_moves(world):
    if (
        not OPENING_FAST_EXPAND_SWARM_ENABLED
        or len(world.my_planets) < OPENING_FAST_EXPAND_SWARM_MIN_PLANETS
    ):
        return None

    candidates = []
    for target in world.neutral_planets:
        if target.owner != -1 or target.id in world.comet_ids:
            continue
        if my_incoming_ships_to(target, world) > 0:
            continue
        if int(target.production) < OPENING_FAST_EXPAND_SWARM_MIN_TARGET_PROD:
            continue
        if int(target.ships) > OPENING_FAST_EXPAND_SWARM_MAX_TARGET_SHIPS:
            continue

        options = []
        for src in nearest_sources_to_target(
            target,
            world.my_planets,
            OPENING_FAST_EXPAND_SWARM_MAX_SOURCES,
        ):
            if opening_mainline_source_inflight(src, world):
                continue
            available = opening_fast_expand_available(src, world)
            if available < OPENING_FAST_EXPAND_MIN_SOURCE:
                continue
            if planet_distance(src, target) > OPENING_FAST_EXPAND_SWARM_MAX_DIST:
                continue
            probe = min(available, max(1, int(target.ships) + 1))
            aim = opening_priority_plan_shot(world, src, target, probe)
            if aim is None:
                continue
            angle, turns, _, _ = aim
            if turns > OPENING_FAST_EXPAND_SWARM_MAX_TURNS:
                continue
            if not candidate_time_valid(target, turns, world, LATE_CAPTURE_BUFFER):
                continue
            options.append((turns, -available, src, available, angle))

        if len(options) < 2:
            continue
        options.sort()
        best = None
        for size in range(2, min(len(options), OPENING_FAST_EXPAND_SWARM_MAX_SOURCES) + 1):
            group = options[:size]
            turns = [item[0] for item in group]
            if max(turns) - min(turns) > OPENING_FAST_EXPAND_SWARM_ETA_TOLERANCE:
                continue
            joint_turn = max(turns)
            total_available = sum(item[3] for item in group)
            need = world.min_ships_to_own_at(
                target.id,
                joint_turn,
                world.player,
                upper_bound=total_available,
            )
            if need <= 0 or need > total_available:
                continue
            desired = min(
                total_available,
                max(
                    int(need),
                    int(target.ships) + OPENING_FAST_EXPAND_SWARM_OVERKILL,
                ),
            )
            if desired < need:
                continue
            score = (
                float(max(1, target.production)) ** 1.45
                / max(1.0, desired + joint_turn * 1.1)
            )
            if int(target.production) >= CORE_PRODUCTION:
                score *= 1.2
            best = (score, joint_turn, desired, group)
            break
        if best is not None:
            score, joint_turn, desired, group = best
            my_dist = min(planet_distance(item[2], target) for item in group)
            enemy_dist = nearest_distance_to_set(target.x, target.y, world.enemy_planets)
            if enemy_dist + OPENING_FAST_EXPAND_RACE_ALLOWANCE < my_dist:
                score *= 0.55
            candidates.append((score, -int(target.production), joint_turn, int(target.ships), target.id, target, desired, group))

    if not candidates:
        return None
    candidates.sort()
    _, _, _, _, _, target, desired, group = candidates[-1]
    remaining = int(desired)
    moves = []
    ordered = sorted(group, key=lambda item: (item[0], item[2].id))
    for idx, (_, _, src, available, _) in enumerate(ordered):
        rest = sum(item[3] for item in ordered[idx + 1 :])
        send = min(int(available), max(0, remaining - rest))
        if send <= 0:
            continue
        aim = opening_priority_plan_shot(world, src, target, send)
        if aim is None:
            return None
        angle, turns, _, _ = aim
        if turns > OPENING_FAST_EXPAND_SWARM_MAX_TURNS:
            return None
        moves.append([src.id, float(angle), int(send)])
        remaining -= int(send)

    if remaining > 0 or not moves:
        return None
    return moves


def build_opening_fast_expand_moves(world):
    if not opening_fast_expand_base_ready(world):
        return None
    swarm_moves = opening_fast_expand_swarm_moves(world)
    if swarm_moves is not None:
        return swarm_moves
    return opening_fast_expand_single_moves(world)


def opening_direct_source_keep(src, world):
    if world.num_players <= 2 and world.step < ENEMY_OCCUPATION_TURN_TWO_PLAYER:
        return 0
    if world.num_players >= 4 and world.step < ENEMY_OCCUPATION_TURN_FOUR_PLAYER:
        return min(int(src.ships), 1)
    return min(int(src.ships), 2 if world.num_players >= 4 else 1)


def opening_direct_target_allowed(src, target, world):
    if target.owner != -1 or target.id in world.comet_ids:
        return False
    if my_incoming_ships_to(target, world) > 0:
        return False
    if int(target.production) < OPENING_DIRECT_MIN_TARGET_PROD:
        return False
    if int(target.ships) > OPENING_DIRECT_MAX_TARGET_SHIPS:
        return False
    my_dist = planet_distance(src, target)
    max_dist = OPENING_DIRECT_MAX_DIST
    if world.num_players <= 2 and target_can_move(target, world.initial_by_id, world.comet_ids):
        max_dist = OPENING_DIRECT_MOVING_MAX_DIST_TWO_PLAYER
    if my_dist > max_dist:
        return False

    enemy_dist = nearest_distance_to_set(target.x, target.y, world.enemy_planets)
    allowance = (
        OPENING_DIRECT_RACE_ALLOWANCE_FOUR_PLAYER
        if world.num_players >= 4
        else OPENING_DIRECT_RACE_ALLOWANCE_TWO_PLAYER
    )
    return enemy_dist + allowance >= my_dist


def opening_direct_target_score(src, target, desired, turns, wait, world):
    production_value = float(max(1, target.production)) ** 1.55
    if world.is_static(target.id):
        production_value *= OPENING_SCORE_STATIC_MULT
    if int(src.production) <= 1 and int(target.production) >= CORE_PRODUCTION:
        production_value *= 1.18
    if int(src.production) <= 2 and int(target.production) >= int(src.production) + 2:
        production_value *= 1.12

    enemy_dist = nearest_distance_to_set(target.x, target.y, world.enemy_planets)
    my_dist = planet_distance(src, target)
    race_mult = 1.0
    if enemy_dist + 2.0 < my_dist:
        race_mult = 0.58
    elif enemy_dist <= my_dist + 3.0:
        race_mult = 0.82

    cost = 1.0 + desired * 0.85 + turns * 1.05 + wait * 2.2 + my_dist * 0.12
    return production_value * race_mult / max(1.0, cost)


def opening_direct_candidates(src, world):
    keep = opening_direct_source_keep(src, world)
    available_now = max(0, int(src.ships) - keep)
    max_wait = (
        OPENING_DIRECT_FOUR_PLAYER_MAX_WAIT
        if world.num_players >= 4
        else OPENING_DIRECT_TWO_PLAYER_MAX_WAIT
    )
    max_turns = (
        OPENING_DIRECT_FOUR_PLAYER_MAX_TURNS
        if world.num_players >= 4
        else OPENING_DIRECT_TWO_PLAYER_MAX_TURNS
    )
    future_cap = available_now + int(src.production) * max_wait
    candidates = []

    if future_cap <= 0:
        return candidates

    for target in world.neutral_planets:
        if not opening_direct_target_allowed(src, target, world):
            continue

        margin = OPENING_DIRECT_MARGIN
        if int(target.production) >= CORE_PRODUCTION:
            margin += OPENING_DIRECT_CORE_MARGIN
        probe = min(max(1, future_cap), max(1, int(target.ships) + margin))
        aim = opening_priority_plan_shot(world, src, target, probe)
        if aim is None:
            continue
        _, turns, _, _ = aim
        if turns > max_turns:
            continue
        if not candidate_time_valid(target, turns, world, LATE_CAPTURE_BUFFER):
            continue

        needed = world.min_ships_to_own_at(
            target.id,
            turns,
            world.player,
            upper_bound=max(1, future_cap),
        )
        desired = max(int(needed), int(target.ships) + margin)
        if needed <= 0 or desired > future_cap:
            continue

        if desired <= available_now:
            wait = 0
        elif int(src.production) > 0:
            wait = int(math.ceil((desired - available_now) / max(1, int(src.production))))
        else:
            continue
        if wait > max_wait:
            continue

        score = opening_direct_target_score(src, target, desired, turns, wait, world)
        candidates.append(
            {
                "score": score,
                "target": target,
                "desired": desired,
                "turns": turns,
                "wait": wait,
            }
        )

    candidates.sort(
        key=lambda item: (
            -item["score"],
            item["wait"],
            -int(item["target"].production),
            int(item["target"].ships),
            item["turns"],
            item["target"].id,
        )
    )
    return candidates


def build_opening_direct_expand_moves(world):
    if (
        not OPENING_DIRECT_ENABLED
        or not world.is_opening
        or world.step > OPENING_DIRECT_TURN_LIMIT
        or len(world.my_planets) > OPENING_DIRECT_MAX_PLANETS
    ):
        return None

    sources = sorted(
        [
            planet
            for planet in world.my_planets
            if is_profile_home(planet)
            and is_static_planet(planet)
            and int(planet.production) <= 2
        ],
        key=lambda planet: (-int(planet.production), planet.id),
    )
    if not sources:
        return None

    for src in sources:
        if opening_mainline_source_inflight(src, world):
            return None

        candidates = opening_direct_candidates(src, world)
        if not candidates:
            continue

        best = candidates[0]
        affordable = [item for item in candidates if item["wait"] == 0]
        chosen = None
        if affordable:
            top_affordable = affordable[0]
            if top_affordable["score"] >= best["score"] * OPENING_DIRECT_AFFORDABLE_RATIO:
                chosen = top_affordable
            elif best["wait"] > 0:
                return []
            else:
                chosen = best
        elif best["wait"] > 0:
            return []

        if chosen is None:
            continue

        keep = opening_direct_source_keep(src, world)
        send = min(max(0, int(src.ships) - keep), int(chosen["desired"]))
        if send < int(chosen["desired"]):
            return []
        aim = opening_priority_plan_shot(world, src, chosen["target"], send)
        if aim is None:
            continue
        angle, _, _, _ = aim
        return [[src.id, float(angle), int(send)]]

    return None


def opening_anchor_has_fast_alternative(src, anchor, world):
    for target in world.neutral_planets:
        if target.id in (anchor.id, src.id) or target.id in world.comet_ids:
            continue
        if target.production < OPENING_ANCHOR_FAST_ALT_PROD:
            continue
        if int(target.ships) > OPENING_ANCHOR_FAST_ALT_MAX_SHIPS:
            continue
        if planet_distance(src, target) <= OPENING_ANCHOR_FAST_ALT_DIST:
            return True
    return False


def opening_anchor_candidates(src, world):
    if (
        not OPENING_ANCHOR_ENABLED
        or not world.is_opening
        or world.step > OPENING_ANCHOR_TURN_LIMIT
        or len(world.my_planets) > OPENING_ANCHOR_MAX_PLANETS
        or not is_profile_home(src)
        or src.production < OPENING_ANCHOR_MIN_HOME_PROD
    ):
        return []

    keep = min(int(src.ships), OPENING_ANCHOR_HOME_KEEP)
    available_now = max(0, int(src.ships) - keep)
    future_cap = available_now + int(src.production) * OPENING_ANCHOR_MAX_WAIT
    candidates = []

    for target in world.neutral_planets:
        if target.id in world.comet_ids:
            continue
        if target.production < OPENING_ANCHOR_MIN_PROD:
            continue
        if not (OPENING_ANCHOR_MIN_SHIPS <= int(target.ships) <= OPENING_ANCHOR_MAX_SHIPS):
            continue
        distance = planet_distance(src, target)
        if distance > OPENING_ANCHOR_MAX_DIST:
            continue
        if opening_anchor_has_fast_alternative(src, target, world):
            continue

        probe_ships = min(
            max(1, future_cap),
            int(target.ships) + OPENING_ANCHOR_MARGIN,
        )
        aim = opening_priority_plan_shot(world, src, target, probe_ships)
        if aim is None:
            continue
        _, turns, _, _ = aim
        if not candidate_time_valid(target, turns, world, LATE_CAPTURE_BUFFER):
            continue
        needed = world.min_ships_to_own_at(
            target.id,
            turns,
            world.player,
            upper_bound=max(1, future_cap),
        )
        desired = max(needed, int(target.ships) + OPENING_ANCHOR_MARGIN)
        if needed <= 0 or desired > future_cap:
            continue
        wait = 0
        if desired > available_now:
            wait = int(math.ceil((desired - available_now) / max(1, int(src.production))))
        if wait > OPENING_ANCHOR_MAX_WAIT:
            continue
        score = target.production / max(1.0, int(target.ships) + distance * 0.55 + wait * 2.0)
        candidates.append((score, wait, desired, target, turns))

    candidates.sort(
        key=lambda item: (
            -item[0],
            item[1],
            -int(item[3].production),
            int(item[3].ships),
            item[4],
            item[3].id,
        )
    )
    return candidates


def build_opening_anchor_moves(world):
    for src in sorted(world.my_planets, key=lambda planet: (not is_profile_home(planet), planet.id)):
        candidates = opening_anchor_candidates(src, world)
        if not candidates:
            continue

        _, wait, desired, target, _ = candidates[0]
        if wait > 0:
            return []

        send = min(max(0, int(src.ships) - OPENING_ANCHOR_HOME_KEEP), desired)
        if send < desired:
            return []
        aim = opening_priority_plan_shot(world, src, target, send)
        if aim is None:
            continue
        angle, _, _, _ = aim
        return [[src.id, float(angle), int(send)]]

    return None


def opening_mainline_source_keep(src, world):
    keep = OPENING_MAINLINE_CORE_KEEP if is_core_planet(src, world) else OPENING_MAINLINE_HOME_KEEP
    if is_profile_home(src):
        keep = max(keep, OPENING_MAINLINE_HOME_KEEP)
    return min(int(src.ships), keep)


def opening_mainline_score(src, target, needed, turns, wait_turns, world):
    production_value = float(target.production) ** OPENING_MAINLINE_PROD_POWER
    if world.is_static(target.id):
        production_value *= OPENING_SCORE_STATIC_MULT
    if target.production <= 1:
        production_value *= OPENING_MAINLINE_LOW_PROD_MULT

    enemy_dist = nearest_distance_to_set(target.x, target.y, world.enemy_planets)
    my_dist = planet_distance(src, target)
    race_mult = 1.0
    if enemy_dist + 5.0 < my_dist:
        race_mult = 0.55
    elif enemy_dist <= my_dist + 3.0:
        race_mult = 0.82

    cost = (
        OPENING_MAINLINE_BASE_COST
        + int(needed) * OPENING_MAINLINE_SHIP_WEIGHT
        + int(turns) * OPENING_MAINLINE_TURN_WEIGHT
        + int(wait_turns) * OPENING_MAINLINE_WAIT_WEIGHT
    )
    return production_value * race_mult / max(1.0, cost)


def opening_mainline_source_inflight(src, world):
    return any(
        fleet.owner == world.player and fleet.from_planet_id == src.id
        for fleet in world.fleets
    )


def opening_local_quality_source_keep(src, world):
    keep = (
        OPENING_LOCAL_QUALITY_CORE_KEEP
        if is_core_planet(src, world)
        else OPENING_LOCAL_QUALITY_HOME_KEEP
    )
    if is_profile_home(src):
        keep = max(keep, OPENING_LOCAL_QUALITY_HOME_KEEP)
    return min(int(src.ships), keep)


def opening_local_quality_profile_active(src, world):
    if not is_profile_home(src):
        return False
    if src.production <= 1:
        cheap_p5 = any(
            target.owner == -1
            and target.id not in world.comet_ids
            and target.production >= CORE_PRODUCTION + 1
            and int(target.ships) <= OPENING_LOCAL_LOW_HOME_CHEAP_P5_SHIPS
            and planet_distance(src, target) <= OPENING_LOCAL_LOW_HOME_CHEAP_P5_DIST
            for target in world.initial_by_id.values()
        )
        if cheap_p5:
            return False
        return any(
            target.owner == -1
            and target.id not in world.comet_ids
            and (
                (
                    target.production >= CORE_PRODUCTION
                    and int(target.ships) <= OPENING_LOCAL_LOW_HOME_CORE_SHIPS
                    and planet_distance(src, target) <= OPENING_LOCAL_LOW_HOME_CORE_DIST
                )
                or (
                    target.production == 3
                    and int(target.ships) <= OPENING_LOCAL_LOW_HOME_P3_SHIPS
                    and planet_distance(src, target) <= OPENING_LOCAL_LOW_HOME_P3_DIST
                )
            )
            for target in world.initial_by_id.values()
        )
    if src.production == 2:
        return any(
            target.owner == -1
            and target.id not in world.comet_ids
            and target.production >= OPENING_LOCAL_P2_CORE_PROD
            and int(target.ships) <= OPENING_LOCAL_P2_CORE_SHIPS
            and planet_distance(src, target) <= OPENING_LOCAL_P2_CORE_DIST
            for target in world.initial_by_id.values()
        )
    if src.production == CORE_PRODUCTION:
        return any(
            target.owner == -1
            and target.id not in world.comet_ids
            and target.production >= CORE_PRODUCTION
            and int(target.ships) <= OPENING_LOCAL_HIGH_HOME_CORE_SHIPS
            and planet_distance(src, target) <= OPENING_LOCAL_HIGH_HOME_CORE_DIST
            for target in world.initial_by_id.values()
        )
    if src.production == 3:
        close_heavy_core = any(
            target.owner == -1
            and target.id not in world.comet_ids
            and target.production >= CORE_PRODUCTION
            and int(target.ships) <= OPENING_LOCAL_MID_HOME_CLOSE_CORE_SHIPS
            and planet_distance(src, target) <= OPENING_LOCAL_MID_HOME_CLOSE_CORE_DIST
            for target in world.initial_by_id.values()
        )
        if close_heavy_core:
            return True
        close_core = any(
            target.owner == -1
            and target.id not in world.comet_ids
            and target.production >= CORE_PRODUCTION
            and int(target.ships) <= OPENING_LOCAL_MID_HOME_CORE_BLOCK_SHIPS
            and planet_distance(src, target) <= OPENING_LOCAL_MID_HOME_CORE_BLOCK_DIST
            for target in world.initial_by_id.values()
        )
        if close_core:
            return False
        return any(
            target.owner == -1
            and target.id not in world.comet_ids
            and target.production == 3
            and int(target.ships) <= OPENING_LOCAL_MID_HOME_P3_SHIPS
            and planet_distance(src, target) <= OPENING_LOCAL_MID_HOME_P3_DIST
            for target in world.initial_by_id.values()
        )
    return False


def opening_local_quality_target_allowed(src, target, world):
    distance = planet_distance(src, target)
    if src.production <= 1:
        return (
            target.production >= CORE_PRODUCTION
            and int(target.ships) <= OPENING_LOCAL_LOW_HOME_CORE_SHIPS
            and distance <= OPENING_LOCAL_LOW_HOME_CORE_DIST
        ) or (
            target.production == 3
            and int(target.ships) <= OPENING_LOCAL_LOW_HOME_P3_SHIPS
            and distance <= OPENING_LOCAL_LOW_HOME_P3_DIST
        )
    if src.production == 2:
        return (
            target.production >= OPENING_LOCAL_P2_CORE_PROD
            and int(target.ships) <= OPENING_LOCAL_P2_CORE_SHIPS
            and distance <= OPENING_LOCAL_P2_CORE_DIST
        )
    if src.production == CORE_PRODUCTION:
        return (
            target.production >= CORE_PRODUCTION
            and int(target.ships) <= OPENING_LOCAL_HIGH_HOME_CORE_SHIPS
            and distance <= OPENING_LOCAL_HIGH_HOME_CORE_DIST
        )
    if src.production == 3:
        return (
            target.production >= CORE_PRODUCTION
            and int(target.ships) <= OPENING_LOCAL_MID_HOME_CLOSE_CORE_SHIPS
            and distance <= OPENING_LOCAL_MID_HOME_CLOSE_CORE_DIST
        ) or (
            target.production == 3
            and int(target.ships) <= OPENING_LOCAL_MID_HOME_P3_SHIPS
            and distance <= OPENING_LOCAL_MID_HOME_P3_DIST
        )
    return False


def build_opening_local_quality_moves(world):
    if (
        not OPENING_LOCAL_QUALITY_ENABLED
        or not world.is_opening
        or world.step > OPENING_LOCAL_QUALITY_TURN_LIMIT
    ):
        return None

    sources = [
        planet
        for planet in world.my_planets
        if opening_local_quality_profile_active(planet, world)
        and len(world.my_planets) <= OPENING_LOCAL_QUALITY_MAX_PLANETS
    ]
    if not sources:
        return None

    for src in sorted(sources, key=lambda planet: (-int(planet.production), planet.id)):
        if opening_mainline_source_inflight(src, world):
            return []
        keep = opening_local_quality_source_keep(src, world)
        available_now = max(0, int(src.ships) - keep)
        future_cap = available_now + int(src.production) * OPENING_LOCAL_QUALITY_MAX_WAIT
        if future_cap <= 0:
            continue

        candidates = []
        for target in world.neutral_planets:
            if target.id in world.comet_ids:
                continue
            if my_incoming_ships_to(target, world) > 0:
                continue
            if not opening_local_quality_target_allowed(src, target, world):
                continue
            route_ships = max(1, min(100, int(target.ships) + OPENING_LOCAL_QUALITY_MARGIN))
            aim = opening_priority_plan_shot(world, src, target, route_ships)
            if aim is None:
                continue
            _, turns, _, _ = aim
            if not candidate_time_valid(target, turns, world, LATE_CAPTURE_BUFFER):
                continue
            need_cap = max(future_cap, int(target.ships) + OPENING_LOCAL_QUALITY_MARGIN, 1)
            needed = world.min_ships_to_own_at(
                target.id,
                turns,
                world.player,
                upper_bound=need_cap,
            )
            if needed <= 0 or needed > need_cap:
                continue
            margin = OPENING_LOCAL_QUALITY_MARGIN
            if target.production >= CORE_PRODUCTION:
                margin += OPENING_LOCAL_QUALITY_CORE_MARGIN
            desired = max(needed, int(target.ships) + margin)
            if desired > future_cap:
                continue
            if desired <= available_now:
                wait_turns = 0
            else:
                wait_turns = int(math.ceil((desired - available_now) / max(1, int(src.production))))
            if wait_turns > OPENING_LOCAL_QUALITY_MAX_WAIT:
                continue

            score = opening_mainline_score(src, target, desired, turns, wait_turns, world)
            candidates.append(
                {
                    "score": score,
                    "target": target,
                    "needed": desired,
                    "turns": turns,
                    "wait": wait_turns,
                }
            )

        if not candidates:
            continue
        candidates.sort(
            key=lambda item: (
                -item["score"],
                item["wait"],
                -int(item["target"].production),
                int(item["target"].ships),
                item["turns"],
                item["target"].id,
            )
        )
        chosen = candidates[0]
        if chosen["needed"] > available_now:
            return []
        send = int(chosen["needed"])
        aim = opening_priority_plan_shot(world, src, chosen["target"], send)
        if aim is None:
            continue
        angle, _, _, _ = aim
        return [[src.id, float(angle), send]]

    return None


def opening_mainline_profile_features(src, world):
    features = {
        "cheap_p2": 0,
        "two_step_p2": 0,
        "close_core": False,
        "low_heavy_core": False,
        "low_trap": False,
        "p2_prize": False,
        "fast_p5": False,
        "fallback_low": False,
        "near_quality": False,
        "p3_core_prize": False,
    }
    for target in world.initial_by_id.values():
        if target.id == src.id or target.owner != -1 or target.id in world.comet_ids:
            continue
        distance = planet_distance(src, target)
        if (
            target.production == 2
            and int(target.ships) <= OPENING_MAINLINE_LOW_HOME_CHEAP_P2_SHIPS
            and distance <= OPENING_MAINLINE_LOW_HOME_CHEAP_P2_DIST
        ):
            features["cheap_p2"] += 1
        if (
            target.production == 2
            and int(target.ships) <= OPENING_MAINLINE_LOW_HOME_TWO_STEP_SHIPS
            and distance <= OPENING_MAINLINE_LOW_HOME_TWO_STEP_DIST
        ):
            features["two_step_p2"] += 1
        if (
            target.production >= CORE_PRODUCTION
            and int(target.ships) <= OPENING_MAINLINE_CLOSE_CORE_SHIPS
            and distance <= OPENING_MAINLINE_CLOSE_CORE_DIST
        ):
            features["close_core"] = True
        if (
            target.production >= OPENING_MAINLINE_LOW_HOME_HEAVY_CORE_PROD
            and int(target.ships) <= OPENING_MAINLINE_LOW_HOME_HEAVY_CORE_SHIPS
            and distance <= OPENING_MAINLINE_LOW_HOME_HEAVY_CORE_DIST
        ):
            features["low_heavy_core"] = True
        if (
            target.production <= 1
            and int(target.ships) >= OPENING_MAINLINE_P2_TRAP_SHIPS
            and distance <= OPENING_MAINLINE_P2_TRAP_DIST
        ):
            features["low_trap"] = True
        if (
            target.production == CORE_PRODUCTION
            and int(target.ships) <= OPENING_MAINLINE_P2_PRIZE_SHIPS
            and distance <= OPENING_MAINLINE_P2_PRIZE_DIST
        ):
            features["p2_prize"] = True
        if (
            target.production >= CORE_PRODUCTION + 1
            and int(target.ships) <= OPENING_MAINLINE_P2_FAST_P5_SHIPS
            and distance <= OPENING_MAINLINE_P2_FAST_P5_DIST
        ):
            features["fast_p5"] = True
        if (
            target.production <= 1
            and int(target.ships) <= OPENING_MAINLINE_P2_FALLBACK_LOW_SHIPS
            and distance <= OPENING_MAINLINE_P2_FALLBACK_LOW_DIST
        ):
            features["fallback_low"] = True
        if (
            target.production >= 2
            and int(target.ships) <= OPENING_MAINLINE_P2_QUALITY_SHIPS
            and distance <= OPENING_MAINLINE_P2_QUALITY_DIST
        ):
            features["near_quality"] = True
        if (
            target.production >= CORE_PRODUCTION
            and int(target.ships) <= OPENING_MAINLINE_P3_CORE_SHIPS
            and OPENING_MAINLINE_P3_CORE_MIN_DIST <= distance <= OPENING_MAINLINE_P3_CORE_DIST
        ):
            features["p3_core_prize"] = True
    return features


def opening_mainline_profile_active(src, world):
    if not is_profile_home(src):
        return False

    features = opening_mainline_profile_features(src, world)
    if src.production <= 1:
        if features["low_heavy_core"]:
            return True
        if features["close_core"]:
            return False
        return features["cheap_p2"] > 0
    if src.production == 2:
        if features["fast_p5"]:
            return False
        return (features["low_trap"] and features["p2_prize"]) or (
            features["fallback_low"] and not features["near_quality"]
        )
    if src.production == 3:
        return features["p3_core_prize"]
    return False


def opening_mainline_max_wait(src, world):
    features = opening_mainline_profile_features(src, world)
    if src.production <= 1 and features["low_heavy_core"]:
        return max(OPENING_MAINLINE_MAX_WAIT, OPENING_MAINLINE_LOW_HOME_HEAVY_MAX_WAIT)
    return OPENING_MAINLINE_MAX_WAIT


def opening_mainline_planet_limit(src, world):
    features = opening_mainline_profile_features(src, world)
    if src.production <= 1 and features["two_step_p2"] >= 2:
        return 2
    return OPENING_MAINLINE_MAX_PLANETS


def opening_mainline_low_fallback_active(src, world):
    if src.production != 2:
        return False
    features = opening_mainline_profile_features(src, world)
    return features["fallback_low"] and not features["near_quality"]


def opening_mainline_target_candidates(src, world):
    if (
        not OPENING_MAINLINE_ENABLED
        or not world.is_opening
        or world.step > OPENING_MAINLINE_TURN_LIMIT
        or len(world.my_planets) > opening_mainline_planet_limit(src, world)
        or not opening_mainline_profile_active(src, world)
    ):
        return []

    keep = opening_mainline_source_keep(src, world)
    available_now = max(0, int(src.ships) - keep)
    production = max(0, int(src.production))
    max_wait = opening_mainline_max_wait(src, world)
    future_cap = available_now + production * max_wait
    if future_cap <= 0:
        return []

    candidates = []
    for target in world.neutral_planets:
        if target.id in world.comet_ids:
            continue
        if my_incoming_ships_to(target, world) > 0:
            continue
        min_target_prod = 1 if opening_mainline_low_fallback_active(src, world) else OPENING_MAINLINE_MIN_TARGET_PROD
        if int(target.production) < min_target_prod:
            continue
        if opening_fill_target(target, world):
            continue

        route_ships = max(1, min(100, int(target.ships) + OPENING_MAINLINE_MARGIN))
        aim = opening_priority_plan_shot(world, src, target, route_ships)
        if aim is None:
            continue
        _, turns, _, _ = aim
        if not candidate_time_valid(target, turns, world, LATE_CAPTURE_BUFFER):
            continue

        need_cap = max(future_cap, int(target.ships) + OPENING_MAINLINE_MARGIN, 1)
        needed = world.min_ships_to_own_at(
            target.id,
            turns,
            world.player,
            upper_bound=need_cap,
        )
        if needed <= 0 or needed > need_cap:
            continue

        target_margin = OPENING_MAINLINE_MARGIN
        if target.production >= CORE_PRODUCTION:
            target_margin += OPENING_MAINLINE_CORE_EXTRA_MARGIN
        desired = max(needed, int(target.ships) + target_margin)
        if desired > future_cap:
            continue
        if desired <= available_now:
            wait_turns = 0
        elif production > 0:
            wait_turns = int(math.ceil((desired - available_now) / max(1, production)))
        else:
            continue
        if wait_turns > max_wait:
            continue

        score = opening_mainline_score(src, target, desired, turns, wait_turns, world)
        score /= 1.0 + wait_turns * OPENING_MAINLINE_WAIT_PENALTY
        min_score = (
            OPENING_MAINLINE_FALLBACK_LOW_MIN_SCORE
            if opening_mainline_low_fallback_active(src, world)
            else OPENING_MAINLINE_MIN_SCORE
        )
        if score < min_score:
            continue

        candidates.append(
            {
                "score": score,
                "target": target,
                "turns": turns,
                "needed": desired,
                "wait": wait_turns,
                "available_now": available_now,
            }
        )

    candidates.sort(
        key=lambda item: (
            -item["score"],
            item["wait"],
            -int(item["target"].production),
            int(item["target"].ships),
            item["turns"],
            item["target"].id,
        )
    )
    return candidates


def build_opening_mainline_moves(world):
    if (
        not OPENING_MAINLINE_ENABLED
        or not world.is_opening
        or world.step > OPENING_MAINLINE_TURN_LIMIT
    ):
        return None

    source_order = sorted(
        [
            planet
            for planet in world.my_planets
            if opening_mainline_profile_active(planet, world)
            and len(world.my_planets) <= opening_mainline_planet_limit(planet, world)
        ],
        key=lambda planet: (-int(planet.production), -int(planet.ships), planet.id),
    )
    if not source_order:
        return None

    for src in source_order:
        if opening_mainline_source_inflight(src, world):
            return []
        keep = opening_mainline_source_keep(src, world)
        available_now = max(0, int(src.ships) - keep)
        candidates = opening_mainline_target_candidates(src, world)
        if not candidates:
            continue

        best = candidates[0]
        max_wait = opening_mainline_max_wait(src, world)
        affordable = [item for item in candidates if item["needed"] <= available_now]
        chosen = None
        if affordable:
            top_affordable = affordable[0]
            if top_affordable["score"] >= best["score"] * OPENING_MAINLINE_ALT_AFFORDABLE_RATIO:
                chosen = top_affordable
            elif (
                world.step <= OPENING_MAINLINE_WAIT_TURN_LIMIT
                and best["wait"] <= max_wait
            ):
                return []
            else:
                chosen = top_affordable
        elif (
            world.step <= OPENING_MAINLINE_WAIT_TURN_LIMIT
            and best["wait"] <= max_wait
        ):
            return []

        if chosen is None:
            continue

        target = chosen["target"]
        send = min(available_now, int(chosen["needed"]))
        if send < int(chosen["needed"]):
            continue
        aim = opening_priority_plan_shot(world, src, target, send)
        if aim is None:
            continue
        angle, _, _, _ = aim
        return [[src.id, float(angle), int(send)]]

    return None


def opening_priority_target_candidates(src, world):
    trap_profile = opening_priority_profile_active(src, world)
    keep = opening_priority_source_keep(src, world)
    available_now = max(0, int(src.ships) - keep)
    production = max(0, int(src.production))
    future_cap = max(available_now, available_now + production * OPENING_PRIORITY_MAX_WAIT)
    candidates = []

    for target in world.neutral_planets:
        if target.id in world.comet_ids:
            continue
        if int(target.production) < OPENING_PRIORITY_MIN_TARGET_PROD:
            continue
        if trap_profile and int(target.production) < OPENING_PRIORITY_PRIZE_MIN_PROD:
            continue
        if opening_fill_target(target, world):
            continue

        route_ships = max(1, min(80, int(target.ships) + 1))
        aim = opening_priority_plan_shot(world, src, target, route_ships)
        if aim is None:
            continue
        _, turns, _, _ = aim
        if not candidate_time_valid(target, turns, world, LATE_CAPTURE_BUFFER):
            continue

        need_cap = max(future_cap, int(target.ships) + 1, 1)
        needed = world.min_ships_to_own_at(
            target.id,
            turns,
            world.player,
            upper_bound=need_cap,
        )
        if needed <= 0 or needed > need_cap:
            continue

        if needed <= available_now:
            wait_turns = 0
        elif production > 0:
            wait_turns = int(math.ceil((needed - available_now) / max(1, production)))
        else:
            continue
        if wait_turns > OPENING_PRIORITY_MAX_WAIT:
            continue

        score = opening_target_score(src, target, world, turns + wait_turns)
        score /= 1.0 + wait_turns * OPENING_PRIORITY_WAIT_PENALTY
        if score < OPENING_PRIORITY_MIN_SCORE:
            continue

        candidates.append(
            {
                "score": score,
                "target": target,
                "turns": turns,
                "needed": needed,
                "wait": wait_turns,
                "available_now": available_now,
            }
        )

    candidates.sort(
        key=lambda item: (
            -item["score"],
            item["wait"],
            -int(item["target"].production),
            int(item["target"].ships),
            item["turns"],
            item["target"].id,
        )
    )
    return candidates


def build_opening_priority_moves(world):
    if (
        not OPENING_PRIORITY_ENABLED
        or not world.is_opening
        or world.step > OPENING_PRIORITY_TURN_LIMIT
        or len(world.my_planets) > OPENING_PRIORITY_MAX_PLANETS
    ):
        return None

    source_order = sorted(
        [
            planet
            for planet in world.my_planets
            if opening_priority_profile_active(planet, world)
        ],
        key=lambda planet: (
            0 if is_profile_home(planet) else 1,
            -int(planet.production),
            -int(planet.ships),
            planet.id,
        ),
    )
    if not source_order:
        return None

    for src in source_order:
        keep = opening_priority_source_keep(src, world)
        available_now = max(0, int(src.ships) - keep)
        if available_now <= 0:
            continue

        candidates = opening_priority_target_candidates(src, world)
        if not candidates:
            continue

        best = candidates[0]
        affordable = [item for item in candidates if item["needed"] <= available_now]
        chosen = None
        if affordable:
            top_affordable = affordable[0]
            if top_affordable["score"] >= best["score"] * OPENING_PRIORITY_ALT_AFFORDABLE_RATIO:
                chosen = top_affordable
            elif (
                world.step <= OPENING_PRIORITY_WAIT_TURN_LIMIT
                and best["wait"] <= OPENING_PRIORITY_MAX_WAIT
                and best["score"] >= top_affordable["score"] * OPENING_PRIORITY_WAIT_ADVANTAGE
            ):
                return []
            else:
                chosen = top_affordable
        elif (
            world.step <= OPENING_PRIORITY_WAIT_TURN_LIMIT
            and best["wait"] <= OPENING_PRIORITY_MAX_WAIT
        ):
            return []

        if chosen is None:
            continue

        target = chosen["target"]
        margin = min(
            OPENING_PRIORITY_MARGIN_CAP,
            NEUTRAL_MARGIN_BASE + int(target.production),
        )
        send = min(available_now, int(chosen["needed"]) + margin)
        if send < int(chosen["needed"]):
            continue
        aim = opening_priority_plan_shot(world, src, target, send)
        if aim is None:
            continue
        angle, _, _, _ = aim
        return [[src.id, float(angle), int(send)]]

    return None


def opening_heavy_prize_source_available(src, world=None):
    reserve = OPENING_HEAVY_PRIZE_RESERVE_BASE + int(src.production) * OPENING_HEAVY_PRIZE_RESERVE_PROD
    if int(src.production) >= CORE_PRODUCTION:
        reserve += 4
    fresh_core_locked = world is not None and fresh_neutral_core_source_lock(src, world)
    if fresh_core_locked:
        reserve = max(reserve, FRESH_CORE_SOURCE_LOCK_KEEP)
    available = max(0, int(src.ships) - reserve)
    if fresh_core_locked:
        available = min(
            available,
            int(max(0, int(src.ships) - reserve) * FRESH_CORE_SOURCE_LOCK_FRACTION),
        )
    return available


def my_incoming_ships_to(target, world):
    return sum(
        int(ships)
        for _, owner, ships in world.arrivals_by_planet.get(target.id, [])
        if owner == world.player and ships > 0
    )


def build_opening_heavy_prize_moves(world):
    if (
        not world.is_opening
        or world.step < OPENING_HEAVY_PRIZE_PLAN_START
        or world.step > OPENING_HEAVY_PRIZE_PLAN_END
        or len(world.my_planets) < 3
        or world.my_prod < 16
    ):
        return None

    candidates = []
    for target in world.neutral_planets:
        if target.id in world.comet_ids:
            continue
        if target.production < OPENING_HEAVY_PRIZE_MIN_PROD:
            continue
        if int(target.ships) < OPENING_HEAVY_PRIZE_MIN_SHIPS:
            continue
        if my_incoming_ships_to(target, world) >= int(target.ships) + 1:
            continue
        my_dist = nearest_distance_to_set(target.x, target.y, world.my_planets)
        enemy_dist = nearest_distance_to_set(target.x, target.y, world.enemy_planets)
        if enemy_dist + 10.0 < my_dist:
            continue
        score = target.production / max(1.0, int(target.ships) + my_dist * 0.65)
        candidates.append((score, target, my_dist))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], -int(item[1].production), item[2], item[1].id))

    for _, target, _ in candidates[:3]:
        options = []
        for src in world.my_planets:
            if is_core_planet(src, world):
                fleet_eta, fleet_stack = enemy_fleet_pressure_to_planet(
                    src,
                    world,
                    OPENING_HEAVY_PRIZE_SOURCE_THREAT_HORIZON,
                )
                if (
                    fleet_eta is not None
                    and fleet_stack >= max(8, int(src.production) * 3)
                ):
                    continue
            available = opening_heavy_prize_source_available(src, world)
            if available < OPENING_HEAVY_PRIZE_MIN_SOURCE:
                continue
            send_probe = min(available, max(OPENING_HEAVY_PRIZE_MIN_SOURCE, int(target.ships) + 1))
            aim = world.plan_shot(src.id, target.id, send_probe)
            if aim is None:
                continue
            angle, turns, _, _ = aim
            if not candidate_time_valid(target, turns, world, LATE_CAPTURE_BUFFER):
                continue
            options.append((turns, -available, src, available, angle))

        if len(options) < 2:
            continue
        options.sort()
        selected = options[:OPENING_HEAVY_PRIZE_MAX_SOURCES]
        total_available = sum(item[3] for item in selected)
        joint_turn = max(item[0] for item in selected)
        need = world.min_ships_to_own_at(
            target.id,
            joint_turn,
            world.player,
            upper_bound=total_available,
        )
        desired = max(need + OPENING_HEAVY_PRIZE_MARGIN, int(target.ships) + 1)
        if need <= 0 or total_available < desired:
            continue

        remaining = desired
        moves = []
        ordered = sorted(selected, key=lambda item: (item[0], item[2].id))
        for idx, (turns, _, src, available, _) in enumerate(ordered):
            rest = sum(item[3] for item in ordered[idx + 1 :])
            send = min(available, max(0, remaining - rest))
            if send <= 0:
                continue
            aim = world.plan_shot(src.id, target.id, send)
            if aim is None:
                moves = []
                break
            angle, _, _, _ = aim
            moves.append([src.id, float(angle), int(send)])
            remaining -= send
        if remaining <= 0 and moves:
            return moves

    return None


def angle_delta(first, second):
    return abs(math.atan2(math.sin(first - second), math.cos(first - second)))


def infer_opening_move_target(world, move):
    if len(move) < 3:
        return None, None
    src_id, angle, ships = move
    source = world.planet_by_id.get(src_id)
    if source is None:
        return None, None

    target_id = first_current_ray_hit(world, src_id, angle)
    if target_id is not None:
        target = world.planet_by_id.get(target_id)
        if target is not None and target.owner != world.player:
            aim = world.plan_shot(src_id, target_id, ships)
            turns = aim[1] if aim is not None else max(1, int(planet_distance(source, target) / fleet_speed(max(1, int(ships)))))
            return target, turns

    best = None
    best_key = None
    for target in world.planets:
        if target.id == src_id or target.owner == world.player:
            continue
        aim = world.plan_shot(src_id, target.id, ships)
        if aim is None:
            continue
        aim_angle, turns, _, _ = aim
        delta = angle_delta(float(angle), aim_angle)
        if delta > 0.22:
            continue
        key = (delta, turns, -int(target.production), int(target.ships), target.id)
        if best_key is None or key < best_key:
            best_key = key
            best = (target, turns)
    if best is None:
        return None, None
    return best


def score_opening_move(move, world):
    if len(move) < 3:
        return None
    src_id, _, ships = move
    source = world.planet_by_id.get(src_id)
    if source is None or int(ships) <= 0:
        return None
    target, turns = infer_opening_move_target(world, move)
    if target is None or target.owner == world.player:
        return None
    if target.id in world.comet_ids:
        return None

    turns = max(1, int(math.ceil(turns or 1)))
    ships = int(ships)
    base = opening_target_score(source, target, world, turns)
    if target.owner not in (-1, world.player):
        base *= 1.25
    if int(target.production) >= CORE_PRODUCTION:
        base *= OPENING_META_CORE_CAPTURE_BONUS
    if world.is_static(target.id):
        base *= OPENING_META_STATIC_CAPTURE_BONUS

    owner_after, _ = world.projected_state(
        target.id,
        turns,
        extra_arrivals=[(turns, world.player, ships)],
    )
    if owner_after != world.player:
        base *= OPENING_META_UNCAPTURED_PENALTY

    my_dist = planet_distance(source, target)
    enemy_dist = nearest_distance_to_set(target.x, target.y, world.enemy_planets)
    if enemy_dist + 4.0 < my_dist:
        base *= OPENING_META_ENEMY_RACE_PENALTY

    if target.owner == -1 and target.production <= 1:
        overspend = ships - (int(target.ships) + OPENING_META_LOW_PROD_OVERSPEND)
        if overspend > 0:
            base *= 1.0 / (1.0 + overspend * 0.08)

    remaining = int(source.ships) - ships
    if is_profile_home(source) and remaining <= 0 and source.production >= CORE_PRODUCTION:
        base *= OPENING_META_HOME_EMPTY_PENALTY

    return base / max(1.0, ships * 0.18 + turns * 0.75)


def score_opening_moves(stage, moves, world):
    if moves is None:
        return None
    if not moves:
        return (
            OPENING_META_WAIT_SCORE,
            stage,
            moves,
            {"wait": True, "score": OPENING_META_WAIT_SCORE},
        )

    score = 0.0
    details = []
    source_used = defaultdict(int)
    target_ids = set()
    for move in moves:
        if len(move) < 3:
            return None
        source = world.planet_by_id.get(move[0])
        if source is None:
            return None
        source_used[source.id] += int(move[2])
        if source_used[source.id] > int(source.ships):
            return None
        move_score = score_opening_move(move, world)
        if move_score is None:
            return None
        target, turns = infer_opening_move_target(world, move)
        target_ids.add(target.id if target is not None else -1)
        score += move_score
        details.append([move[0], target.id if target is not None else None, int(move[2]), int(turns or 0), round(move_score, 4)])

    if len(target_ids) < len(moves):
        score *= 0.92
    score *= OPENING_META_STAGE_WEIGHTS.get(stage, 1.0)
    return (
        score,
        stage,
        moves,
        {"wait": False, "score": round(score, 4), "moves": details},
    )


def build_opening_meta_moves(world, debug_set=None):
    if not OPENING_META_ENABLED or not world.is_opening:
        return None

    builders = (
        ("opening_direct", build_opening_direct_expand_moves),
        ("opening_fast_expand", build_opening_fast_expand_moves),
        ("opening_anchor", build_opening_anchor_moves),
        ("opening_priority", build_opening_priority_moves),
        ("opening_local_quality", build_opening_local_quality_moves),
        ("opening_mainline", build_opening_mainline_moves),
        ("opening_heavy_prize", build_opening_heavy_prize_moves),
    )

    candidates = []
    for stage, builder in builders:
        moves = builder(world)
        scored = score_opening_moves(stage, moves, world)
        if scored is not None:
            candidates.append(scored)

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], item[1]))
    best_score, best_stage, best_moves, best_debug = candidates[0]
    if debug_set is not None:
        debug_set(
            "opening_meta",
            [
                [stage, round(score, 4), len(moves)]
                for score, stage, moves, _ in candidates[:ORBIT_LOG_SAMPLE_LIMIT]
            ],
        )
        debug_set("opening_meta_choice", best_debug)

    if not best_moves:
        return []
    if best_score < OPENING_META_MIN_ACTION_SCORE:
        return None
    return best_moves


def swarm_eta_tolerance(options, target, world):
    tolerance = MULTI_SOURCE_ETA_TOLERANCE
    if len(options) >= 3:
        tolerance = THREE_SOURCE_ETA_TOLERANCE
    if target.owner not in (-1, world.player):
        tolerance = HOSTILE_SWARM_ETA_TOLERANCE

    max_enemy_prod = max(
        (
            int(prod)
            for owner, prod in world.owner_production.items()
            if owner not in (-1, world.player)
        ),
        default=0,
    )
    if world.my_prod + BEHIND_HOSTILE_BREAK_OWNER_PROD_GAP < max_enemy_prod:
        tolerance += BEHIND_SWARM_ETA_BONUS
        if target.owner not in (-1, world.player):
            tolerance += BEHIND_HOSTILE_SWARM_ETA_BONUS
    return tolerance


def anchored_swarm_worth_trying(target, world, modes):
    if target.owner == world.player or world.is_very_late:
        return False
    if target.owner == -1 and target.production < MIDGAME_NEUTRAL_FIRST_MIN_PROD + 1:
        return False
    max_enemy_prod = max(
        (
            int(prod)
            for owner, prod in world.owner_production.items()
            if owner not in (-1, world.player)
        ),
        default=0,
    )
    return (
        target.owner not in (-1, world.player)
        or modes.get("is_behind")
        or world.my_prod + BEHIND_HOSTILE_BREAK_OWNER_PROD_GAP < max_enemy_prod
    )


def detect_enemy_crashes(world):
    crashes = []
    for target_id, arrivals in world.arrivals_by_planet.items():
        enemy_events = [
            (int(math.ceil(eta)), owner, int(ships))
            for eta, owner, ships in arrivals
            if owner not in (-1, world.player) and ships > 0
        ]
        enemy_events.sort()
        for i in range(len(enemy_events)):
            eta_a, owner_a, ships_a = enemy_events[i]
            for j in range(i + 1, len(enemy_events)):
                eta_b, owner_b, ships_b = enemy_events[j]
                if owner_a == owner_b:
                    continue
                if abs(eta_a - eta_b) > CRASH_EXPLOIT_ETA_WINDOW:
                    break
                if ships_a + ships_b < CRASH_EXPLOIT_MIN_TOTAL_SHIPS:
                    continue
                crashes.append(
                    {
                        "target_id": target_id,
                        "crash_turn": max(eta_a, eta_b),
                        "owners": (owner_a, owner_b),
                        "ships": (ships_a, ships_b),
                    }
                )
    return crashes


def build_policy_state(world, deadline=None):
    def expired():
        return deadline is not None and time.perf_counter() > deadline

    indirect_wealth_map = {}
    for target_id, features in world.indirect_feature_map.items():
        friendly, neutral, enemy = features
        indirect_wealth_map[target_id] = (
            friendly * INDIRECT_FRIENDLY_WEIGHT
            + neutral * INDIRECT_NEUTRAL_WEIGHT
            + enemy * INDIRECT_ENEMY_WEIGHT
        )

    reserve = {}
    attack_budget = {}
    reaction_time_map = {}

    for target in world.planets:
        if expired():
            break
        if target.owner == world.player:
            continue
        my_sources = nearest_sources_to_target(target, world.my_planets, REACTION_SOURCE_TOP_K_MY)
        enemy_sources = nearest_sources_to_target(target, world.enemy_planets, REACTION_SOURCE_TOP_K_ENEMY)
        my_t = min_legal_reaction_time(target, my_sources, world)
        enemy_t = min_legal_reaction_time(target, enemy_sources, world)
        reaction_time_map[target.id] = (my_t, enemy_t)

    for planet in world.my_planets:
        if expired():
            break
        exact_keep = world.keep_needed_map.get(planet.id, 0)
        defense_horizon = (
            AGGRESSIVE_DEFENSE_HORIZON
            if AGGRESSIVE_DEFENSE_ENABLED
            else PROACTIVE_DEFENSE_HORIZON
        )
        defense_ratio = (
            AGGRESSIVE_DEFENSE_RATIO
            if AGGRESSIVE_DEFENSE_ENABLED
            else PROACTIVE_DEFENSE_RATIO
        )

        proactive_keep = 0
        prep_window_open = enemy_occupation_prep_window_open(world)
        predictive_defense_allowed = not (
            world.my_prod >= world.enemy_prod + 6
            and world.my_total >= world.enemy_total * 1.05
        )
        predicted_enemy_eta = None
        predicted_enemy_stack = 0
        if prep_window_open:
            fleet_horizon = defense_horizon
            if is_defense_core_planet(planet, world):
                fleet_horizon = max(fleet_horizon, CORE_THREAT_HORIZON)
            fleet_horizon = min(fleet_horizon, PREDICTED_FLEET_THREAT_HORIZON)
            if predictive_defense_allowed:
                predicted_enemy_eta, predicted_enemy_stack = enemy_fleet_pressure_to_planet(
                    planet,
                    world,
                    fleet_horizon,
                )
                if predicted_enemy_eta is not None:
                    fleet_ratio = defense_ratio
                    if is_defense_core_planet(planet, world):
                        fleet_ratio = max(fleet_ratio, CORE_THREAT_RATIO)
                        if predicted_enemy_eta <= CORE_URGENT_THREAT_TURN:
                            fleet_ratio = max(fleet_ratio, CORE_URGENT_THREAT_RATIO)
                    proactive_keep = max(proactive_keep, int(predicted_enemy_stack * fleet_ratio))

        reserve[planet.id] = min(int(planet.ships), max(exact_keep, proactive_keep))
        fall_turn = world.fall_turn_map.get(planet.id)
        first_enemy = world.first_enemy_map.get(planet.id)
        if predicted_enemy_eta is not None:
            first_enemy = (
                predicted_enemy_eta
                if first_enemy is None
                else min(first_enemy, predicted_enemy_eta)
            )
        if (
            world.step >= RESERVE_RELIEF_START_STEP
            and exact_keep <= 0
            and first_enemy is None
            and fall_turn is None
            and reserve[planet.id] >= int(planet.ships)
            and world.enemy_planets
        ):
            max_enemy_prod = max(
                (
                    int(production)
                    for owner, production in world.owner_production.items()
                    if owner not in (-1, world.player)
                ),
                default=0,
            )
            prod_trapped = world.my_prod + RESERVE_RELIEF_PROD_GAP < max_enemy_prod
            total_trapped = world.my_total < world.max_enemy_strength * RESERVE_RELIEF_TOTAL_RATIO
            if prod_trapped or total_trapped:
                relief_keep = max(
                    RESERVE_RELIEF_KEEP_BASE + int(planet.production) * RESERVE_RELIEF_PROD_WEIGHT,
                    int(int(planet.ships) * RESERVE_RELIEF_KEEP_FRACTION),
                )
                reserve[planet.id] = min(reserve[planet.id], relief_keep)
        available = max(0, int(planet.ships) - reserve[planet.id])

        attack_budget[planet.id] = available

    return {
        "indirect_wealth_map": indirect_wealth_map,
        "reserve": reserve,
        "attack_budget": attack_budget,
        "reaction_time_map": reaction_time_map,
        "local_neutral_opportunity": local_neutral_opportunity(world),
    }


def build_modes(world):
    domination = (world.my_total - world.enemy_total) / max(1, world.my_total + world.enemy_total)
    is_behind = domination < BEHIND_DOMINATION
    is_ahead = domination > AHEAD_DOMINATION
    is_dominating = is_ahead or (
        world.max_enemy_strength > 0 and world.my_total > world.max_enemy_strength * 1.25
    )
    is_finishing = (
        domination > FINISHING_DOMINATION
        and world.my_prod > world.enemy_prod * FINISHING_PROD_RATIO
        and world.step > 100
    )

    attack_margin_mult = 1.0
    if is_ahead:
        attack_margin_mult += AHEAD_ATTACK_MARGIN_BONUS
    if is_behind:
        attack_margin_mult -= BEHIND_ATTACK_MARGIN_PENALTY
    if is_finishing:
        attack_margin_mult += FINISHING_ATTACK_MARGIN_BONUS

    return {
        "domination": domination,
        "is_behind": is_behind,
        "is_ahead": is_ahead,
        "is_dominating": is_dominating,
        "is_finishing": is_finishing,
        "attack_margin_mult": attack_margin_mult,
    }


def is_safe_neutral(target, policy):
    if target.owner != -1:
        return False
    my_t, enemy_t = policy_reaction_times(target.id, policy)
    return my_t <= enemy_t - SAFE_NEUTRAL_MARGIN


def is_contested_neutral(target, policy):
    if target.owner != -1:
        return False
    my_t, enemy_t = policy_reaction_times(target.id, policy)
    return abs(my_t - enemy_t) <= CONTESTED_NEUTRAL_MARGIN


def local_neutral_multiplier(target, world, policy):
    if target.owner != -1 or target.id in world.comet_ids:
        return 1.0

    my_dist = nearest_distance_to_set(target.x, target.y, world.my_planets)
    if my_dist > LOCAL_NEUTRAL_OUTER_DIST:
        return 1.0

    enemy_dist = nearest_distance_to_set(target.x, target.y, world.enemy_planets)
    proximity = (LOCAL_NEUTRAL_OUTER_DIST - my_dist) / LOCAL_NEUTRAL_OUTER_DIST
    affordability = max(0.2, (LOCAL_NEUTRAL_SOFT_SHIPS - int(target.ships)) / LOCAL_NEUTRAL_SOFT_SHIPS)
    production_pull = min(1.0, target.production / 4.0)
    enemy_race = 1.0
    if enemy_dist < my_dist - 3.0:
        enemy_race = 0.45
    elif enemy_dist <= my_dist + 4.0:
        enemy_race = 0.75

    bonus = LOCAL_NEUTRAL_MAX_BONUS * proximity * affordability * production_pull * enemy_race
    return 1.0 + max(0.0, bonus)


def local_neutral_opportunity(world):
    best = 0.0
    for target in world.neutral_planets:
        if target.id in world.comet_ids:
            continue
        if target.production < 2:
            continue
        my_dist = nearest_distance_to_set(target.x, target.y, world.my_planets)
        if my_dist > LOCAL_NEUTRAL_OUTER_DIST:
            continue
        enemy_dist = nearest_distance_to_set(target.x, target.y, world.enemy_planets)
        if enemy_dist < my_dist - 3.0:
            continue
        proximity = (LOCAL_NEUTRAL_OUTER_DIST - my_dist) / LOCAL_NEUTRAL_OUTER_DIST
        roi = target.production / max(1.0, int(target.ships) + 1.0)
        best = max(best, proximity * roi * 10.0)
    return best


def valuable_neutral_remaining(world):
    if world.step >= MIDGAME_NEUTRAL_FIRST_TURN_LIMIT:
        return False
    for target in world.neutral_planets:
        if target.id in world.comet_ids:
            continue
        if target.production >= MIDGAME_NEUTRAL_FIRST_MIN_PROD:
            return True
    return False


def opening_filter(target, arrival_turns, needed, src_available, world, policy):
    if not world.is_opening or target.owner != -1:
        return False
    if target.id in world.comet_ids:
        return False
    if opening_fill_target(target, world):
        if target.production <= 1 and (
            needed > OPENING_FILL_LOW_PROD_MAX_NEEDED
            or needed > int(target.ships) + OPENING_FILL_LOW_PROD_MAX_OVER_NEUTRAL
        ):
            return True
        if target.production <= 2 and needed > OPENING_FILL_WEAK_PROD_MAX_NEEDED:
            return True
        return False
    if poor_opening_target(target, world):
        return True
    if opening_far_neutral_detour(target, arrival_turns, world):
        return True
    if world.is_static(target.id):
        return False

    my_t, enemy_t = policy_reaction_times(target.id, policy)
    reaction_gap = enemy_t - my_t
    if (
        target.production >= SAFE_OPENING_PROD_THRESHOLD
        and arrival_turns <= SAFE_OPENING_TURN_LIMIT
        and reaction_gap >= SAFE_NEUTRAL_MARGIN
    ):
        return False

    if world.is_four_player:
        affordable = needed <= max(
            PARTIAL_SOURCE_MIN_SHIPS,
            int(src_available * FOUR_PLAYER_ROTATING_SEND_RATIO),
        )
        if (
            affordable
            and arrival_turns <= FOUR_PLAYER_ROTATING_TURN_LIMIT
            and reaction_gap >= FOUR_PLAYER_ROTATING_REACTION_GAP
        ):
            return False
        return True

    return arrival_turns > ROTATING_OPENING_MAX_TURNS or target.production <= ROTATING_OPENING_LOW_PROD


def reserve_source_for_local_prize(src, target, world):
    if not world.is_early or target.owner != -1 or target.id in world.comet_ids:
        return False
    src_radius = orbital_radius(src)
    src_diag = abs((src.x - CENTER_X) - (src.y - CENTER_Y))
    target_dist = planet_distance(src, target)
    static_p2_reserve = (
        is_static_planet(src)
        and src.production == 2
        and 42.0 <= src_radius <= 55.0
        and src_diag <= 8.0
    )
    low_value_detour = target.production <= 1
    far_detour = target.production <= 3 and target_dist >= 35.0

    if not (static_p2_reserve or EDGE_AIM_ENABLED):
        return False
    if not (static_p2_reserve or low_value_detour or far_detour):
        return False
    if target.production >= LOCAL_PRIZE_MIN_PRODUCTION:
        if not far_detour:
            return False

    future_budget = int(src.ships) + int(src.production) * LOCAL_PRIZE_RESERVE_TURNS
    best_prize = None
    best_key = None
    required_production = LOCAL_PRIZE_MIN_PRODUCTION
    if not far_detour:
        required_production = max(
            LOCAL_PRIZE_MIN_PRODUCTION,
            target.production + LOCAL_PRIZE_PROD_GAP,
        )

    for prize in world.neutral_planets:
        if prize.id == target.id or prize.id in world.comet_ids:
            continue
        if prize.production < required_production:
            continue

        prize_dist = planet_distance(src, prize)
        if prize_dist > LOCAL_PRIZE_MAX_DIST:
            continue
        if int(prize.ships) + 1 > future_budget:
            continue

        enemy_dist = nearest_distance_to_set(prize.x, prize.y, world.enemy_planets)
        if enemy_dist < prize_dist - LOCAL_PRIZE_ENEMY_LEAD_ALLOWANCE:
            continue

        prize_key = (
            -prize.production,
            int(prize.ships),
            prize_dist,
            prize.id,
        )
        if best_key is None or prize_key < best_key:
            best_key = prize_key
            best_prize = (prize, prize_dist)

    if best_prize is None:
        return False

    prize, prize_dist = best_prize
    if target.production <= 1:
        return True
    return target_dist > prize_dist + LOCAL_PRIZE_FAR_EXTRA_DIST


def opening_home_half_core_multiplier(target, world):
    if (
        not world.is_opening
        or world.is_four_player
        or target.owner != -1
        or target.production < CORE_PRODUCTION
    ):
        return 1.0

    target_initial = world.initial_by_id.get(target.id) or target
    homes = [
        world.initial_by_id.get(home_id) or world.planet_by_id.get(home_id)
        for home_id in PROFILE_HOME_IDS
    ]
    homes = [home for home in homes if home is not None]
    if not homes:
        return 1.0

    def side_of(planet):
        return 1 if float(planet.x) >= CENTER_X else -1

    home_sides = {side_of(home) for home in homes}
    target_same_side = side_of(target_initial) in home_sides
    if target_same_side:
        return OPENING_HOME_HALF_CORE_MULT

    for other in world.neutral_planets:
        if other.id == target.id or other.id in world.comet_ids:
            continue
        if other.production < CORE_PRODUCTION:
            continue
        other_initial = world.initial_by_id.get(other.id) or other
        if side_of(other_initial) not in home_sides:
            continue
        if int(other.ships) <= int(target.ships) + OPENING_HOME_HALF_CORE_SHIP_ALLOWANCE:
            return OPENING_AWAY_HALF_CORE_PENALTY
    return 1.0


def opening_away_half_core_blocked(target, world):
    if opening_home_half_core_multiplier(target, world) >= 1.0:
        return False
    if not world.my_planets:
        return False

    homes = [
        world.initial_by_id.get(home_id) or world.planet_by_id.get(home_id)
        for home_id in PROFILE_HOME_IDS
    ]
    homes = [home for home in homes if home is not None]
    if not homes:
        return False

    def side_of(planet):
        return 1 if float(planet.x) >= CENTER_X else -1

    home_sides = {side_of(home) for home in homes}
    for other in world.neutral_planets:
        if other.id == target.id or other.id in world.comet_ids:
            continue
        if other.production < CORE_PRODUCTION:
            continue
        other_initial = world.initial_by_id.get(other.id) or other
        if side_of(other_initial) not in home_sides:
            continue
        if int(other.ships) > int(target.ships) + OPENING_HOME_HALF_CORE_SHIP_ALLOWANCE:
            continue
        for src in world.my_planets:
            cap = int(src.ships)
            if cap <= int(other.ships):
                continue
            seeded = world.best_probe_aim(
                src.id,
                other.id,
                cap,
                hints=(int(other.ships) + 1,),
            )
            if seeded is None:
                continue
            _, aim = seeded
            turns = aim[1]
            needed = world.min_ships_to_own_at(
                other.id,
                turns,
                world.player,
                upper_bound=cap,
            )
            if needed > 0 and needed <= cap:
                return True
    return False


def target_value(target, arrival_turns, mission, world, modes, policy):
    turns_profit = max(1, world.remaining_steps - arrival_turns)
    if target.id in world.comet_ids:
        life = world.comet_life(target.id)
        turns_profit = max(0, min(turns_profit, life - arrival_turns))
        if turns_profit <= 0:
            return -1.0

    value = target.production * turns_profit
    value += policy["indirect_wealth_map"][target.id] * turns_profit * INDIRECT_VALUE_SCALE

    if world.is_static(target.id):
        value *= STATIC_NEUTRAL_VALUE_MULT if target.owner == -1 else STATIC_HOSTILE_VALUE_MULT
    else:
        value *= ROTATING_OPENING_VALUE_MULT if world.is_opening else 1.0

    if target.owner not in (-1, world.player):
        value *= OPENING_HOSTILE_TARGET_VALUE_MULT if world.is_opening else HOSTILE_TARGET_VALUE_MULT
        leader_ahead = (
            world.is_four_player
            and
            world.step >= LEADER_SUPPRESSION_START_STEP
            and world.leading_enemy_owner is not None
            and (
                world.leading_enemy_prod >= world.my_prod + LEADER_SUPPRESSION_PROD_GAP
                or world.leading_enemy_strength >= world.my_total * LEADER_SUPPRESSION_STRENGTH_RATIO
            )
        )
        if (
            leader_ahead
            and target.owner == world.leading_enemy_owner
        ):
            value *= LEADER_SUPPRESSION_SCORE_MULT
            if target.production >= CORE_PRODUCTION:
                value *= LEADER_SUPPRESSION_CORE_SCORE_MULT
        elif leader_ahead and world.is_four_player:
            value *= FOUR_PLAYER_NON_LEADER_HOSTILE_MULT
        if initial_owner_of(target, world) == world.player:
            value *= RECLAIM_HOME_VALUE_MULT
        elif valuable_neutral_remaining(world) and not modes["is_finishing"]:
            value *= (
                MIDGAME_CORE_HOSTILE_WITH_NEUTRALS_MULT
                if target.production >= CORE_PRODUCTION
                else MIDGAME_HOSTILE_WITH_NEUTRALS_MULT
            )

    if target.owner == -1:
        value *= opening_home_half_core_multiplier(target, world)
        value *= opening_roi_multiplier(target, arrival_turns, world)
        if is_safe_neutral(target, policy):
            value *= SAFE_NEUTRAL_VALUE_MULT
        elif is_contested_neutral(target, policy):
            value *= CONTESTED_NEUTRAL_VALUE_MULT
        if world.is_early:
            value *= local_neutral_multiplier(target, world, policy)
            local_pressure = policy.get("local_neutral_opportunity", 0.0)
            my_dist = nearest_distance_to_set(target.x, target.y, world.my_planets)
            if (
                local_pressure >= 1.2
                and my_dist > FAR_NEUTRAL_OPPORTUNITY_DIST
                and target.production <= 2
            ):
                value *= FAR_NEUTRAL_OPPORTUNITY_PENALTY
        if world.is_early:
            value *= EARLY_NEUTRAL_VALUE_MULT
            if (
                LOCAL_OPENING_ENABLED
                and target.production >= 2
                and int(target.ships) <= 10
                and nearest_distance_to_set(target.x, target.y, world.my_planets) <= 18.0
            ):
                value *= 1.8

    if target.id in world.comet_ids:
        value *= COMET_VALUE_MULT

    if mission == "snipe":
        value *= SNIPE_VALUE_MULT
    elif mission == "swarm":
        value *= SWARM_VALUE_MULT
    elif mission == "reinforce":
        value *= REINFORCE_VALUE_MULT
    elif mission == "crash_exploit":
        value *= CRASH_EXPLOIT_VALUE_MULT

    if world.is_late:
        value += max(0, target.ships) * LATE_IMMEDIATE_SHIP_VALUE
        if target.owner not in (-1, world.player):
            enemy_strength = world.owner_strength.get(target.owner, 0)
            if enemy_strength <= WEAK_ENEMY_THRESHOLD:
                value += ELIMINATION_BONUS

    if modes["is_finishing"] and target.owner not in (-1, world.player):
        value *= FINISHING_HOSTILE_VALUE_MULT
    if target.owner not in (-1, world.player) and target.production >= CORE_PRODUCTION:
        my_t, enemy_t = policy_reaction_times(target.id, policy)
        if my_t <= enemy_t + 2:
            value *= THREATENED_CORE_VALUE_MULT
    if modes["is_behind"] and target.owner == -1 and not world.is_static(target.id):
        value *= BEHIND_ROTATING_NEUTRAL_VALUE_MULT
    if modes["is_behind"] and target.owner == -1 and is_safe_neutral(target, policy):
        value *= 1.08
    if modes["is_dominating"] and target.owner == -1 and is_contested_neutral(target, policy):
        value *= 0.92

    return value


def reinforce_value(target, hold_until, world, policy):
    saved_turns = max(1, world.remaining_steps - hold_until)
    value = target.production * saved_turns + max(0, target.ships) * DEFENSE_SHIP_VALUE
    if world.enemy_planets and nearest_distance_to_set(target.x, target.y, world.enemy_planets) < 22:
        value *= DEFENSE_FRONTIER_SCORE_MULT
    value += policy["indirect_wealth_map"][target.id] * saved_turns * INDIRECT_VALUE_SCALE * 0.35
    return value * REINFORCE_VALUE_MULT


def post_capture_enemy_stack(target, arrival_turns, world):
    stack = 0
    start = int(math.ceil(arrival_turns))
    window = (
        POST_CAPTURE_HIGH_PROD_HOLD_WINDOW
        if target.production >= 5
        else POST_CAPTURE_HOLD_WINDOW
    )
    end = start + window
    for eta, owner, ships in world.arrivals_by_planet.get(target.id, []):
        if owner == world.player:
            continue
        if start < eta <= end:
            stack += int(ships)
    return stack


def preferred_send(target, base_needed, arrival_turns, src_available, world, modes, policy):
    send = max(base_needed, int(math.ceil(base_needed * modes["attack_margin_mult"])))
    margin = 0
    if (
        LEAN_OPENING_ENABLED
        and world.is_early
        and target.owner == -1
        and target.production >= 3
        and int(target.ships) <= 10
    ):
        return min(src_available, max(send, base_needed + 1))
    if target.owner == -1:
        margin += min(
            NEUTRAL_MARGIN_CAP,
            NEUTRAL_MARGIN_BASE + target.production * NEUTRAL_MARGIN_PROD_WEIGHT,
        )
    else:
        margin += min(
            HOSTILE_MARGIN_CAP,
            HOSTILE_MARGIN_BASE + target.production * HOSTILE_MARGIN_PROD_WEIGHT,
        )
    if world.is_static(target.id):
        margin += STATIC_TARGET_MARGIN
    if is_contested_neutral(target, policy):
        margin += CONTESTED_TARGET_MARGIN
    if world.is_four_player:
        margin += FOUR_PLAYER_TARGET_MARGIN
    if arrival_turns > LONG_TRAVEL_MARGIN_START:
        margin += min(LONG_TRAVEL_MARGIN_CAP, arrival_turns // LONG_TRAVEL_MARGIN_DIVISOR)
    if target.id in world.comet_ids:
        margin = max(0, margin - COMET_MARGIN_RELIEF)
    if modes["is_finishing"] and target.owner not in (-1, world.player):
        margin += FINISHING_HOSTILE_SEND_BONUS
    if target.owner == -1 and target.production >= CORE_PRODUCTION:
        followup_stack = post_capture_enemy_stack(target, arrival_turns, world)
        if followup_stack > 0:
            margin += min(
                POST_CAPTURE_HOLD_MARGIN_CAP,
                max(2, int(followup_stack * POST_CAPTURE_HOLD_RATIO) - int(target.production)),
            )
    return min(src_available, send + margin)


def apply_score_modifiers(base_score, target, mission, world):
    score = base_score
    if world.is_static(target.id):
        score *= STATIC_TARGET_SCORE_MULT
    if world.is_early and target.owner == -1 and world.is_static(target.id):
        score *= EARLY_STATIC_NEUTRAL_SCORE_MULT
    if (
        world.is_opening
        and target.owner == -1
        and target.production >= OPENING_HEAVY_PRIZE_MIN_PROD
        and int(target.ships) >= OPENING_HEAVY_PRIZE_MIN_SHIPS
    ):
        score *= OPENING_HEAVY_PRIZE_SCORE_MULT
    if world.is_four_player and target.owner == -1 and not world.is_static(target.id):
        score *= FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT
    if (
        len(world.static_neutral_planets) >= DENSE_STATIC_NEUTRAL_COUNT
        and target.owner == -1
        and not world.is_static(target.id)
    ):
        score *= DENSE_ROTATING_NEUTRAL_SCORE_MULT
    if mission == "snipe":
        score *= SNIPE_SCORE_MULT
    elif mission == "swarm":
        score *= SWARM_SCORE_MULT
    elif mission == "crash_exploit":
        score *= CRASH_EXPLOIT_SCORE_MULT
    return score


def settle_plan(
    src,
    target,
    src_cap,
    send_guess,
    world,
    planned_commitments,
    modes,
    policy,
    mission="capture",
    eval_turn_fn=None,
    anchor_turn=None,
    anchor_tolerance=None,
    max_iter=4,
):
    if src_cap < 1:
        return None

    seed_hint = max(1, min(src_cap, int(send_guess)))
    eval_turn_fn = eval_turn_fn or (lambda turns: turns)
    anchor_tolerance = (
        anchor_tolerance
        if anchor_tolerance is not None
        else (1 if mission == "snipe" else None)
    )
    tested = {}
    tested_order = []

    def evaluate(send):
        send = max(1, min(src_cap, int(send)))
        cached = tested.get(send)
        if cached is not None or send in tested:
            return cached

        aim = world.plan_shot(src.id, target.id, send)
        if aim is None:
            tested[send] = None
            return None

        angle, turns, _, _ = aim
        if mission == "crash_exploit" and anchor_turn is not None and turns < anchor_turn:
            tested[send] = None
            return None
        raw_eval_turn = int(math.ceil(eval_turn_fn(turns)))
        if raw_eval_turn < turns:
            tested[send] = None
            return None
        eval_turn = raw_eval_turn
        need = world.min_ships_to_own_by(
            target.id,
            eval_turn,
            world.player,
            arrival_turn=turns,
            planned_commitments=planned_commitments,
            upper_bound=src_cap,
        )
        if need <= 0 or need > src_cap:
            tested[send] = None
            return None

        if mission in ("snipe", "crash_exploit"):
            desired = need
        elif mission == "rescue":
            desired = min(
                src_cap,
                max(
                    need,
                    need + DEFENSE_SEND_MARGIN_BASE + target.production * DEFENSE_SEND_MARGIN_PROD_WEIGHT,
                ),
            )
        else:
            desired = min(
                src_cap,
                max(need, preferred_send(target, need, turns, src_cap, world, modes, policy)),
            )

        result = (angle, turns, eval_turn, need, send, desired)
        tested[send] = result
        tested_order.append(send)
        return result

    initial_candidates = sorted(
        world.probe_ship_candidates(
            src.id,
            target.id,
            src_cap,
            hints=(seed_hint,),
        ),
        key=lambda send: (abs(send - seed_hint), send),
    )

    current_send = None
    for seed in initial_candidates:
        result = evaluate(seed)
        if result is None:
            continue
        if (
            anchor_turn is not None
            and anchor_tolerance is not None
            and abs(result[1] - anchor_turn) > anchor_tolerance
        ):
            continue
        current_send = seed
        break

    if current_send is None:
        return None

    for _ in range(max_iter):
        result = evaluate(current_send)
        if result is None:
            break

        angle, turns, eval_turn, need, actual_send, desired = result
        if desired == actual_send:
            if (
                anchor_turn is not None
                and anchor_tolerance is not None
                and abs(turns - anchor_turn) > anchor_tolerance
            ):
                return None
            if mission == "rescue" and turns > eval_turn:
                return None
            return angle, turns, eval_turn, need, actual_send

        next_send = max(1, min(src_cap, int(desired)))
        if next_send in tested:
            current_send = next_send
            break
        current_send = next_send

    candidate_sends = sorted(
        [send for send in tested_order if tested.get(send) is not None],
        key=lambda send: (
            0
            if mission != "snipe" or anchor_turn is None
            else abs(tested[send][1] - anchor_turn),
            abs(send - seed_hint),
            tested[send][1],
            send,
        ),
    )

    seen = set()
    for send in candidate_sends:
        if send in seen:
            continue
        seen.add(send)
        result = tested.get(send)
        if result is None:
            continue
        angle, turns, eval_turn, need, actual_send, _ = result
        if actual_send < need:
            continue
        if (
            anchor_turn is not None
            and anchor_tolerance is not None
            and abs(turns - anchor_turn) > anchor_tolerance
        ):
            continue
        if mission == "rescue" and turns > eval_turn:
            continue
        return angle, turns, eval_turn, need, actual_send

    return None


def predict_enemy_arrivals(world, deadline=None):
    if not ENEMY_PREDICTION_ENABLED or world.step > ENEMY_PREDICTION_MAX_STEP:
        return {}
    if len(world.planets) > HEAVY_ROUTE_PLANET_LIMIT:
        return {}

    def expired():
        return deadline is not None and time.perf_counter() > deadline - OPTIONAL_PHASE_MIN_TIME

    predicted = defaultdict(list)
    enemy_owners = sorted(
        owner
        for owner in world.owner_strength
        if owner not in (-1, world.player) and world.owner_strength[owner] > 0
    )

    for owner in enemy_owners:
        if expired():
            break

        enemy_world = WorldModel(
            player=owner,
            step=world.step,
            planets=world.planets,
            fleets=world.fleets,
            initial_by_id=world.initial_by_id,
            ang_vel=world.ang_vel,
            comets=world.comets,
            comet_ids=world.comet_ids,
        )
        enemy_modes = build_modes(enemy_world)
        enemy_policy = build_policy_state(enemy_world, deadline=deadline)
        enemy_policy["predicted_enemy_arrivals"] = {}
        enemy_commitments = defaultdict(list)

        sources = sorted(
            enemy_world.my_planets,
            key=lambda planet: (-int(planet.ships), -int(planet.production), planet.id),
        )[:ENEMY_PREDICTION_SOURCE_LIMIT]

        for src in sources:
            if expired():
                break
            src_available = enemy_policy["attack_budget"].get(src.id, 0)
            if src_available < PARTIAL_SOURCE_MIN_SHIPS:
                continue

            targets = [
                target
                for target in enemy_world.planets
                if target.id != src.id and target.owner != owner
            ]
            targets.sort(
                key=lambda target: (
                    planet_distance(src, target) / max(1, target.production),
                    0 if target.owner == world.player else 1,
                    -int(target.production),
                    int(target.ships),
                )
            )

            best = None
            for target in targets[:ENEMY_PREDICTION_TARGET_LIMIT]:
                if expired():
                    break
                seeded = enemy_world.best_probe_aim(
                    src.id,
                    target.id,
                    src_available,
                    hints=(int(target.ships) + 1,),
                )
                if seeded is None:
                    continue

                _, rough_aim = seeded
                rough_turns = rough_aim[1]
                if not candidate_time_valid(target, rough_turns, enemy_world, LATE_CAPTURE_BUFFER):
                    continue

                rough_needed = enemy_world.min_ships_to_own_at(
                    target.id,
                    rough_turns,
                    owner,
                    planned_commitments=enemy_commitments,
                    upper_bound=src_available,
                )
                if rough_needed <= 0 or rough_needed > src_available:
                    continue
                if opening_filter(
                    target,
                    rough_turns,
                    rough_needed,
                    src_available,
                    enemy_world,
                    enemy_policy,
                ):
                    continue

                send_guess = preferred_send(
                    target,
                    rough_needed,
                    rough_turns,
                    src_available,
                    enemy_world,
                    enemy_modes,
                    enemy_policy,
                )
                plan = settle_plan(
                    src,
                    target,
                    src_available,
                    send_guess,
                    enemy_world,
                    enemy_commitments,
                    enemy_modes,
                    enemy_policy,
                    mission="capture",
                    max_iter=2,
                )
                if plan is None:
                    continue

                angle, turns, _, need, send = plan
                if send < need:
                    continue
                value = target_value(target, turns, "capture", enemy_world, enemy_modes, enemy_policy)
                if value <= 0:
                    continue
                score = apply_score_modifiers(
                    value / (send + turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                    target,
                    "capture",
                    enemy_world,
                )
                if best is None or score > best[0]:
                    best = (score, target.id, turns, send)

            if best is None or best[0] <= ENEMY_PREDICTION_MIN_SCORE:
                continue

            _, target_id, turns, send = best
            predicted[target_id].append((turns, owner, int(send)))
            enemy_commitments[target_id].append((turns, owner, int(send)))

    return dict(predicted)


def settle_reinforce_plan(
    src,
    target,
    src_cap,
    send_guess,
    world,
    planned_commitments,
    hold_until,
    max_arrival_turn,
    max_iter=4,
):
    if src_cap < 1:
        return None

    seed_hint = max(1, min(src_cap, int(send_guess)))
    tested = {}
    tested_order = []

    def evaluate(send):
        send = max(1, min(src_cap, int(send)))
        cached = tested.get(send)
        if cached is not None or send in tested:
            return cached

        aim = world.plan_shot(src.id, target.id, send)
        if aim is None:
            tested[send] = None
            return None

        angle, turns, _, _ = aim
        if turns > max_arrival_turn:
            tested[send] = None
            return None

        need = world.reinforcement_needed_to_hold_until(
            target.id,
            turns,
            hold_until,
            planned_commitments=planned_commitments,
            upper_bound=src_cap,
        )
        if need <= 0 or need > src_cap:
            tested[send] = None
            return None

        desired = min(src_cap, need + REINFORCE_SAFETY_MARGIN)
        result = (angle, turns, hold_until, need, send, desired)
        tested[send] = result
        tested_order.append(send)
        return result

    initial_candidates = sorted(
        world.probe_ship_candidates(
            src.id,
            target.id,
            src_cap,
            hints=(seed_hint,),
        ),
        key=lambda send: (abs(send - seed_hint), send),
    )

    current_send = None
    for seed in initial_candidates:
        result = evaluate(seed)
        if result is None:
            continue
        current_send = seed
        break

    if current_send is None:
        return None

    for _ in range(max_iter):
        result = evaluate(current_send)
        if result is None:
            break

        angle, turns, eval_turn, need, actual_send, desired = result
        if desired == actual_send:
            return angle, turns, eval_turn, need, actual_send

        next_send = max(1, min(src_cap, int(desired)))
        if next_send in tested:
            current_send = next_send
            break
        current_send = next_send

    candidate_sends = sorted(
        [send for send in tested_order if tested.get(send) is not None],
        key=lambda send: (abs(send - seed_hint), tested[send][1], send),
    )
    for send in candidate_sends:
        result = tested.get(send)
        if result is None:
            continue
        angle, turns, eval_turn, need, actual_send, _ = result
        if actual_send < need or turns > max_arrival_turn:
            continue
        return angle, turns, eval_turn, need, actual_send

    return None


def build_snipe_mission(src, target, src_available, world, planned_commitments, modes, policy):
    if target.owner != -1:
        return None

    enemy_etas = sorted(
        {
            int(math.ceil(eta))
            for eta, owner, ships in world.arrivals_by_planet.get(target.id, [])
            if owner not in (-1, world.player) and ships > 0
        }
    )
    if not enemy_etas:
        return None

    best = None
    for enemy_eta in enemy_etas[:3]:
        delays = WAIT_STRIKE_DELAYS if WAIT_STRIKE_ENABLED and DELAYED_SNIPE_ENABLED else (0,)
        for delay in delays:
            desired_turn = enemy_eta + delay
            seeded = world.best_probe_aim(
                src.id,
                target.id,
                src_available,
                hints=(int(target.ships) + 1, int(target.ships) + 8),
                anchor_turn=desired_turn,
                max_anchor_diff=1,
            )
            if seeded is None:
                continue

            probe, rough = seeded
            sync_turn = max(rough[1], desired_turn)
            if target.id in world.comet_ids:
                life = world.comet_life(target.id)
                if sync_turn >= life or sync_turn > COMET_MAX_CHASE_TURNS:
                    continue

            plan = settle_plan(
                src,
                target,
                src_available,
                probe,
                world,
                planned_commitments,
                modes,
                policy,
                mission="snipe",
                eval_turn_fn=lambda turns, desired_turn=desired_turn: max(turns, desired_turn),
                anchor_turn=desired_turn,
            )
            if plan is None:
                continue

            angle, turns, sync_turn, need, send_pref = plan
            if opening_filter(target, turns, need, src_available, world, policy):
                continue
            if target.id in world.comet_ids:
                life = world.comet_life(target.id)
                if sync_turn >= life or sync_turn > COMET_MAX_CHASE_TURNS:
                    continue

            value = target_value(target, sync_turn, "snipe", world, modes, policy)
            if value <= 0:
                continue

            score = apply_score_modifiers(
                value / (send_pref + sync_turn * SNIPE_COST_TURN_WEIGHT + 1.0),
                target,
                "snipe",
                world,
            )
            if delay:
                score *= 1.04
            option = ShotOption(
                score=score,
                src_id=src.id,
                target_id=target.id,
                angle=angle,
                turns=turns,
                needed=need,
                send_cap=send_pref,
                mission="snipe",
                anchor_turn=desired_turn,
            )
            mission_obj = Mission(
                kind="snipe",
                score=score,
                target_id=target.id,
                turns=sync_turn,
                options=[option],
            )
            if best is None or mission_obj.score > best.score:
                best = mission_obj

    return best


def build_rescue_missions(world, policy, planned_commitments, modes):
    missions = []

    for target in world.my_planets:
        fall_turn = world.fall_turn_map.get(target.id)
        if fall_turn is None or fall_turn > DEFENSE_LOOKAHEAD_TURNS:
            continue

        for src in world.my_planets:
            if src.id == target.id:
                continue

            src_available = policy["attack_budget"].get(src.id, 0)
            if src_available < PARTIAL_SOURCE_MIN_SHIPS:
                continue

            seeded = world.best_probe_aim(
                src.id,
                target.id,
                src_available,
                hints=(target.production + DEFENSE_SEND_MARGIN_BASE + 2,),
                max_turn=fall_turn,
            )
            if seeded is None:
                continue
            probe, probe_aim = seeded
            plan = settle_plan(
                src,
                target,
                src_available,
                probe,
                world,
                planned_commitments,
                modes,
                policy,
                mission="rescue",
                eval_turn_fn=lambda _turns, fall_turn=fall_turn: fall_turn,
                anchor_turn=fall_turn,
            )
            if plan is None:
                continue

            angle, turns, _, need, send_pref = plan
            saved_turns = max(1, world.remaining_steps - fall_turn)
            value = target.production * saved_turns + max(0, target.ships) * DEFENSE_SHIP_VALUE
            if world.enemy_planets and nearest_distance_to_set(target.x, target.y, world.enemy_planets) < 22:
                value *= DEFENSE_FRONTIER_SCORE_MULT
            score = value / (send_pref + turns * DEFENSE_COST_TURN_WEIGHT + 1.0)
            if fall_turn <= URGENT_DEFENSE_TURN:
                score *= URGENT_DEFENSE_SCORE_MULT
            else:
                score *= EARLY_DEFENSE_SCORE_MULT
            if is_defense_core_planet(target, world):
                score *= CORE_DEFENSE_SCORE_MULT
                if fall_turn <= CORE_VISIBLE_THREAT_TURN:
                    score *= CORE_URGENT_DEFENSE_SCORE_MULT

            option = ShotOption(
                score=score,
                src_id=src.id,
                target_id=target.id,
                angle=angle,
                turns=turns,
                needed=need,
                send_cap=send_pref,
                mission="rescue",
                anchor_turn=fall_turn,
            )
            missions.append(
                Mission(
                    kind="rescue",
                    score=score,
                    target_id=target.id,
                    turns=fall_turn,
                    options=[option],
                )
            )

    return missions


def build_recapture_missions(world, policy, planned_commitments, modes):
    missions = []

    for target in world.my_planets:
        fall_turn = world.fall_turn_map.get(target.id)
        if fall_turn is None or fall_turn > DEFENSE_LOOKAHEAD_TURNS:
            continue

        for src in world.my_planets:
            if src.id == target.id:
                continue

            src_available = policy["attack_budget"].get(src.id, 0)
            if src_available < PARTIAL_SOURCE_MIN_SHIPS:
                continue

            seeded = world.best_probe_aim(
                src.id,
                target.id,
                src_available,
                hints=(target.production + DEFENSE_SEND_MARGIN_BASE + 2,),
                min_turn=fall_turn + 1,
                max_turn=fall_turn + RECAPTURE_LOOKAHEAD_TURNS,
            )
            if seeded is None:
                continue
            probe, probe_aim = seeded
            probe_turns = probe_aim[1]

            plan = settle_plan(
                src,
                target,
                src_available,
                probe,
                world,
                planned_commitments,
                modes,
                policy,
                mission="capture",
            )
            if plan is None:
                continue

            angle, turns, _, need, send_pref = plan
            if turns <= fall_turn or turns - fall_turn > RECAPTURE_LOOKAHEAD_TURNS:
                continue

            saved_turns = max(1, world.remaining_steps - turns)
            value = (
                RECAPTURE_PRODUCTION_WEIGHT * target.production * saved_turns
                + RECAPTURE_IMMEDIATE_WEIGHT * max(0, target.ships)
            )
            if world.enemy_planets and nearest_distance_to_set(target.x, target.y, world.enemy_planets) < 22:
                value *= RECAPTURE_FRONTIER_MULT
            value *= RECAPTURE_VALUE_MULT
            score = value / (send_pref + turns * RECAPTURE_COST_TURN_WEIGHT + 1.0)
            if fall_turn <= URGENT_DEFENSE_TURN:
                score *= URGENT_DEFENSE_SCORE_MULT
            else:
                score *= EARLY_DEFENSE_SCORE_MULT
            if is_defense_core_planet(target, world):
                score *= CORE_DEFENSE_SCORE_MULT
                if fall_turn <= CORE_VISIBLE_THREAT_TURN:
                    score *= CORE_URGENT_DEFENSE_SCORE_MULT

            option = ShotOption(
                score=score,
                src_id=src.id,
                target_id=target.id,
                angle=angle,
                turns=turns,
                needed=need,
                send_cap=send_pref,
                mission="recapture",
                anchor_turn=fall_turn,
            )
            missions.append(
                Mission(
                    kind="recapture",
                    score=score,
                    target_id=target.id,
                    turns=turns,
                    options=[option],
                )
            )

    return missions


def build_reinforce_missions(world, policy, planned_commitments, modes, inventory_left_fn):
    if not REINFORCE_ENABLED:
        return []

    missions = []
    if world.remaining_steps < REINFORCE_MIN_FUTURE_TURNS:
        return missions

    for target in world.my_planets:
        fall_turn = world.fall_turn_map.get(target.id)
        if fall_turn is None:
            continue
        if target.production < REINFORCE_MIN_PRODUCTION:
            continue

        hold_until = min(HORIZON, fall_turn + REINFORCE_HOLD_LOOKAHEAD)
        max_arrival_turn = min(fall_turn, REINFORCE_MAX_TRAVEL_TURNS)

        for src in world.my_planets:
            if src.id == target.id:
                continue

            budget = inventory_left_fn(src.id)
            source_cap = min(budget, int(src.ships * REINFORCE_MAX_SOURCE_FRACTION))
            if source_cap < PARTIAL_SOURCE_MIN_SHIPS:
                continue

            seeded = world.best_probe_aim(
                src.id,
                target.id,
                source_cap,
                hints=(target.production + REINFORCE_SAFETY_MARGIN + 2,),
                max_turn=max_arrival_turn,
            )
            if seeded is None:
                continue
            probe, _ = seeded

            plan = settle_reinforce_plan(
                src,
                target,
                source_cap,
                probe,
                world,
                planned_commitments,
                hold_until,
                max_arrival_turn,
            )
            if plan is None:
                continue

            angle, turns, _, need, send_pref = plan
            value = reinforce_value(target, hold_until, world, policy)
            score = value / (send_pref + turns * REINFORCE_COST_TURN_WEIGHT + 1.0)
            if fall_turn <= URGENT_DEFENSE_TURN:
                score *= URGENT_DEFENSE_SCORE_MULT
            else:
                score *= EARLY_DEFENSE_SCORE_MULT
            if is_defense_core_planet(target, world):
                score *= CORE_DEFENSE_SCORE_MULT
                if fall_turn <= CORE_VISIBLE_THREAT_TURN:
                    score *= CORE_URGENT_DEFENSE_SCORE_MULT

            option = ShotOption(
                score=score,
                src_id=src.id,
                target_id=target.id,
                angle=angle,
                turns=turns,
                needed=need,
                send_cap=send_pref,
                mission="reinforce",
                anchor_turn=hold_until,
            )
            missions.append(
                Mission(
                    kind="reinforce",
                    score=score,
                    target_id=target.id,
                    turns=fall_turn,
                    options=[option],
                )
            )

    return missions


def build_crash_exploit_missions(world, policy, planned_commitments, modes):
    if not CRASH_EXPLOIT_ENABLED or not world.is_four_player:
        return []

    missions = []
    for crash in detect_enemy_crashes(world):
        target = world.planet_by_id[crash["target_id"]]
        if target.owner == world.player:
            continue
        desired_arrival = crash["crash_turn"] + CRASH_EXPLOIT_POST_CRASH_DELAY

        for src in world.my_planets:
            src_available = policy["attack_budget"].get(src.id, 0)
            if src_available < PARTIAL_SOURCE_MIN_SHIPS:
                continue

            seeded = world.best_probe_aim(
                src.id,
                target.id,
                src_available,
                hints=(12, int(target.ships) + 1),
                anchor_turn=desired_arrival,
                max_anchor_diff=CRASH_EXPLOIT_ETA_WINDOW,
            )
            if seeded is None:
                continue
            probe, _ = seeded

            plan = settle_plan(
                src,
                target,
                src_available,
                probe,
                world,
                planned_commitments,
                modes,
                policy,
                mission="crash_exploit",
                eval_turn_fn=lambda turns, desired_arrival=desired_arrival: max(turns, desired_arrival),
                anchor_turn=desired_arrival,
                anchor_tolerance=CRASH_EXPLOIT_ETA_WINDOW,
            )
            if plan is None:
                continue

            angle, turns, _, need, send_pref = plan
            if not candidate_time_valid(target, turns, world, LATE_CAPTURE_BUFFER):
                continue
            value = target_value(target, turns, "crash_exploit", world, modes, policy)
            if value <= 0:
                continue

            score = apply_score_modifiers(
                value / (send_pref + turns * SNIPE_COST_TURN_WEIGHT + 1.0),
                target,
                "crash_exploit",
                world,
            )
            option = ShotOption(
                score=score,
                src_id=src.id,
                target_id=target.id,
                angle=angle,
                turns=turns,
                needed=need,
                send_cap=send_pref,
                mission="crash_exploit",
                anchor_turn=desired_arrival,
            )
            missions.append(
                Mission(
                    kind="crash_exploit",
                    score=score,
                    target_id=target.id,
                    turns=turns,
                    options=[option],
                )
            )

    return missions


def heavy_assault_cluster_value(target, world):
    value = target.production
    for other in world.enemy_planets:
        if other.id == target.id:
            continue
        if planet_distance(target, other) <= HEAVY_ASSAULT_CLUSTER_RADIUS:
            value += other.production * HEAVY_ASSAULT_CLUSTER_PROD_WEIGHT
    if initial_owner_of(target, world) == world.player:
        value += target.production
    return value


def early_heavy_assault_profile(world):
    homes = [
        world.initial_by_id.get(home_id) or world.planet_by_id.get(home_id)
        for home_id in PROFILE_HOME_IDS
    ]
    homes = [home for home in homes if home is not None and home.production == 2]
    for home in homes:
        has_medium_p5 = False
        has_cheap_core = False
        for target in world.initial_by_id.values():
            if target.id == home.id or target.owner != -1:
                continue
            if target.production < CORE_PRODUCTION:
                continue
            distance = planet_distance(home, target)
            if (
                target.production >= CORE_PRODUCTION + 1
                and int(target.ships) <= EARLY_HEAVY_P2_CORE_MAX_SHIPS
                and distance <= EARLY_HEAVY_P2_CORE_DIST
            ):
                has_medium_p5 = True
            if (
                int(target.ships) <= EARLY_HEAVY_P2_CHEAP_CORE_SHIPS
                and distance <= EARLY_HEAVY_P2_CHEAP_CORE_DIST
            ):
                has_cheap_core = True
        if has_medium_p5 and not has_cheap_core:
            return True
    return False


def build_heavy_assault_missions(world, policy, planned_commitments, modes, attack_left_fn):
    early_profile = early_heavy_assault_profile(world)
    start_step = EARLY_PROFILE_HEAVY_ASSAULT_START_STEP if early_profile else HEAVY_ASSAULT_START_STEP
    min_total_required = EARLY_PROFILE_HEAVY_ASSAULT_MIN_TOTAL if early_profile else HEAVY_ASSAULT_MIN_TOTAL
    min_source_required = EARLY_PROFILE_HEAVY_ASSAULT_MIN_SOURCE if early_profile else HEAVY_ASSAULT_MIN_SOURCE_SHIPS
    if (
        not HEAVY_ASSAULT_ENABLED
        or world.step < start_step
        or world.is_late
        or len(world.my_planets) < 3
        or not world.enemy_planets
    ):
        return []

    missions = []
    targets = sorted(
        world.enemy_planets,
        key=lambda target: (
            -heavy_assault_cluster_value(target, world),
            initial_owner_of(target, world) != world.player,
            planet_distance(
                target,
                min(world.my_planets, key=lambda src: planet_distance(src, target)),
            ),
            target.id,
        ),
    )[:6]

    for target in targets:
        options = []
        for src in nearest_sources_to_target(target, world.my_planets, HEAVY_ASSAULT_MAX_SOURCES):
            src_available = attack_left_fn(src.id)
            if src_available < min_source_required:
                continue

            send_cap = src_available
            seeded = world.best_probe_aim(
                src.id,
                target.id,
                send_cap,
                hints=(send_cap, max(min_source_required, int(target.ships) + 1)),
            )
            if seeded is None:
                continue
            _, aim = seeded
            angle, turns, _, _ = aim
            if not candidate_time_valid(target, turns, world, LATE_CAPTURE_BUFFER):
                continue
            if opening_filter(target, turns, int(target.ships) + 1, send_cap, world, policy):
                continue
            options.append(
                ShotOption(
                    score=send_cap / max(1.0, turns),
                    src_id=src.id,
                    target_id=target.id,
                    angle=angle,
                    turns=turns,
                    needed=int(target.ships) + 1,
                    send_cap=send_cap,
                    mission="heavy_swarm",
                )
            )

        if len(options) < 2:
            continue

        options.sort(key=lambda option: (-option.send_cap, option.turns, option.src_id))
        top_options = options[:HEAVY_ASSAULT_MAX_SOURCES]
        for size in (3, 2):
            if len(top_options) < size:
                continue
            for combo in combinations(top_options, size):
                turns = [option.turns for option in combo]
                if max(turns) - min(turns) > HEAVY_ASSAULT_ETA_TOLERANCE:
                    continue
                joint_turn = max(turns)
                total_cap = sum(option.send_cap for option in combo)
                if total_cap < min_total_required:
                    continue
                need = world.min_ships_to_own_at(
                    target.id,
                    joint_turn,
                    world.player,
                    planned_commitments=planned_commitments,
                    upper_bound=total_cap,
                )
                if need <= 0 or need > total_cap:
                    continue

                desired_total = min(
                    total_cap,
                    max(need + HEAVY_ASSAULT_OVERKILL, min_total_required),
                )
                cluster_value = heavy_assault_cluster_value(target, world)
                value = target_value(target, joint_turn, "swarm", world, modes, policy)
                value *= max(1.0, cluster_value / max(1.0, target.production))
                score = apply_score_modifiers(
                    value / (desired_total + joint_turn * ATTACK_COST_TURN_WEIGHT + 1.0),
                    target,
                    "swarm",
                    world,
                )
                score *= HEAVY_ASSAULT_SCORE_MULT
                if (
                    world.enemy_prod >= world.my_prod + DEFICIT_HEAVY_ASSAULT_PROD_GAP
                    and world.my_total >= DEFICIT_HEAVY_ASSAULT_MIN_TOTAL_SHIPS
                ):
                    score *= DEFICIT_HEAVY_ASSAULT_SCORE_MULT
                missions.append(
                    Mission(
                        kind="heavy_swarm",
                        score=score,
                        target_id=target.id,
                        turns=joint_turn,
                        options=list(combo),
                        min_total=desired_total,
                    )
                )
                break

    return missions


def build_turtle_breakout_moves(world, policy):
    if (
        world.step < TURTLE_BREAKOUT_START_STEP
        or world.is_late
        or len(world.my_planets) > TURTLE_BREAKOUT_MAX_PLANETS
        or world.my_total < TURTLE_BREAKOUT_MIN_TOTAL
        or world.enemy_prod < world.my_prod + TURTLE_BREAKOUT_PROD_GAP
        or not world.enemy_planets
    ):
        return None

    targets = sorted(
        world.enemy_planets,
        key=lambda target: (
            -heavy_assault_cluster_value(target, world),
            -int(target.production),
            nearest_distance_to_set(target.x, target.y, world.my_planets),
            target.id,
        ),
    )[:4]

    for target in targets:
        options = []
        for src in nearest_sources_to_target(target, world.my_planets, TURTLE_BREAKOUT_MAX_SOURCES):
            keep = max(
                policy["reserve"].get(src.id, 0),
                int(src.ships * (1.0 - TURTLE_BREAKOUT_SOURCE_FRACTION)),
            )
            available = max(0, int(src.ships) - keep)
            if available < TURTLE_BREAKOUT_MIN_SOURCE:
                continue
            aim = world.plan_shot(src.id, target.id, available)
            if aim is None:
                continue
            angle, turns, _, _ = aim
            if not candidate_time_valid(target, turns, world, LATE_CAPTURE_BUFFER):
                continue
            options.append((turns, -available, src, available, angle))

        if len(options) < 2:
            continue
        options.sort()
        selected = options[:TURTLE_BREAKOUT_MAX_SOURCES]
        joint_turn = max(item[0] for item in selected)
        total_available = sum(item[3] for item in selected)
        if total_available < TURTLE_BREAKOUT_MIN_TOTAL_SEND:
            continue
        need = world.min_ships_to_own_at(
            target.id,
            joint_turn,
            world.player,
            upper_bound=total_available,
        )
        desired = min(
            total_available,
            max(TURTLE_BREAKOUT_MIN_TOTAL_SEND, need + TURTLE_BREAKOUT_OVERKILL),
        )
        if need <= 0 or total_available < desired:
            continue

        remaining = desired
        moves = []
        ordered = sorted(selected, key=lambda item: (item[0], item[2].id))
        for idx, (_, _, src, available, _) in enumerate(ordered):
            rest = sum(item[3] for item in ordered[idx + 1 :])
            send = min(available, max(0, remaining - rest))
            if send <= 0:
                continue
            aim = world.plan_shot(src.id, target.id, send)
            if aim is None:
                moves = []
                break
            angle, _, _, _ = aim
            moves.append([src.id, float(angle), int(send)])
            remaining -= send
        if remaining <= 0 and moves:
            return moves

    return None


def build_timeout_fallback_moves(world):
    if world.step < TIMEOUT_FALLBACK_START_STEP or world.is_very_late:
        return []

    targets = list(world.enemy_planets)
    targets.extend(
        planet
        for planet in world.neutral_planets
        if int(planet.production) >= TIMEOUT_FALLBACK_NEUTRAL_MIN_PROD
        and planet.id not in world.comet_ids
    )
    if not targets:
        return []

    def source_left(source):
        keep = int(world.keep_needed_map.get(source.id, 0))
        return max(0, int(source.ships) - keep)

    sources = [
        planet
        for planet in world.my_planets
        if source_left(planet) >= TIMEOUT_FALLBACK_MIN_SOURCE
    ]
    sources.sort(
        key=lambda planet: (
            -source_left(planet),
            -int(planet.production),
            planet.id,
        )
    )

    moves = []
    used = defaultdict(int)
    for source in sources:
        if len(moves) >= TIMEOUT_FALLBACK_MAX_MOVES:
            break
        left = max(0, source_left(source) - used[source.id])
        if left < TIMEOUT_FALLBACK_MIN_SOURCE:
            continue

        ranked_targets = []
        for target in targets:
            if target.id == source.id or target.owner == world.player:
                continue
            if target.owner == -1 and int(target.ships) + 1 > left:
                continue
            distance = planet_distance(source, target)
            value = float(target.production) * 10.0
            if target.owner not in (-1, world.player):
                value += 12.0
            score = value / (1.0 + distance * 0.25 + max(0, int(target.ships)) * 0.12)
            ranked_targets.append((score, target))

        ranked_targets.sort(
            key=lambda item: (
                -item[0],
                -int(item[1].production),
                planet_distance(source, item[1]),
                item[1].id,
            )
        )

        for _, target in ranked_targets[:TIMEOUT_FALLBACK_MAX_TARGETS]:
            left = max(0, source_left(source) - used[source.id])
            if left < TIMEOUT_FALLBACK_MIN_SOURCE:
                break
            if target.owner == -1:
                send = min(
                    left,
                    max(
                        int(target.ships) + 1 + int(target.production),
                        int(left * TIMEOUT_FALLBACK_SEND_RATIO),
                    ),
                )
            else:
                send = min(
                    left,
                    max(TIMEOUT_FALLBACK_MIN_SOURCE, int(left * TIMEOUT_FALLBACK_SEND_RATIO)),
                )
            aim = world.plan_shot(source.id, target.id, send)
            if aim is None:
                continue
            angle, turns, _, _ = aim
            if turns > TIMEOUT_FALLBACK_MAX_TURNS:
                continue
            moves.append([source.id, float(angle), int(send)])
            used[source.id] += send
            break

    return moves


def plan_moves(world, deadline=None):
    world.debug_info = {
        "stage": "start",
        "counts": {},
        "samples": {},
    }

    def debug_set(key, value):
        world.debug_info[key] = value

    def debug_count(key, sample=None):
        counts = world.debug_info.setdefault("counts", {})
        counts[key] = counts.get(key, 0) + 1
        if sample is not None:
            samples = world.debug_info.setdefault("samples", {}).setdefault(key, [])
            if len(samples) < ORBIT_LOG_SAMPLE_LIMIT:
                samples.append(sample)

    def expired():
        return deadline is not None and time.perf_counter() > deadline

    def time_left():
        if deadline is None:
            return 10**9
        return deadline - time.perf_counter()

    def allow_heavy_phase():
        return time_left() > HEAVY_PHASE_MIN_TIME and len(world.planets) <= HEAVY_ROUTE_PLANET_LIMIT

    def allow_optional_phase():
        return time_left() > OPTIONAL_PHASE_MIN_TIME

    opening_meta_moves = timed_call(
        world,
        "strategy.opening_meta",
        build_opening_meta_moves,
        world,
        debug_set=debug_set,
    )
    if opening_meta_moves is not None:
        debug_set("stage", "opening_meta")
        debug_set("opening_return", len(opening_meta_moves))
        return opening_meta_moves

    opening_direct_moves = timed_call(
        world,
        "strategy.opening_direct",
        build_opening_direct_expand_moves,
        world,
    )
    if opening_direct_moves is not None:
        debug_set("stage", "opening_direct")
        debug_set("opening_return", len(opening_direct_moves))
        return opening_direct_moves
    opening_fast_expand_moves = timed_call(
        world,
        "strategy.opening_fast_expand",
        build_opening_fast_expand_moves,
        world,
    )
    if opening_fast_expand_moves is not None:
        debug_set("stage", "opening_fast_expand")
        debug_set("opening_return", len(opening_fast_expand_moves))
        return opening_fast_expand_moves
    opening_anchor_moves = timed_call(
        world,
        "strategy.opening_anchor",
        build_opening_anchor_moves,
        world,
    )
    if opening_anchor_moves is not None:
        debug_set("stage", "opening_anchor")
        debug_set("opening_return", len(opening_anchor_moves))
        return opening_anchor_moves
    opening_priority_moves = timed_call(
        world,
        "strategy.opening_priority",
        build_opening_priority_moves,
        world,
    )
    if opening_priority_moves is not None:
        debug_set("stage", "opening_priority")
        debug_set("opening_return", len(opening_priority_moves))
        return opening_priority_moves
    opening_local_quality_moves = timed_call(
        world,
        "strategy.opening_local_quality",
        build_opening_local_quality_moves,
        world,
    )
    if opening_local_quality_moves is not None:
        debug_set("stage", "opening_local_quality")
        debug_set("opening_return", len(opening_local_quality_moves))
        return opening_local_quality_moves
    opening_mainline_moves = timed_call(
        world,
        "strategy.opening_mainline",
        build_opening_mainline_moves,
        world,
    )
    if opening_mainline_moves is not None:
        debug_set("stage", "opening_mainline")
        debug_set("opening_return", len(opening_mainline_moves))
        return opening_mainline_moves
    opening_heavy_prize_moves = timed_call(
        world,
        "strategy.opening_heavy_prize",
        build_opening_heavy_prize_moves,
        world,
    )
    if opening_heavy_prize_moves is not None:
        debug_set("stage", "opening_heavy_prize")
        debug_set("opening_return", len(opening_heavy_prize_moves))
        return opening_heavy_prize_moves

    modes = timed_call(world, "phase.build_modes", build_modes, world)
    policy = timed_call(world, "phase.build_policy_state", build_policy_state, world, deadline=deadline)
    debug_set("stage", "policy")
    debug_set(
        "budget",
        [
            [
                planet.id,
                int(planet.production),
                int(planet.ships),
                int(policy["reserve"].get(planet.id, 0)),
                int(policy["attack_budget"].get(planet.id, 0)),
                world.fall_turn_map.get(planet.id),
            ]
            for planet in sorted(
                world.my_planets,
                key=lambda item: (
                    -int(policy["attack_budget"].get(item.id, 0)),
                    -int(item.production),
                    item.id,
                ),
            )[:ORBIT_LOG_SAMPLE_LIMIT]
        ],
    )
    policy["predicted_enemy_arrivals"] = (
        timed_call(
            world,
            "phase.predict_enemy_arrivals",
            predict_enemy_arrivals,
            world,
            deadline=deadline,
        )
        if allow_heavy_phase()
        else {}
    )
    planned_commitments = defaultdict(list)
    source_options_by_target = defaultdict(list)
    missions = []
    moves = []
    spent_total = defaultdict(int)

    def source_inventory_left(source_id):
        return world.source_inventory_left(source_id, spent_total)

    def source_attack_left(source_id):
        budget = policy["attack_budget"].get(source_id, 0)
        return max(0, budget - spent_total[source_id])

    def emergency_heavy_source_left(source_id):
        src = world.planet_by_id.get(source_id)
        if src is None:
            return source_attack_left(source_id)
        left = source_attack_left(source_id)
        fall_turn = world.fall_turn_map.get(source_id)
        if (
            fall_turn is not None
            and fall_turn <= DOOMED_EVAC_TURN_LIMIT
            and (src.production >= CORE_PRODUCTION or is_defense_core_planet(src, world))
        ):
            evac_keep = max(1, int(int(src.ships) * BOXED_BREAKOUT_EVAC_KEEP_FRACTION))
            left = max(left, max(0, source_inventory_left(source_id) - evac_keep))
        return left

    def mission_source_left(kind, source_id):
        if kind == "heavy_swarm":
            return emergency_heavy_source_left(source_id)
        return source_attack_left(source_id)

    def append_move(src_id, angle, ships, target_id=None, force=False):
        inferred_target_id = target_id
        if inferred_target_id is None and not force:
            inferred_target_id = first_current_ray_hit(world, src_id, angle)
        elif (
            target_id is not None
            and not force
            and MIDGAME_UNINTENDED_BLOCKER_ENABLED
            and MIDGAME_UNINTENDED_BLOCKER_START_STEP <= world.step <= MIDGAME_UNINTENDED_BLOCKER_TURN_LIMIT
        ):
            first_hit_id = first_current_ray_hit(world, src_id, angle)
            if first_hit_id is not None and first_hit_id != target_id:
                blocker = world.planet_by_id.get(first_hit_id)
                intended = world.planet_by_id.get(target_id)
                if (
                    blocker is not None
                    and intended is not None
                    and blocker.owner != world.player
                    and blocker.production <= MIDGAME_UNINTENDED_BLOCKER_MAX_PROD
                    and intended.production >= blocker.production + MIDGAME_UNINTENDED_BLOCKER_TARGET_PROD_GAP
                ):
                    debug_count(
                        "unintended_blocker",
                        [
                            src_id,
                            target_id,
                            first_hit_id,
                            int(blocker.production),
                            int(intended.production),
                            int(ships),
                        ],
                    )
                    return 0
        if inferred_target_id is not None and not force:
            target = world.planet_by_id.get(inferred_target_id)
            if (
                target is not None
                and world.is_opening
                and target.owner == -1
                and target.production <= 1
                and (
                    int(ships) > int(target.ships) + OPENING_FILL_LOW_PROD_MAX_OVER_NEUTRAL
                    or int(target.ships) >= OPENING_LOW_PROD_HEAVY_TARGET_SHIPS
                    or (
                        low_production_home_profile(world)
                        and int(ships) > OPENING_FILL_LOW_PROD_MAX_NEEDED
                    )
                )
            ):
                return 0
            if target is not None and poor_opening_target(target, world):
                return 0
        send_cap = source_inventory_left(src_id)
        if target_id is not None and not force:
            target = world.planet_by_id.get(target_id)
            source = world.planet_by_id.get(src_id)
            if (
                target is not None
                and source is not None
                and target.owner != world.player
                and is_defense_core_planet(source, world)
                and world.fall_turn_map.get(src_id) is not None
                and world.fall_turn_map.get(src_id) <= SOURCE_LOCK_FALL_TURNS
            ):
                send_cap = min(send_cap, source_attack_left(src_id))
        send = min(int(ships), send_cap)
        if send < 1:
            return 0
        moves.append([src_id, float(angle), int(send)])
        spent_total[src_id] += send
        return send

    def build_timeout_fallback():
        if deadline is None or time.perf_counter() < deadline - TIMEOUT_FALLBACK_NEAR_DEADLINE:
            return []
        if world.step < TIMEOUT_FALLBACK_START_STEP or world.is_very_late:
            return []
        fallback = build_timeout_fallback_moves(world)
        debug_set("timeout_fallback", len(fallback))
        return fallback

    def build_boxed_breakout():
        if (
            world.step < BOXED_BREAKOUT_START_STEP
            or world.is_very_late
            or len(world.my_planets) > BOXED_BREAKOUT_MAX_PLANETS
            or not world.enemy_planets
            or not allow_optional_phase()
        ):
            return 0

        max_enemy_prod = max(
            (
                int(production)
                for owner, production in world.owner_production.items()
                if owner not in (-1, world.player)
            ),
            default=0,
        )
        if max_enemy_prod <= 0:
            return 0

        budget_locked = (
            bool(world.my_planets)
            and all(source_attack_left(planet.id) <= 0 for planet in world.my_planets)
        )
        counts = world.debug_info.get("counts", {})
        planner_deadlocked = (
            counts.get("partial_option", 0) > 0
            and (
                not world.debug_info.get("mission_count", 0)
                or counts.get("multi_eta_fail", 0) > 0
                or counts.get("multi_missing_fail", 0) > 0
            )
        )
        prod_trapped = (
            world.my_prod + BOXED_BREAKOUT_PROD_DEFICIT < max_enemy_prod
            or world.my_prod < max_enemy_prod * BOXED_BREAKOUT_PROD_RATIO
        )
        total_trapped = world.my_total < world.max_enemy_strength * BOXED_BREAKOUT_TOTAL_RATIO
        if not (budget_locked or planner_deadlocked) or not (prod_trapped or total_trapped):
            return 0

        debug_count(
            "boxed_considered",
            [
                int(world.my_prod),
                int(max_enemy_prod),
                int(world.my_total),
                int(world.max_enemy_strength),
                1 if budget_locked else 0,
                1 if planner_deadlocked else 0,
            ],
        )

        chosen_targets = set()
        accepted = 0

        for src in sorted(
            world.my_planets,
            key=lambda planet: (
                -int(planet.ships),
                -int(planet.production),
                planet.id,
            ),
        ):
            if accepted >= BOXED_BREAKOUT_MAX_MOVES or expired():
                break

            inventory_left = source_inventory_left(src.id)
            if inventory_left < BOXED_BREAKOUT_MIN_SOURCE:
                debug_count("boxed_source_small", [src.id, int(src.ships), int(inventory_left)])
                continue

            base_keep = BOXED_BREAKOUT_KEEP_BASE + int(src.production) * BOXED_BREAKOUT_KEEP_PROD_WEIGHT
            fall_turn = world.fall_turn_map.get(src.id)
            if fall_turn is not None and fall_turn <= DOOMED_EVAC_TURN_LIMIT:
                keep = min(
                    base_keep,
                    max(1, int(int(src.ships) * BOXED_BREAKOUT_EVAC_KEEP_FRACTION)),
                )
            else:
                keep = max(base_keep, int(int(src.ships) * BOXED_BREAKOUT_KEEP_FRACTION))
                keep = max(keep, int(policy["reserve"].get(src.id, 0)))
            cap = max(0, inventory_left - keep)
            if cap < BOXED_BREAKOUT_MIN_SOURCE:
                debug_count("boxed_cap_small", [src.id, int(src.ships), int(keep), int(cap), fall_turn])
                continue

            best = None
            for target in sorted(
                world.planets,
                key=lambda planet: (
                    planet.id in chosen_targets,
                    -int(planet.production),
                    int(planet.ships),
                    planet.id,
                ),
            ):
                if expired():
                    return accepted
                if target.id == src.id or target.owner == world.player:
                    continue
                if target.id in world.comet_ids:
                    continue
                if target.owner == -1:
                    if target.production < BOXED_BREAKOUT_NEUTRAL_MIN_PROD:
                        continue
                elif target.production < BOXED_BREAKOUT_HOSTILE_MIN_PROD:
                    continue

                seeded = world.best_probe_aim(
                    src.id,
                    target.id,
                    cap,
                    hints=(cap, int(target.ships) + BOXED_BREAKOUT_MARGIN),
                )
                if seeded is None:
                    debug_count("boxed_no_route", [src.id, target.id, int(target.production), int(target.ships)])
                    continue
                _, rough_aim = seeded
                rough_turns = rough_aim[1]
                if rough_turns > BOXED_BREAKOUT_MAX_TURNS:
                    debug_count("boxed_too_slow", [src.id, target.id, int(rough_turns)])
                    continue
                if not candidate_time_valid(target, rough_turns, world, LATE_CAPTURE_BUFFER):
                    debug_count("boxed_time_invalid", [src.id, target.id, int(rough_turns)])
                    continue

                needed = world.min_ships_to_own_at(
                    target.id,
                    rough_turns,
                    world.player,
                    planned_commitments=planned_commitments,
                    upper_bound=cap,
                )
                near_miss = False
                if needed > cap and target.production >= CORE_PRODUCTION:
                    exact_needed = world.min_ships_to_own_at(
                        target.id,
                        rough_turns,
                        world.player,
                        planned_commitments=planned_commitments,
                    )
                    near_miss = (
                        exact_needed > cap
                        and exact_needed <= cap + BOXED_BREAKOUT_NEAR_MISS_SHIPS
                        and cap >= int(exact_needed * BOXED_BREAKOUT_NEAR_MISS_RATIO)
                    )
                    if near_miss:
                        needed = exact_needed

                if needed <= 0 or (needed > cap and not near_miss):
                    debug_count("boxed_need_fail", [src.id, target.id, int(needed), int(cap)])
                    continue

                if near_miss:
                    fixed_aim = world.plan_shot(src.id, target.id, cap)
                    if fixed_aim is None:
                        debug_count("boxed_no_route", [src.id, target.id, int(target.production), int(target.ships)])
                        continue
                    angle, turns, _, _ = fixed_aim
                    need = needed
                    send = cap
                else:
                    send_guess = min(
                        cap,
                        max(
                            int(needed) + BOXED_BREAKOUT_MARGIN,
                            int(cap * BOXED_BREAKOUT_SEND_RATIO),
                        ),
                    )
                    plan = settle_plan(
                        src,
                        target,
                        cap,
                        send_guess,
                        world,
                        planned_commitments,
                        modes,
                        policy,
                        mission="capture",
                    )
                    if plan is None:
                        debug_count("boxed_settle_none", [src.id, target.id, int(cap), int(send_guess)])
                        continue

                    angle, turns, _, need, send = plan
                    if send < need:
                        debug_count("boxed_send_short", [src.id, target.id, int(send), int(need)])
                        continue
                if not midgame_attack_allowed(src, target, send, turns, world, modes):
                    debug_count("boxed_midgame_block", [src.id, target.id, int(send), int(turns)])
                    continue

                value = target_value(target, turns, "capture", world, modes, policy)
                if value <= 0:
                    debug_count("boxed_value_zero", [src.id, target.id, int(target.production)])
                    continue
                if target.owner not in (-1, world.player):
                    value *= 1.08
                if target.id in chosen_targets:
                    value *= 0.82
                if near_miss:
                    value *= BOXED_BREAKOUT_NEAR_MISS_SCORE_MULT
                score = apply_score_modifiers(
                    value / (send + turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                    target,
                    "capture",
                    world,
                )
                candidate = (score, target, angle, turns, need, send, near_miss)
                if best is None or candidate[0] > best[0]:
                    best = candidate

            if best is None:
                debug_count("boxed_no_target", [src.id, int(src.ships), int(cap)])
                continue

            _, target, angle, turns, need, send, near_miss = best
            actual = append_move(src.id, angle, send, target.id, force=True)
            if actual < need and not near_miss:
                debug_count("boxed_append_short", [src.id, target.id, int(actual), int(need)])
                continue
            if actual >= need:
                planned_commitments[target.id].append((turns, world.player, int(actual)))
            chosen_targets.add(target.id)
            accepted += 1
            if near_miss:
                debug_count("boxed_near_miss_accept", [src.id, target.id, int(actual), int(need), int(turns)])
            else:
                debug_count("boxed_accept", [src.id, target.id, int(actual), int(turns)])

        return accepted

    def build_soften_fallback():
        if (
            not SOFTEN_FALLBACK_ENABLED
            or world.is_very_late
            or world.step < SOFTEN_FALLBACK_MIN_STEP
            or world.step > SOFTEN_FALLBACK_MAX_STEP
            or len(world.my_planets) > SOFTEN_FALLBACK_MAX_PLANETS
            or not allow_optional_phase()
        ):
            return 0
        counts = world.debug_info.get("counts", {})
        if not (
            counts.get("partial_option", 0) > 0
            or counts.get("multi_pair_eta_fail", 0) > 0
            or counts.get("multi_eta_fail", 0) > 0
        ):
            return 0

        def followup_feasible(src, target, turns, send):
            follow_turn = min(
                SOFTEN_FALLBACK_FOLLOWUP_MAX_TURNS,
                int(math.ceil(turns)) + SOFTEN_FALLBACK_FOLLOWUP_WINDOW,
            )
            extra = [(turns, world.player, int(send))]
            owner_after, ships_after = world.projected_state(
                target.id,
                turns,
                extra_arrivals=extra,
            )
            if owner_after == world.player:
                if target.production >= CORE_PRODUCTION:
                    arrival_turn = max(1, int(math.ceil(turns)))
                    hold_turn = min(
                        SOFTEN_FALLBACK_FOLLOWUP_MAX_TURNS,
                        arrival_turn + SOFTEN_FALLBACK_CORE_HOLD_WINDOW,
                    )
                    timeline = world.projected_timeline(
                        target.id,
                        hold_turn,
                        extra_arrivals=extra,
                    )
                    holds_core = all(
                        timeline["owner_at"].get(turn) == world.player
                        for turn in range(arrival_turn, hold_turn + 1)
                    )
                    if not holds_core:
                        return False, int(target.ships) + 1, 0
                    min_garrison = min(
                        timeline["ships_at"].get(turn, 0.0)
                        for turn in range(arrival_turn, hold_turn + 1)
                        if timeline["owner_at"].get(turn) == world.player
                    )
                    required_garrison = (
                        SOFTEN_FALLBACK_CORE_MIN_HOLD_SHIPS
                        + int(target.production)
                    )
                    if min_garrison < required_garrison:
                        return False, required_garrison, int(min_garrison)
                return True, 0, 0

            follow_cap = 0
            for ally in world.my_planets:
                left = source_attack_left(ally.id)
                if ally.id == src.id:
                    left = max(0, left - int(send))
                future_cap = max(0, int(left) + int(ally.production) * SOFTEN_FALLBACK_FOLLOWUP_WINDOW)
                if future_cap < SOFTEN_FALLBACK_MIN_SEND:
                    continue
                if planet_distance(ally, target) > SOFTEN_FALLBACK_FOLLOWUP_MAX_DIST:
                    continue
                route_ships = max(1, min(future_cap, int(math.ceil(ships_after)) + 1))
                aim = world.plan_shot(ally.id, target.id, route_ships)
                if aim is None:
                    continue
                _, ally_turns, _, _ = aim
                if ally_turns <= follow_turn:
                    follow_cap += future_cap

            if follow_cap <= 0:
                return False, 0, 0
            need = world.min_ships_to_own_by(
                target.id,
                follow_turn,
                world.player,
                arrival_turn=follow_turn,
                extra_arrivals=extra,
                upper_bound=follow_cap,
            )
            return (
                need > 0
                and follow_cap >= int(math.ceil(need * SOFTEN_FALLBACK_FOLLOWUP_MIN_CAP_RATIO)),
                int(need),
                int(follow_cap),
            )

        best = None
        for src in sorted(
            world.my_planets,
            key=lambda planet: (-source_attack_left(planet.id), -int(planet.production), planet.id),
        ):
            left = source_attack_left(src.id)
            if left < SOFTEN_FALLBACK_MIN_SEND:
                continue
            for target in world.neutral_planets:
                if target.id == src.id or target.id in world.comet_ids:
                    continue
                if my_incoming_ships_to(target, world) > 0:
                    continue
                if target.production < SOFTEN_FALLBACK_MIN_PROD:
                    continue
                if not (SOFTEN_FALLBACK_MIN_SHIPS <= int(target.ships) <= SOFTEN_FALLBACK_MAX_SHIPS):
                    continue

                target_dist = planet_distance(src, target)
                if target_dist > SOFTEN_FALLBACK_MAX_DIST:
                    continue
                enemy_dist = nearest_distance_to_set(target.x, target.y, world.enemy_planets)
                if enemy_dist + SOFTEN_FALLBACK_RACE_ALLOWANCE < target_dist:
                    continue

                send = min(left, max(SOFTEN_FALLBACK_MIN_SEND, int(left * SOFTEN_FALLBACK_SEND_RATIO)))
                damage_ratio = send / max(1.0, float(target.ships))
                min_damage = (
                    SOFTEN_FALLBACK_HIGH_PROD_DAMAGE_RATIO
                    if target.production >= CORE_PRODUCTION + 1
                    else SOFTEN_FALLBACK_MIN_DAMAGE_RATIO
                )
                if damage_ratio < min_damage:
                    continue
                if send >= world.min_ships_to_own_at(target.id, SOFTEN_FALLBACK_MAX_TURNS, world.player):
                    continue

                aim = world.plan_shot(src.id, target.id, send)
                if aim is None:
                    debug_count("soften_no_route", [src.id, target.id, int(send)])
                    continue
                angle, turns, _, _ = aim
                if turns > SOFTEN_FALLBACK_MAX_TURNS:
                    debug_count("soften_too_slow", [src.id, target.id, int(turns)])
                    continue
                can_follow, follow_need, follow_cap = followup_feasible(src, target, turns, send)
                if not can_follow:
                    debug_count("soften_follow_fail", [src.id, target.id, int(send), int(follow_need), int(follow_cap)])
                    continue

                value = target.production * max(1, world.remaining_steps - turns)
                race_mult = 1.0 if enemy_dist >= target_dist else 0.72
                follow_mult = min(1.2, max(0.65, follow_cap / max(1.0, follow_need + send)))
                score = (
                    value
                    * damage_ratio
                    * race_mult
                    * follow_mult
                    / (send + turns * ATTACK_COST_TURN_WEIGHT + 1.0)
                )
                candidate = (score, src, target, angle, turns, send, damage_ratio, follow_need, follow_cap)
                if best is None or candidate[0] > best[0]:
                    best = candidate

        if best is None:
            debug_count("soften_no_target")
            return 0

        _, src, target, angle, turns, send, damage_ratio, follow_need, follow_cap = best
        actual = append_move(src.id, angle, send, target.id, force=True)
        if actual <= 0:
            debug_count("soften_append_short", [src.id, target.id, int(send)])
            return 0
        debug_count(
            "soften_accept",
            [
                src.id,
                target.id,
                int(actual),
                int(turns),
                round(float(damage_ratio), 3),
                int(follow_need),
                int(follow_cap),
            ],
        )
        return 1

    def evacuate_departing_comets():
        evacuated = 0
        for src in sorted(
            world.my_planets,
            key=lambda planet: (world.comet_life(planet.id), planet.id),
        ):
            if src.id not in world.comet_ids:
                continue
            if world.comet_life(src.id) > COMET_EVAC_LIFE_TURNS:
                continue
            available = source_inventory_left(src.id)
            if available < COMET_EVAC_MIN_SHIPS:
                continue

            best = None
            fallback_targets = [
                target
                for target in world.my_planets
                if target.id != src.id and target.id not in world.comet_ids
            ]
            fallback_targets.sort(
                key=lambda target: (
                    planet_distance(src, target),
                    -int(target.production),
                    target.id,
                )
            )
            for target in fallback_targets:
                aim = world.plan_shot(src.id, target.id, available)
                if aim is None:
                    continue
                angle, turns, _, _ = aim
                best = (0, target, angle, turns)
                break

            if best is None:
                attack_targets = [
                    target
                    for target in world.planets
                    if target.id != src.id and target.id not in world.comet_ids
                ]
                attack_targets.sort(
                    key=lambda target: (
                        target.owner == world.player,
                        -target_value(target, 1, "capture", world, modes, policy),
                        planet_distance(src, target),
                        target.id,
                    )
                )
                for target in attack_targets[:8]:
                    aim = world.plan_shot(src.id, target.id, available)
                    if aim is None:
                        continue
                    angle, turns, _, _ = aim
                    if target.owner != world.player and not candidate_time_valid(target, turns, world, LATE_CAPTURE_BUFFER):
                        continue
                    best = (1, target, angle, turns)
                    break

            if best is None:
                debug_count("comet_evac_no_route", [src.id, int(available), int(world.comet_life(src.id))])
                continue

            _, target, angle, turns = best
            actual = append_move(src.id, angle, available, target.id, force=True)
            if actual > 0:
                evacuated += 1
                debug_count(
                    "comet_evac",
                    [src.id, target.id, int(actual), int(world.comet_life(src.id)), int(turns)],
                )
        return evacuated

    def finalize_moves():
        evacuate_departing_comets()
        forced_stage = None
        if not moves and build_boxed_breakout():
            forced_stage = "boxed_breakout"
            debug_set("stage", forced_stage)
        if not moves and build_soften_fallback():
            forced_stage = "soften_fallback"
            debug_set("stage", forced_stage)
        final_moves = []
        used_final = defaultdict(int)
        for src_id, angle, ships in moves:
            source = world.planet_by_id[src_id]
            max_allowed = int(source.ships) - used_final[src_id]
            send = min(int(ships), max_allowed)
            if send >= 1:
                final_moves.append([src_id, float(angle), int(send)])
                used_final[src_id] += send
        if final_moves:
            if forced_stage is None:
                debug_set("stage", "final")
            debug_set("final_return", len(final_moves))
            return final_moves
        fallback = build_timeout_fallback()
        if fallback:
            debug_set("stage", "timeout_fallback")
            return fallback
        debug_set("stage", "empty")
        debug_set("final_return", 0)
        return []

    def compute_live_doomed():
        doomed = set()
        for planet in world.my_planets:
            status = world.hold_status(
                planet.id,
                planned_commitments=planned_commitments,
                horizon=DOOMED_EVAC_TURN_LIMIT,
            )
            if (
                not status["holds_full"]
                and status["fall_turn"] is not None
                and status["fall_turn"] <= DOOMED_EVAC_TURN_LIMIT
                and source_inventory_left(planet.id) >= DOOMED_MIN_SHIPS
            ):
                doomed.add(planet.id)
        return doomed

    def time_filters_pass(target, turns, needed, src_cap):
        if not candidate_time_valid(target, turns, world, VERY_LATE_CAPTURE_BUFFER if world.is_very_late else LATE_CAPTURE_BUFFER):
            return False
        if opening_filter(target, turns, needed, src_cap, world, policy):
            return False
        return True

    if OPENING_COMMITTED_ENABLED and world.step <= OPENING_COMMIT_TURN_LIMIT:
        committed_target_seen = False
        for src in world.my_planets:
            if expired():
                return finalize_moves()
            if src.production > OPENING_COMMIT_HOME_PROD_MAX:
                continue
            target = opening_committed_target(src, world)
            if target is None:
                continue
            committed_target_seen = True
            src_available = source_inventory_left(src.id)
            max_send = max(0, src_available - OPENING_COMMIT_MIN_RESERVE)
            if max_send <= 0:
                return finalize_moves()
            seeded = world.best_probe_aim(
                src.id,
                target.id,
                max_send,
                hints=(int(target.ships) + OPENING_COMMIT_MARGIN,),
            )
            if seeded is None:
                return finalize_moves()
            probe, rough_aim = seeded
            rough_turns = rough_aim[1]
            needed = world.min_ships_to_own_at(
                target.id,
                rough_turns,
                world.player,
                planned_commitments=planned_commitments,
                upper_bound=max_send,
            )
            required = max(needed, int(target.ships) + OPENING_COMMIT_MARGIN)
            if needed <= 0 or required > max_send:
                return finalize_moves()
            plan = settle_plan(
                src,
                target,
                max_send,
                required,
                world,
                planned_commitments,
                modes,
                policy,
                mission="capture",
            )
            if plan is None:
                return finalize_moves()
            angle, turns, _, need, send = plan
            send = max(send, required)
            if send > max_send:
                return finalize_moves()
            actual = append_move(src.id, angle, send, target.id)
            if actual >= need:
                planned_commitments[target.id].append((turns, world.player, int(actual)))
                return finalize_moves()
        if committed_target_seen:
            return finalize_moves()

    if LOCAL_OPENING_ENABLED and world.step <= 6:
        for src in world.my_planets:
            if expired():
                return finalize_moves()
            src_available = source_attack_left(src.id)
            if src_available <= 0:
                continue

            local_targets = [
                target
                for target in world.neutral_planets
                if target.production >= 2
                and int(target.ships) <= 10
                and planet_distance(src, target) <= 18.0
            ]
            local_targets.sort(
                key=lambda target: (
                    -int(target.production),
                    int(target.ships),
                    planet_distance(src, target),
                    target.id,
                )
            )

            for target in local_targets:
                seeded = world.best_probe_aim(
                    src.id,
                    target.id,
                    src_available,
                    hints=(int(target.ships) + 1,),
                )
                if seeded is None:
                    continue
                probe, rough_aim = seeded
                rough_turns = rough_aim[1]
                if not candidate_time_valid(target, rough_turns, world, LATE_CAPTURE_BUFFER):
                    continue
                needed = world.min_ships_to_own_at(
                    target.id,
                    rough_turns,
                    world.player,
                    planned_commitments=planned_commitments,
                    upper_bound=src_available,
                )
                if needed <= 0 or needed > src_available:
                    continue
                plan = settle_plan(
                    src,
                    target,
                    src_available,
                    max(probe, needed),
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="capture",
                )
                if plan is None:
                    continue
                angle, turns, _, need, send = plan
                if send < need:
                    continue
                actual = append_move(src.id, angle, send, target.id)
                if actual >= need:
                    planned_commitments[target.id].append((turns, world.player, int(actual)))
                    break

    if allow_heavy_phase():
        missions.extend(
            build_reinforce_missions(
                world,
                policy,
                planned_commitments,
                modes,
                source_attack_left,
            )
        )
    missions.extend(build_rescue_missions(world, policy, planned_commitments, modes))
    missions.extend(build_recapture_missions(world, policy, planned_commitments, modes))

    # Only build candidates after solving an intercept so timing decisions come
    # from a real route.
    for src in world.my_planets:
        if expired():
            debug_count("expired_source_loop")
            return finalize_moves()
        src_available = source_attack_left(src.id)
        if src_available <= 0:
            debug_count("source_no_budget", [src.id, int(src.ships), int(policy["reserve"].get(src.id, 0))])
            continue

        for target in world.planets:
            if expired():
                debug_count("expired_target_loop")
                return finalize_moves()
            if target.id == src.id or target.owner == world.player:
                continue
            debug_count("candidate_pair")
            if reserve_source_for_local_prize(src, target, world):
                debug_count("reserve_local_prize", [src.id, target.id, int(target.production), int(target.ships)])
                continue

            seeded = world.best_probe_aim(
                src.id,
                target.id,
                src_available,
                hints=(int(target.ships) + 1,),
            )
            if seeded is None:
                debug_count("no_route", [src.id, target.id, int(target.production), int(target.ships)])
                continue
            _, rough_aim = seeded

            rough_turns = rough_aim[1]
            if not candidate_time_valid(
                target,
                rough_turns,
                world,
                VERY_LATE_CAPTURE_BUFFER if world.is_very_late else LATE_CAPTURE_BUFFER,
            ):
                debug_count("time_invalid", [src.id, target.id, int(target.production), int(rough_turns)])
                continue

            global_needed = world.min_ships_to_own_at(
                target.id,
                rough_turns,
                world.player,
                planned_commitments=planned_commitments,
            )
            if global_needed <= 0:
                debug_count("no_need", [src.id, target.id, int(target.production), int(rough_turns)])
                continue
            if opening_filter(target, rough_turns, global_needed, src_available, world, policy):
                debug_count("opening_filter", [src.id, target.id, int(target.production), int(global_needed), int(src_available)])
                continue

            partial_send_cap = min(
                src_available,
                preferred_send(
                    target,
                    global_needed,
                    rough_turns,
                    src_available,
                    world,
                    modes,
                    policy,
                ),
            )
            if partial_send_cap >= PARTIAL_SOURCE_MIN_SHIPS:
                partial_seed = world.best_probe_aim(
                    src.id,
                    target.id,
                    partial_send_cap,
                    hints=(partial_send_cap, global_needed, int(target.ships) + 1),
                )
                if partial_seed is not None:
                    _, partial_aim = partial_seed
                    p_angle, p_turns, _, _ = partial_aim
                    if time_filters_pass(target, p_turns, global_needed, src_available):
                        partial_value = target_value(target, p_turns, "swarm", world, modes, policy)
                        if partial_value > 0:
                            partial_score = apply_score_modifiers(
                                partial_value / (partial_send_cap + p_turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                                target,
                                "swarm",
                                world,
                            )
                            source_options_by_target[target.id].append(
                                ShotOption(
                                    score=partial_score,
                                    src_id=src.id,
                                    target_id=target.id,
                                    angle=p_angle,
                                    turns=p_turns,
                                    needed=global_needed,
                                    send_cap=partial_send_cap,
                                    mission="swarm",
                                )
                            )
                            debug_count("partial_option", [src.id, target.id, int(target.production), int(partial_send_cap), int(p_turns)])

            if global_needed <= src_available:
                send_guess = preferred_send(
                    target,
                    global_needed,
                    rough_turns,
                    src_available,
                    world,
                    modes,
                    policy,
                )
                plan = settle_plan(
                    src,
                    target,
                    src_available,
                    send_guess,
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="capture",
                )
                if plan is None:
                    debug_count("settle_none", [src.id, target.id, int(target.production), int(global_needed), int(src_available)])
                    continue

                angle, turns, _, needed, send_cap = plan
                if not time_filters_pass(target, turns, needed, src_available):
                    debug_count("time_filter_fail", [src.id, target.id, int(target.production), int(turns), int(needed)])
                    continue
                if send_cap < 1:
                    debug_count("send_cap_zero", [src.id, target.id, int(target.production)])
                    continue
                if not midgame_attack_allowed(src, target, send_cap, turns, world, modes):
                    debug_count("midgame_block", [src.id, target.id, int(target.production), int(send_cap), int(turns)])
                    continue

                value = target_value(target, turns, "capture", world, modes, policy)
                if value <= 0:
                    debug_count("nonpositive_value", [src.id, target.id, int(target.production), int(turns)])
                    continue

                score = apply_score_modifiers(
                    value / (send_cap + turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                    target,
                    "capture",
                    world,
                )

                option = ShotOption(
                    score=score,
                    src_id=src.id,
                    target_id=target.id,
                    angle=angle,
                    turns=turns,
                    needed=needed,
                    send_cap=send_cap,
                    mission="capture",
                )

                if send_cap >= needed:
                    debug_count("single_mission", [src.id, target.id, int(target.production), int(send_cap), int(turns)])
                    missions.append(
                        Mission(
                            kind="single",
                            score=score,
                            target_id=target.id,
                            turns=turns,
                            options=[option],
                        )
                    )

            snipe = build_snipe_mission(src, target, src_available, world, planned_commitments, modes, policy)
            if snipe is not None:
                missions.append(snipe)

    # Allow small synchronized two-source finishes when one source is not
    # enough on its own.
    for target_id, options in source_options_by_target.items():
        if expired():
            return finalize_moves()
        if len(options) < 2:
            continue

        target = world.planet_by_id[target_id]
        top_options = sorted(options, key=lambda item: -item.score)[:MULTI_SOURCE_TOP_K]
        for i in range(len(top_options)):
            for j in range(i + 1, len(top_options)):
                first = top_options[i]
                second = top_options[j]
                if first.src_id == second.src_id:
                    debug_count("multi_pair_same_source", [target_id, first.src_id])
                    continue
                pair_tol = swarm_eta_tolerance((first, second), target, world)
                if abs(first.turns - second.turns) > pair_tol:
                    debug_count("multi_pair_eta_fail", [target_id, int(abs(first.turns - second.turns)), int(pair_tol)])
                    continue

                joint_turn = max(first.turns, second.turns)
                total_cap = first.send_cap + second.send_cap
                need = world.min_ships_to_own_at(
                    target_id,
                    joint_turn,
                    world.player,
                    planned_commitments=planned_commitments,
                    upper_bound=total_cap,
                )
                if need <= 0:
                    debug_count("multi_pair_no_need", [target_id, int(joint_turn), int(total_cap)])
                    continue
                if first.send_cap >= need or second.send_cap >= need:
                    debug_count("multi_pair_single_covers", [target_id, int(need), int(first.send_cap), int(second.send_cap)])
                    continue
                if total_cap < need:
                    debug_count("multi_pair_total_fail", [target_id, int(need), int(total_cap)])
                    continue

                value = target_value(target, joint_turn, "swarm", world, modes, policy)
                if value <= 0:
                    debug_count("multi_pair_value_fail", [target_id, int(joint_turn)])
                    continue

                pair_score = apply_score_modifiers(
                    value / (need + joint_turn * ATTACK_COST_TURN_WEIGHT + 1.0),
                    target,
                    "swarm",
                    world,
                )
                pair_score *= MULTI_SOURCE_PLAN_PENALTY
                missions.append(
                    Mission(
                        kind="swarm",
                        score=pair_score,
                        target_id=target_id,
                        turns=joint_turn,
                        options=[first, second],
                    )
                )

        if (
            THREE_SOURCE_SWARM_ENABLED
            and allow_heavy_phase()
            and target.owner not in (-1, world.player)
            and int(target.ships) >= THREE_SOURCE_MIN_TARGET_SHIPS
            and len(top_options) >= 3
        ):
            for i in range(len(top_options)):
                for j in range(i + 1, len(top_options)):
                    for k in range(j + 1, len(top_options)):
                        if expired():
                            return finalize_moves()
                        trio = [top_options[i], top_options[j], top_options[k]]
                        if len({option.src_id for option in trio}) < 3:
                            debug_count("multi_trio_same_source", [target_id])
                            continue
                        trio_tol = swarm_eta_tolerance(tuple(trio), target, world)
                        turns = [option.turns for option in trio]
                        if max(turns) - min(turns) > trio_tol:
                            debug_count("multi_trio_eta_fail", [target_id, int(max(turns) - min(turns)), int(trio_tol)])
                            continue

                        joint_turn = max(turns)
                        total_cap = sum(option.send_cap for option in trio)
                        need = world.min_ships_to_own_at(
                            target_id,
                            joint_turn,
                            world.player,
                            planned_commitments=planned_commitments,
                            upper_bound=total_cap,
                        )
                        if need <= 0:
                            debug_count("multi_trio_no_need", [target_id, int(joint_turn), int(total_cap)])
                            continue
                        if total_cap < need:
                            debug_count("multi_trio_total_fail", [target_id, int(need), int(total_cap)])
                            continue
                        if any(
                            trio[a].send_cap + trio[b].send_cap >= need
                            for a in range(3)
                            for b in range(a + 1, 3)
                        ):
                            debug_count("multi_trio_pair_covers", [target_id, int(need), int(total_cap)])
                            continue

                        value = target_value(target, joint_turn, "swarm", world, modes, policy)
                        if value <= 0:
                            debug_count("multi_trio_value_fail", [target_id, int(joint_turn)])
                            continue

                        trio_score = apply_score_modifiers(
                            value / (need + joint_turn * ATTACK_COST_TURN_WEIGHT + 1.0),
                            target,
                            "swarm",
                            world,
                        )
                        trio_score *= THREE_SOURCE_PLAN_PENALTY
                        missions.append(
                            Mission(
                                kind="swarm",
                                score=trio_score,
                                target_id=target_id,
                                turns=joint_turn,
                                options=trio,
                            )
                        )

        if (
            ANCHORED_SWARM_ENABLED
            and allow_heavy_phase()
            and len(options) >= 2
            and anchored_swarm_worth_trying(target, world, modes)
        ):
            unique_options = []
            seen_sources = set()
            for option in sorted(options, key=lambda item: (-item.score, item.turns, item.src_id)):
                if option.src_id in seen_sources:
                    continue
                if source_attack_left(option.src_id) < PARTIAL_SOURCE_MIN_SHIPS:
                    continue
                unique_options.append(option)
                seen_sources.add(option.src_id)
                if len(unique_options) >= ANCHORED_SWARM_MAX_SOURCES:
                    break

            if len(unique_options) >= 2:
                option_turns = sorted(option.turns for option in unique_options)
                median_turn = option_turns[len(option_turns) // 2]
                anchor_turns = sorted(
                    {option.turns for option in unique_options},
                    key=lambda turn: (abs(turn - median_turn), turn),
                )[:ANCHORED_SWARM_MAX_ANCHORS]

                for anchor_turn in anchor_turns:
                    if expired():
                        return finalize_moves()
                    anchor_tol = (
                        swarm_eta_tolerance(unique_options, target, world)
                        + ANCHORED_SWARM_EXTRA_TOLERANCE
                    )
                    anchored_options = []
                    for option in unique_options:
                        left = source_attack_left(option.src_id)
                        if left < PARTIAL_SOURCE_MIN_SHIPS:
                            continue
                        src = world.planet_by_id[option.src_id]
                        seeded = world.best_probe_aim(
                            src.id,
                            target.id,
                            left,
                            hints=(option.send_cap, option.needed, int(target.ships) + 1),
                            anchor_turn=anchor_turn,
                            max_anchor_diff=anchor_tol,
                        )
                        if seeded is None:
                            debug_count("anchored_no_aim", [target_id, option.src_id, int(anchor_turn)])
                            continue
                        send, aim = seeded
                        angle, turns, _, _ = aim
                        if send < PARTIAL_SOURCE_MIN_SHIPS:
                            debug_count("anchored_small_send", [target_id, option.src_id, int(send)])
                            continue
                        if not time_filters_pass(target, turns, option.needed, left):
                            debug_count("anchored_time_filter", [target_id, option.src_id, int(turns)])
                            continue
                        value = target_value(target, turns, "swarm", world, modes, policy)
                        if value <= 0:
                            debug_count("anchored_value_fail", [target_id, option.src_id, int(turns)])
                            continue
                        score = apply_score_modifiers(
                            value / (send + turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                            target,
                            "swarm",
                            world,
                        )
                        anchored_options.append(
                            ShotOption(
                                score=score,
                                src_id=option.src_id,
                                target_id=target_id,
                                angle=angle,
                                turns=turns,
                                needed=option.needed,
                                send_cap=send,
                                mission="anchored_swarm",
                                anchor_turn=anchor_turn,
                            )
                        )

                    anchored_options.sort(key=lambda item: (-item.score, item.turns, item.src_id))

                    def add_anchored_combo(combo):
                        joint_turn = max(option.turns for option in combo)
                        total_send = sum(option.send_cap for option in combo)
                        need = world.min_ships_to_own_at(
                            target_id,
                            joint_turn,
                            world.player,
                            planned_commitments=planned_commitments,
                            upper_bound=total_send,
                        )
                        if need <= 0:
                            debug_count("anchored_no_need", [target_id, int(joint_turn), int(total_send)])
                            return
                        if total_send < need:
                            debug_count("anchored_total_fail", [target_id, int(need), int(total_send)])
                            return
                        overcommit_limit = max(
                            int(need * ANCHORED_SWARM_MAX_OVERCOMMIT_RATIO),
                            int(need + ANCHORED_SWARM_MAX_OVERCOMMIT_FLAT),
                        )
                        if total_send > overcommit_limit:
                            debug_count("anchored_overcommit", [target_id, int(need), int(total_send)])
                            return
                        owner_after, _ = world.projected_state(
                            target_id,
                            joint_turn,
                            planned_commitments=planned_commitments,
                            extra_arrivals=[
                                (option.turns, world.player, option.send_cap)
                                for option in combo
                            ],
                        )
                        if owner_after != world.player:
                            debug_count("anchored_owner_fail", [target_id, int(joint_turn), int(total_send)])
                            return
                        value = target_value(target, joint_turn, "swarm", world, modes, policy)
                        if value <= 0:
                            debug_count("anchored_combo_value_fail", [target_id, int(joint_turn)])
                            return
                        score = apply_score_modifiers(
                            value / (total_send + joint_turn * ATTACK_COST_TURN_WEIGHT + 1.0),
                            target,
                            "swarm",
                            world,
                        )
                        score *= ANCHORED_SWARM_SCORE_MULT
                        debug_count(
                            "anchored_swarm_mission",
                            [target_id, int(joint_turn), int(total_send), int(need), len(combo)],
                        )
                        missions.append(
                            Mission(
                                kind="anchored_swarm",
                                score=score,
                                target_id=target_id,
                                turns=joint_turn,
                                options=list(combo),
                                min_total=total_send,
                            )
                        )

                    for i in range(len(anchored_options)):
                        for j in range(i + 1, len(anchored_options)):
                            add_anchored_combo((anchored_options[i], anchored_options[j]))

                    if len(anchored_options) >= 3 and target.owner not in (-1, world.player):
                        for i in range(len(anchored_options)):
                            for j in range(i + 1, len(anchored_options)):
                                for k in range(j + 1, len(anchored_options)):
                                    add_anchored_combo(
                                        (
                                            anchored_options[i],
                                            anchored_options[j],
                                            anchored_options[k],
                                        )
                                    )

    if allow_heavy_phase():
        missions.extend(build_crash_exploit_missions(world, policy, planned_commitments, modes))
        missions.extend(
            build_heavy_assault_missions(
                world,
                policy,
                planned_commitments,
                modes,
                emergency_heavy_source_left,
            )
        )

    missions.sort(key=lambda item: -item.score)
    debug_set("mission_count", len(missions))
    debug_set(
        "top_missions",
        [
            [
                mission.kind,
                mission.target_id,
                round(float(mission.score), 4),
                int(mission.turns),
                len(mission.options),
            ]
            for mission in missions[:ORBIT_LOG_SAMPLE_LIMIT]
        ],
    )

    # Update commitments after every accepted launch so later plans see the
    # timing that is already spoken for.
    for mission in missions:
        if expired():
            debug_count("expired_mission_loop")
            return finalize_moves()
        target = world.planet_by_id[mission.target_id]

        if mission.kind in ("single", "snipe", "rescue", "recapture", "reinforce", "crash_exploit"):
            option = mission.options[0]
            src = world.planet_by_id[option.src_id]
            if mission.kind == "reinforce":
                left = min(
                    source_attack_left(option.src_id),
                    int(src.ships * REINFORCE_MAX_SOURCE_FRACTION),
                )
            else:
                left = source_attack_left(option.src_id)
            if left <= 0:
                debug_count("mission_source_empty", [mission.kind, option.src_id, mission.target_id])
                continue

            if mission.kind == "reinforce":
                plan = settle_reinforce_plan(
                    src,
                    target,
                    left,
                    min(left, option.send_cap),
                    world,
                    planned_commitments,
                    option.anchor_turn,
                    mission.turns,
                )
            elif mission.kind == "rescue":
                plan = settle_plan(
                    src,
                    target,
                    left,
                    min(left, option.send_cap),
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="rescue",
                    eval_turn_fn=lambda _turns, hold_turn=mission.turns: hold_turn,
                    anchor_turn=option.anchor_turn,
                )
            elif mission.kind == "snipe":
                plan = settle_plan(
                    src,
                    target,
                    left,
                    min(left, option.send_cap),
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="snipe",
                    eval_turn_fn=lambda turns, enemy_eta=option.anchor_turn: max(turns, enemy_eta),
                    anchor_turn=option.anchor_turn,
                )
            elif mission.kind == "crash_exploit":
                plan = settle_plan(
                    src,
                    target,
                    left,
                    min(left, option.send_cap),
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="crash_exploit",
                    eval_turn_fn=lambda turns, desired_arrival=option.anchor_turn: max(turns, desired_arrival),
                    anchor_turn=option.anchor_turn,
                    anchor_tolerance=CRASH_EXPLOIT_ETA_WINDOW,
                )
            else:
                plan = settle_plan(
                    src,
                    target,
                    left,
                    min(left, option.send_cap),
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="capture",
                )

            if plan is None:
                debug_count("mission_settle_none", [mission.kind, option.src_id, mission.target_id, int(left)])
                continue
            angle, turns, _, need, send = plan
            if send < need or need > left:
                debug_count("mission_need_fail", [mission.kind, option.src_id, mission.target_id, int(need), int(left)])
                continue
            if (
                mission.kind in ("single", "crash_exploit")
                and not midgame_attack_allowed(src, target, send, turns, world, modes)
            ):
                debug_count("mission_midgame_block", [mission.kind, option.src_id, mission.target_id, int(send), int(turns)])
                continue

            sent = append_move(option.src_id, angle, send, target.id)
            if sent < need:
                debug_count("append_short", [mission.kind, option.src_id, mission.target_id, int(sent), int(need)])
                continue
            debug_count("accepted_mission", [mission.kind, option.src_id, mission.target_id, int(sent), int(turns)])
            planned_commitments[target.id].append((turns, world.player, int(sent)))
            continue

        limits = []
        for option in mission.options:
            left = mission_source_left(mission.kind, option.src_id)
            limits.append(min(left, option.send_cap))
        if min(limits) <= 0:
            debug_count("multi_source_empty", [mission.kind, mission.target_id])
            continue

        missing = world.min_ships_to_own_at(
            target.id,
            mission.turns,
            world.player,
            planned_commitments=planned_commitments,
            upper_bound=sum(limits),
        )
        if missing <= 0 or sum(limits) < missing:
            debug_count("multi_missing_fail", [mission.kind, mission.target_id, int(missing), int(sum(limits))])
            continue
        desired_total = max(missing, int(mission.min_total or 0))
        if sum(limits) < desired_total:
            debug_count("multi_total_fail", [mission.kind, mission.target_id, int(desired_total), int(sum(limits))])
            continue

        ordered = sorted(
            zip(mission.options, limits),
            key=lambda item: (item[0].turns, -item[1], item[0].src_id),
        )
        sends = {}
        if mission.kind == "anchored_swarm":
            for option, limit in ordered:
                sends[option.src_id] = min(limit, option.send_cap)
            if sum(sends.values()) < desired_total:
                debug_count("multi_allocate_fail", [mission.kind, mission.target_id, int(desired_total - sum(sends.values()))])
                continue
        else:
            remaining = desired_total
            for idx, (option, limit) in enumerate(ordered):
                remaining_other = sum(other_limit for _, other_limit in ordered[idx + 1 :])
                send = min(limit, max(0, remaining - remaining_other))
                sends[option.src_id] = send
                remaining -= send
            if remaining > 0:
                debug_count("multi_allocate_fail", [mission.kind, mission.target_id, int(remaining)])
                continue

        reaimed = []
        for option, _ in ordered:
            send = sends.get(option.src_id, 0)
            if send <= 0:
                continue
            src = world.planet_by_id[option.src_id]
            fixed_aim = world.plan_shot(src.id, target.id, send)
            if fixed_aim is None:
                reaimed = []
                break
            angle, turns, _, _ = fixed_aim
            reaimed.append((option.src_id, angle, turns, send))
        if not reaimed:
            debug_count("multi_reaim_fail", [mission.kind, mission.target_id])
            continue

        turns_only = [item[2] for item in reaimed]
        eta_tol = swarm_eta_tolerance(mission.options, target, world)
        if max(turns_only) - min(turns_only) > eta_tol:
            debug_count("multi_eta_fail", [mission.kind, mission.target_id, int(max(turns_only) - min(turns_only))])
            continue

        actual_joint_turn = max(turns_only)
        owner_after, _ = world.projected_state(
            target.id,
            actual_joint_turn,
            planned_commitments=planned_commitments,
            extra_arrivals=[(turns, world.player, send) for _, _, turns, send in reaimed],
        )
        if owner_after != world.player:
            debug_count("multi_owner_fail", [mission.kind, mission.target_id, int(actual_joint_turn)])
            continue

        committed = []
        for src_id, angle, turns, send in reaimed:
            actual = append_move(
                src_id,
                angle,
                send,
                target.id,
                force=mission.kind == "heavy_swarm",
            )
            if actual <= 0:
                continue
            committed.append((turns, world.player, int(actual)))
        if sum(item[2] for item in committed) < desired_total:
            debug_count("multi_commit_short", [mission.kind, mission.target_id, int(sum(item[2] for item in committed)), int(desired_total)])
            continue
        debug_count("accepted_mission", [mission.kind, "multi", mission.target_id, int(sum(item[2] for item in committed)), int(actual_joint_turn)])
        planned_commitments[target.id].extend(committed)

    # Use leftover attack budget for one more pass after the first commitment
    # wave is fixed.
    if not world.is_very_late and allow_optional_phase():
        for src in world.my_planets:
            if expired():
                return finalize_moves()
            src_left = source_attack_left(src.id)
            if src_left < FOLLOWUP_MIN_SHIPS:
                continue

            best = None
            for target in world.planets:
                if expired():
                    return finalize_moves()
                if target.id == src.id or target.owner == world.player:
                    continue
                if reserve_source_for_local_prize(src, target, world):
                    continue
                if target.id in world.comet_ids and target.production <= LOW_VALUE_COMET_PRODUCTION:
                    continue

                seeded = world.best_probe_aim(
                    src.id,
                    target.id,
                    src_left,
                    hints=(int(target.ships) + 1,),
                )
                if seeded is None:
                    continue
                rough_ships, rough_aim = seeded

                est_turns = rough_aim[1]
                if world.is_late and est_turns > world.remaining_steps - LATE_CAPTURE_BUFFER:
                    continue

                rough_needed = world.min_ships_to_own_at(
                    target.id,
                    est_turns,
                    world.player,
                    planned_commitments=planned_commitments,
                    upper_bound=src_left,
                )
                if rough_needed <= 0 or rough_needed > src_left:
                    continue
                if opening_filter(target, est_turns, rough_needed, src_left, world, policy):
                    continue

                send = preferred_send(target, rough_needed, est_turns, src_left, world, modes, policy)
                if send < rough_needed:
                    continue

                plan = settle_plan(
                    src,
                    target,
                    src_left,
                    send,
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="capture",
                )
                if plan is None:
                    continue

                _, turns, _, need, final_send = plan
                if world.is_late and turns > world.remaining_steps - LATE_CAPTURE_BUFFER:
                    continue
                if final_send < need:
                    continue
                if not midgame_attack_allowed(src, target, final_send, turns, world, modes):
                    continue

                value = target_value(target, turns, "capture", world, modes, policy)
                if value <= 0:
                    continue

                score = apply_score_modifiers(
                    value / (final_send + turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                    target,
                    "capture",
                    world,
                )
                if best is None or score > best[0]:
                    best = (score, target, plan)

            if best is None:
                continue

            _, target, plan = best
            angle, turns, _, need, send = plan
            src_left = source_attack_left(src.id)
            if need > src_left:
                continue

            plan = settle_plan(
                src,
                target,
                src_left,
                min(src_left, send),
                world,
                planned_commitments,
                modes,
                policy,
                mission="capture",
            )
            if plan is None:
                continue

            angle, turns, _, need, send = plan
            if send < need:
                continue

            actual = append_move(src.id, angle, send, target.id)
            if actual < need:
                continue
            planned_commitments[target.id].append((turns, world.player, int(actual)))

    # If a planet cannot hold soon, prefer reinforcement first. For stacks that
    # still look doomed after the main mission pass, prefer a last useful
    # capture; otherwise retreat the stack to a safer ally.
    if expired():
        return finalize_moves()
    live_doomed = compute_live_doomed()
    if live_doomed:
        frontier_targets = (
            world.enemy_planets
            if world.enemy_planets
            else (world.static_neutral_planets or world.neutral_planets)
        )
        if frontier_targets:
            frontier_distance = {
                planet.id: nearest_distance_to_set(planet.x, planet.y, frontier_targets)
                for planet in world.my_planets
            }
        else:
            frontier_distance = {planet.id: 10**9 for planet in world.my_planets}

        for planet in world.my_planets:
            if expired():
                return finalize_moves()
            if planet.id not in live_doomed:
                continue

            if is_defense_core_planet(planet, world):
                fleet_eta, fleet_stack = enemy_fleet_pressure_to_planet(
                    planet,
                    world,
                    DOOMED_CORE_THREAT_HORIZON,
                )
                if (
                    fleet_eta is not None
                    and fleet_stack >= max(8, int(planet.production) * 3)
                ):
                    continue

            available_now = source_inventory_left(planet.id)
            if available_now < policy["reserve"].get(planet.id, 0):
                continue

            best_capture = None
            for target in world.planets:
                if expired():
                    return finalize_moves()
                if target.id == planet.id or target.owner == world.player:
                    continue
                seeded = world.best_probe_aim(
                    planet.id,
                    target.id,
                    available_now,
                    hints=(available_now, int(target.ships) + 1),
                )
                if seeded is None:
                    continue
                _, probe_aim = seeded
                probe_turns = probe_aim[1]
                if probe_turns > world.remaining_steps - 2:
                    continue

                need = world.min_ships_to_own_at(
                    target.id,
                    probe_turns,
                    world.player,
                    planned_commitments=planned_commitments,
                    upper_bound=available_now,
                )
                if need <= 0 or need > available_now:
                    continue

                plan = settle_plan(
                    planet,
                    target,
                    available_now,
                    min(available_now, max(need, int(target.ships) + 1)),
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="capture",
                )
                if plan is None:
                    continue
                angle, turns, _, plan_need, send = plan
                if send < plan_need:
                    continue
                score = target_value(target, turns, "capture", world, modes, policy) / (send + turns + 1.0)
                if target.owner not in (-1, world.player):
                    score *= 1.05
                if best_capture is None or score > best_capture[0]:
                    best_capture = (score, target.id, angle, turns, send)

            if best_capture is not None:
                _, target_id, angle, turns, need = best_capture
                actual = append_move(planet.id, angle, need, target_id)
                if actual >= 1:
                    planned_commitments[target_id].append((turns, world.player, int(actual)))
                continue

            safe_allies = [
                ally
                for ally in world.my_planets
                if ally.id != planet.id and ally.id not in live_doomed
            ]
            if not safe_allies:
                continue

            retreat_target = min(
                safe_allies,
                key=lambda ally: (
                    frontier_distance.get(ally.id, 10**9),
                    planet_distance(planet, ally),
                ),
            )
            aim = world.plan_shot(planet.id, retreat_target.id, available_now)
            if aim is None:
                continue
            angle, _, _, _ = aim
            append_move(planet.id, angle, available_now)

    # Rear planets feed the frontier through staging allies instead of acting
    # as slow solo attackers.
    if (
        (world.enemy_planets or world.neutral_planets)
        and len(world.my_planets) > 1
        and not world.is_late
        and allow_optional_phase()
    ):
        live_doomed = compute_live_doomed()
        frontier_targets = (
            world.enemy_planets
            if world.enemy_planets
            else (world.static_neutral_planets or world.neutral_planets)
        )
        frontier_distance = {
            planet.id: nearest_distance_to_set(planet.x, planet.y, frontier_targets)
            for planet in world.my_planets
        }
        safe_fronts = [
            planet for planet in world.my_planets if planet.id not in live_doomed
        ]
        if safe_fronts:
            front_anchor = min(safe_fronts, key=lambda planet: frontier_distance[planet.id])
            send_ratio = (
                REAR_SEND_RATIO_FOUR_PLAYER if world.is_four_player else REAR_SEND_RATIO_TWO_PLAYER
            )
            if modes["is_finishing"]:
                send_ratio = max(send_ratio, REAR_SEND_RATIO_FOUR_PLAYER)

            for rear in sorted(world.my_planets, key=lambda planet: -frontier_distance[planet.id]):
                if expired():
                    return finalize_moves()
                if rear.id == front_anchor.id or rear.id in live_doomed:
                    continue
                if source_attack_left(rear.id) < REAR_SOURCE_MIN_SHIPS:
                    continue
                if frontier_distance[rear.id] < frontier_distance[front_anchor.id] * REAR_DISTANCE_RATIO:
                    continue

                stage_candidates = [
                    planet
                    for planet in safe_fronts
                    if planet.id != rear.id
                    and frontier_distance[planet.id] < frontier_distance[rear.id] * REAR_STAGE_PROGRESS
                ]
                if stage_candidates:
                    front = min(
                        stage_candidates,
                        key=lambda planet: planet_distance(rear, planet),
                    )
                else:
                    objective = min(
                        frontier_targets,
                        key=lambda target: planet_distance(rear, target),
                    )
                    remaining_fronts = [planet for planet in safe_fronts if planet.id != rear.id]
                    if not remaining_fronts:
                        continue
                    front = min(
                        remaining_fronts,
                        key=lambda planet: planet_distance(planet, objective),
                    )

                if front.id == rear.id:
                    continue

                send = int(source_attack_left(rear.id) * send_ratio)
                if send < REAR_SEND_MIN_SHIPS:
                    continue

                aim = world.plan_shot(rear.id, front.id, send)
                if aim is None:
                    continue

                angle, turns, _, _ = aim
                if turns > REAR_MAX_TRAVEL_TURNS:
                    continue
                append_move(rear.id, angle, send)

    return finalize_moves()

# ============================================================
# Agent Entry Point
# ============================================================

def _read(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    getter = getattr(obs, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            value = getter(key)
            return default if value is None else value
    try:
        return obs[key]
    except Exception:
        return getattr(obs, key, default)


def activate_strategy_profile(step):
    global EDGE_AIM_ENABLED, DELAYED_SNIPE_ENABLED, LOCAL_OPENING_ENABLED, LEAN_OPENING_ENABLED
    global AGGRESSIVE_DEFENSE_ENABLED, OPENING_ROUTE_GUARD_ENABLED, OPENING_COMMITTED_ENABLED
    global OPENING_META_ENABLED

    opening_active = step < OPENING_TURN_LIMIT
    EDGE_AIM_ENABLED = PROFILE_EDGE_AIM and opening_active
    DELAYED_SNIPE_ENABLED = PROFILE_DELAYED_SNIPE and opening_active
    LOCAL_OPENING_ENABLED = PROFILE_LOCAL_OPENING and opening_active
    LEAN_OPENING_ENABLED = PROFILE_LEAN_OPENING and opening_active
    AGGRESSIVE_DEFENSE_ENABLED = PROFILE_AGGRESSIVE_DEFENSE and opening_active
    OPENING_META_ENABLED = False
    OPENING_ROUTE_GUARD_ENABLED = opening_active and (
        OPENING_ROUTE_GUARD_ALWAYS or PROFILE_OPENING_ROUTE_GUARD
    )
    OPENING_COMMITTED_ENABLED = PROFILE_OPENING_ROUTE_GUARD and opening_active


def update_profile_capture_memory(player, step, planets):
    global PROFILE_LAST_OWNERS, PROFILE_CAPTURED_AT

    for planet in planets:
        previous_owner = PROFILE_LAST_OWNERS.get(planet.id)
        if planet.owner == player and previous_owner != player:
            PROFILE_CAPTURED_AT[planet.id] = int(step)
        elif planet.owner != player and previous_owner == player:
            PROFILE_CAPTURED_AT.pop(planet.id, None)
        PROFILE_LAST_OWNERS[planet.id] = planet.owner


def configure_strategy_profile(obs):
    global EDGE_AIM_ENABLED, DELAYED_SNIPE_ENABLED, LOCAL_OPENING_ENABLED, LEAN_OPENING_ENABLED
    global AGGRESSIVE_DEFENSE_ENABLED, OPENING_ROUTE_GUARD_ENABLED, OPENING_COMMITTED_ENABLED
    global OPENING_META_ENABLED
    global PROFILE_SIGNATURE, PROFILE_EDGE_AIM, PROFILE_DELAYED_SNIPE
    global PROFILE_LOCAL_OPENING, PROFILE_LEAN_OPENING, PROFILE_AGGRESSIVE_DEFENSE
    global PROFILE_OPENING_ROUTE_GUARD, PROFILE_ARCHETYPE, PROFILE_HOME_IDS
    global PROFILE_LAST_OWNERS, PROFILE_CAPTURED_AT

    player = _read(obs, "player", 0)
    step = _read(obs, "step", 0) or 0
    raw_planets = _read(obs, "planets", []) or []
    raw_initial = _read(obs, "initial_planets", []) or raw_planets
    planets = [Planet(*planet) for planet in raw_planets]
    initial_planets = [Planet(*planet) for planet in raw_initial]
    my_planets = [planet for planet in planets if planet.owner == player]

    signature = (
        player,
        tuple(
            (
                planet[0],
                int(planet[6]),
                int(planet[5]),
                round(planet[2], 1),
                round(planet[3], 1),
            )
            for planet in raw_initial[:12]
        ),
    )

    if signature == PROFILE_SIGNATURE:
        activate_strategy_profile(step)
    else:
        PROFILE_LAST_OWNERS = {}
        PROFILE_CAPTURED_AT = {}

    if signature != PROFILE_SIGNATURE and my_planets:
        PROFILE_HOME_IDS = tuple(sorted(planet.id for planet in my_planets))
        low_orbit_route_profile = False
        low_orbit_delay_profile = False
        precise_cluster_profile = False
        delay_race_profile = False
        delay_frontier_profile = False
        static_prize_route_profile = False
        static_close_route_profile = False
        static_local_first_profile = False
        static_far_delay_profile = False
        dynamic_equal_static_delay_profile = False
        low_static_route_profile = False
        aggressive_defense_profile = False
        mid_orbit_route_profile = False
        local_opening_profile = False
        lean_opening_profile = False
        fast_local_prize_profile = False
        route_only_opening_profile = False
        tail_delay_profile = False
        tail_both_profile = False
        suppress_edge_profile = False
        opening_route_guard_profile = False
        for src in my_planets:
            src_radius = orbital_radius(src)
            src_diag = abs((src.x - CENTER_X) - (src.y - CENTER_Y))
            close_affordable_prize = any(
                target.id != src.id
                and target.owner != player
                and target.production >= 5
                and int(target.ships) <= 12
                and planet_distance(src, target) <= 13.0
                for target in initial_planets
            )
            if src.production <= 1:
                close_bad_blocker = any(
                    target.id != src.id
                    and target.owner != player
                    and target.production <= OPENING_BAD_TARGET_MAX_PROD
                    and int(target.ships) >= OPENING_BAD_TARGET_MIN_SHIPS
                    and planet_distance(src, target) <= 14.0
                    for target in initial_planets
                )
                better_local_target = any(
                    target.id != src.id
                    and target.owner != player
                    and target.production >= OPENING_BETTER_TARGET_MIN_PROD
                    and int(target.ships) <= OPENING_BETTER_TARGET_MAX_SHIPS
                    and planet_distance(src, target) <= OPENING_BETTER_TARGET_DIST
                    for target in initial_planets
                )
                if close_bad_blocker and better_local_target:
                    opening_route_guard_profile = True

            if (
                not is_static_planet(src)
                and src.production <= 1
                and src_diag > 5.0
            ):
                hold_for_local_front = any(
                    target.id != src.id
                    and target.owner != player
                    and target.production >= 4
                    and int(target.ships) <= 20
                    and planet_distance(src, target) <= 26.0
                    for target in initial_planets
                )
                close_fast_prize = any(
                    target.id != src.id
                    and target.owner != player
                    and target.production >= 4
                    and int(target.ships) <= 10
                    and planet_distance(src, target) <= 13.0
                    for target in initial_planets
                )
                close_route_prize = close_fast_prize or any(
                    target.id != src.id
                    and target.owner != player
                    and target.production >= 4
                    and int(target.ships) <= 22
                    and planet_distance(src, target) <= 14.0
                    for target in initial_planets
                )
                if close_route_prize:
                    low_orbit_route_profile = True
                elif hold_for_local_front:
                    pass
                else:
                    low_orbit_delay_profile = True

                for target in initial_planets:
                    if target.id == src.id or target.owner == player:
                        continue
                    target_dist = planet_distance(src, target)
                    if (
                        is_static_planet(target)
                        and target.production >= 4
                        and int(target.ships) <= 10
                        and 18.0 <= target_dist <= 21.0
                        and src_radius < 35.0
                    ):
                        aggressive_defense_profile = True
                        break

            if (
                not is_static_planet(src)
                and src.production == 2
                and src_radius >= 35.0
                and src_diag > 18.0
            ):
                close_bargain_prize = any(
                    target.id != src.id
                    and target.owner != player
                    and (
                        (
                            target.production >= 5
                            and int(target.ships) <= 12
                            and planet_distance(src, target) <= 14.5
                        )
                        or (
                            target.production >= 3
                            and int(target.ships) <= 20
                            and planet_distance(src, target) <= 18.0
                        )
                    )
                    for target in initial_planets
                )
                if not close_bargain_prize:
                    mid_orbit_route_profile = True

            if (
                is_static_planet(src)
                and src.production == 2
            ):
                close_local_gain = False
                distant_fast_prize = False
                brittle_far_prize = False
                for target in initial_planets:
                    if target.id == src.id or target.owner == player:
                        continue
                    target_dist = planet_distance(src, target)
                    if (
                        target.production == 3
                        and int(target.ships) <= 10
                        and target_dist <= 16.0
                    ):
                        close_local_gain = True
                    if (
                        target.production >= 5
                        and int(target.ships) <= 12
                        and target_dist <= 28.0
                    ):
                        distant_fast_prize = True
                    if (
                        target.production >= 5
                        and int(target.ships) <= 6
                        and 18.0 < target_dist <= 25.0
                    ):
                        brittle_far_prize = True
                if close_local_gain and distant_fast_prize:
                    static_local_first_profile = True
                if brittle_far_prize:
                    static_local_first_profile = True

            if (
                is_static_planet(src)
                and src.production <= 1
            ):
                avoid_outer_low_static_route = (
                    src.production <= 1
                    and src_radius > 55.0
                    and src_diag < 8.0
                )
                for target in initial_planets:
                    if target.id == src.id or target.owner == player:
                        continue
                    if (
                        not avoid_outer_low_static_route
                        and
                        (
                            (2 <= target.production <= 4 and int(target.ships) <= 22)
                            or (target.production >= 5 and int(target.ships) <= 12)
                        )
                        and planet_distance(src, target) <= 25.0
                    ):
                        low_static_route_profile = True
                        mid_orbit_route_profile = True
                        break
                    if (
                        not avoid_outer_low_static_route
                        and
                        src_diag > 35.0
                        and target.production >= 5
                        and int(target.ships) <= 20
                        and planet_distance(src, target) <= 22.0
                    ):
                        low_static_route_profile = True
                        mid_orbit_route_profile = True
                        break

            if (
                is_static_planet(src)
                and src.production == 4
                and src_radius > 60.0
                and src_diag < 2.0
            ):
                static_far_delay_profile = True

            if (
                is_static_planet(src)
                and src.production == 4
                and src_radius <= 55.0
            ):
                for target in initial_planets:
                    if target.id == src.id or target.owner == player:
                        continue
                    target_dist = planet_distance(src, target)
                    if (
                        target.production >= 4
                        and int(target.ships) <= 10
                        and 20.0 <= target_dist <= 32.0
                    ):
                        mid_orbit_route_profile = True
                        break

            if (
                is_static_planet(src)
                and src.production == 4
                and 48.0 <= src_radius <= 52.0
                and src_diag < 3.0
            ):
                for target in initial_planets:
                    if target.id == src.id or target.owner == player:
                        continue
                    if (
                        target.production >= 5
                        and int(target.ships) <= 10
                        and planet_distance(src, target) <= 18.0
                    ):
                        tail_both_profile = True
                        mid_orbit_route_profile = True
                        break

            if (
                is_static_planet(src)
                and src.production == 3
                and src_radius < 50.0
            ):
                for target in initial_planets:
                    if target.id == src.id or target.owner == player:
                        continue
                    if (
                        2 <= target.production <= 3
                        and int(target.ships) <= 14
                        and planet_distance(src, target) <= 23.0
                    ):
                        mid_orbit_route_profile = True
                        break

            if (
                is_static_planet(src)
                and src.production == 3
                and 45.0 <= src_radius <= 52.0
                and src_diag > 25.0
            ):
                for target in initial_planets:
                    if target.id == src.id or target.owner == player:
                        continue
                    if (
                        target.production >= 3
                        and int(target.ships) <= 12
                        and planet_distance(src, target) <= 14.0
                    ):
                        route_only_opening_profile = True
                        mid_orbit_route_profile = True
                        break

            if (
                is_static_planet(src)
                and src.production >= 5
                and src_radius > 50.0
                and src_diag > 20.0
            ):
                for target in initial_planets:
                    if target.id == src.id or target.owner == player:
                        continue
                    if (
                        is_static_planet(target)
                        and target.production == 2
                        and int(target.ships) <= 12
                        and planet_distance(src, target) <= 22.0
                    ):
                        aggressive_defense_profile = True
                        break

            if (
                is_static_planet(src)
                and src.production >= 5
                and 48.0 <= src_radius <= 55.0
            ):
                for target in initial_planets:
                    if target.id == src.id or target.owner == player:
                        continue
                    target_dist = planet_distance(src, target)
                    if (
                        target.production == 4
                        and int(target.ships) <= 12
                        and 20.0 <= target_dist <= 26.0
                    ):
                        tail_both_profile = True
                        mid_orbit_route_profile = True
                    if (
                        src_diag > 15.0
                        and target.production >= 4
                        and int(target.ships) <= 30
                        and target_dist <= 14.0
                    ):
                        tail_delay_profile = True
                    if tail_both_profile or tail_delay_profile:
                        break

            if (
                not is_static_planet(src)
                and src.production == 4
                and 30.0 <= src_radius <= 40.0
                and src_diag > 10.0
            ):
                for target in initial_planets:
                    if target.id == src.id or target.owner == player:
                        continue
                    if (
                        is_static_planet(target)
                        and target.production == src.production
                        and int(target.ships) <= 30
                        and planet_distance(src, target) <= 14.0
                    ):
                        dynamic_equal_static_delay_profile = True
                        break

            if (
                not is_static_planet(src)
                and src.production == 4
                and 34.0 <= src_radius <= 38.0
                and 6.0 <= src_diag <= 12.0
            ):
                close_static_front = False
                close_heavy_front = False
                for target in initial_planets:
                    if target.id == src.id or target.owner == player:
                        continue
                    target_dist = planet_distance(src, target)
                    if (
                        is_static_planet(target)
                        and target.production >= 3
                        and int(target.ships) <= 18
                        and target_dist <= 14.0
                    ):
                        close_static_front = True
                    if (
                        target.production >= 4
                        and int(target.ships) >= 35
                        and target_dist <= 14.0
                    ):
                        close_heavy_front = True
                if close_static_front and close_heavy_front:
                    tail_delay_profile = True

            if (
                not is_static_planet(src)
                and src.production == 4
                and src_radius < 30.0
                and src_diag < 3.0
            ):
                for target in initial_planets:
                    if target.id == src.id or target.owner == player:
                        continue
                    if (
                        target.production >= 5
                        and int(target.ships) <= 25
                        and planet_distance(src, target) <= 14.0
                    ):
                        delay_race_profile = True
                        break

            if (
                not is_static_planet(src)
                and src.production == 4
                and 30.0 <= src_radius <= 35.0
                and src_diag < 3.0
            ):
                for target in initial_planets:
                    if target.id == src.id or target.owner == player:
                        continue
                    if (
                        target.production >= 5
                        and int(target.ships) <= 8
                        and planet_distance(src, target) <= 28.0
                    ):
                        mid_orbit_route_profile = True
                        break

            if (
                not is_static_planet(src)
                and src.production == 3
                and 30.0 <= src_radius <= 45.0
                and src_diag > 10.0
            ):
                close_fast_prize = False
                cheap_static_detour = False
                far_cheap_route = False
                nearby_heavy_front = False
                wide_static_front = False
                for target in initial_planets:
                    if target.id == src.id or target.owner == player:
                        continue
                    target_dist = planet_distance(src, target)
                    if (
                        target.production >= 4
                        and int(target.ships) <= 10
                        and target_dist <= 18.0
                    ):
                        close_fast_prize = True
                        if is_static_planet(target) and src_radius >= 35.0 and src_diag >= 18.0:
                            wide_static_front = True
                    if (
                        target.production >= 3
                        and int(target.ships) <= 30
                        and target_dist <= 24.0
                    ):
                        nearby_heavy_front = True
                    if (
                        target.production >= 3
                        and int(target.ships) <= 10
                        and 35.0 <= target_dist <= 46.0
                    ):
                        far_cheap_route = True
                    if (
                        is_static_planet(target)
                        and target.production <= 1
                        and int(target.ships) <= 10
                        and target_dist <= 18.0
                    ):
                        cheap_static_detour = True
                if wide_static_front:
                    tail_delay_profile = True
                elif close_fast_prize and not cheap_static_detour:
                    mid_orbit_route_profile = True
                if far_cheap_route and nearby_heavy_front:
                    mid_orbit_route_profile = True

            if (
                not is_static_planet(src)
                and src.production == 2
                and player == 1
                and src.y < 20.0
            ):
                low_orbit_route_profile = True

            if (
                not is_static_planet(src)
                and src.production == 3
                and src_radius < 25.0
                and src_diag < 3.0
            ):
                has_equal_front = False
                has_outer_prize = False
                for target in initial_planets:
                    if target.id == src.id or target.owner == player:
                        continue
                    target_dist = planet_distance(src, target)
                    if (
                        target.production >= 3
                        and int(target.ships) <= 16
                        and target_dist <= 16.0
                    ):
                        has_equal_front = True
                    if (
                        target.production >= 5
                        and int(target.ships) <= 12
                        and target_dist <= 36.0
                    ):
                        has_outer_prize = True
                if has_equal_front and has_outer_prize:
                    tail_delay_profile = True
                    suppress_edge_profile = True

            if (
                not is_static_planet(src)
                and src.production >= 5
                and src_radius < 30.0
                and src_diag < 3.0
            ):
                for target in initial_planets:
                    if target.id == src.id or target.owner == player:
                        continue
                    if (
                        target.production >= 2
                        and int(target.ships) <= 10
                        and planet_distance(src, target) <= 25.0
                    ):
                        delay_race_profile = True
                        break

            if (
                not is_static_planet(src)
                and src.production >= 5
                and src_radius < 32.0
                and src_diag < 3.0
            ):
                for target in initial_planets:
                    if target.id == src.id or target.owner == player:
                        continue
                    if (
                        target.production >= 4
                        and 12 <= int(target.ships) <= 24
                        and planet_distance(src, target) <= 26.0
                    ):
                        mid_orbit_route_profile = True
                        break

            for target in initial_planets:
                if target.id == src.id or target.owner == player:
                    continue
                target_dist = planet_distance(src, target)
                if (
                    not is_static_planet(src)
                    and src.production >= 5
                    and src_radius < 30.0
                    and src_diag <= 8.0
                    and target.production >= src.production
                    and int(target.ships) <= 10
                    and target_dist <= 14.0
                ):
                    fast_local_prize_profile = True
                if (
                    is_static_planet(src)
                    and src.production in (3, 4)
                    and src_radius <= 60.0
                    and target.production >= src.production + 1
                    and int(target.ships) <= 30
                    and int(target.ships) >= int(src.ships) + 2
                    and target_dist <= 22.0
                ):
                    delay_frontier_profile = True
                if (
                    not is_static_planet(src)
                    and src.production >= 4
                    and src_diag > 5.0
                    and target.production >= src.production
                    and int(target.ships) <= 25
                    and target_dist <= 22.0
                ):
                    delay_frontier_profile = True
                if (
                    not is_static_planet(src)
                    and src.production <= 2
                    and not close_affordable_prize
                    and not (
                        src.production == 2
                        and player == 1
                        and src.y < 20.0
                    )
                    and target.production >= src.production + 2
                    and int(target.ships) <= 25
                    and target_dist <= 18.0
                ):
                    delay_frontier_profile = True
                if (
                    is_static_planet(src)
                    and src.production == 2
                    and not static_local_first_profile
                    and target.production >= 5
                    and int(target.ships) <= 12
                    and target_dist <= 28.0
                ):
                    static_prize_route_profile = True

            if (
                not is_static_planet(src)
                and src.production == 2
                and 25.0 <= src_radius <= 35.0
                and src_diag < 3.0
            ):
                for target in initial_planets:
                    if target.id == src.id or target.owner == player:
                        continue
                    if (
                        target.production >= 2
                        and int(target.ships) <= 10
                        and planet_distance(src, target) <= 18.0
                    ):
                        local_opening_profile = True
                        break

            if is_static_planet(src) and src.production == 3:
                for target in initial_planets:
                    if target.id == src.id or target.owner == player:
                        continue
                    if (
                        target.production >= 2
                        and int(target.ships) <= 12
                        and planet_distance(src, target) <= 16.0
                    ):
                        local_opening_profile = True
                        break

            if (
                not is_static_planet(src)
                and src.production == 1
                and 25.0 <= src_radius <= 30.0
                and src_diag < 3.0
                and not close_affordable_prize
            ):
                for target in initial_planets:
                    if target.id == src.id or target.owner == player:
                        continue
                    if (
                        target.production >= 3
                        and int(target.ships) <= 10
                        and planet_distance(src, target) <= 28.0
                    ):
                        lean_opening_profile = True
                        break

            if src.production == 3:
                for target in initial_planets:
                    if target.id == src.id or target.owner == player:
                        continue
                    if (
                        is_static_planet(src)
                        and target.production == 4
                        and int(target.ships) <= 6
                        and planet_distance(src, target) <= 32.0
                    ):
                        lean_opening_profile = True
                        if planet_distance(src, target) <= 18.0:
                            static_close_route_profile = True
                        break

            if is_static_planet(src) and src.production == 3:
                for target in initial_planets:
                    if target.id == src.id or target.owner == player:
                        continue
                    if (
                        target.production == 2
                        and int(target.ships) <= 8
                        and planet_distance(src, target) <= 15.0
                    ):
                        static_close_route_profile = True
                        break

            if (
                not is_static_planet(src)
                and src.production >= 5
                and src_radius < 30.0
                and src_diag < 3.0
            ):
                for target in initial_planets:
                    if target.id == src.id or target.owner == player:
                        continue
                    if int(target.ships) <= 6 and planet_distance(src, target) <= 12.0:
                        delay_race_profile = True
                        break

        low_static_home = [
            planet
            for planet in my_planets
            if planet.production <= 2 and is_static_planet(planet)
        ]
        static_delay_profile = False
        for src in low_static_home:
            for target in initial_planets:
                if target.id == src.id or target.production < 2:
                    continue
                if int(target.ships) + 1 <= 10:
                    static_delay_profile = True
                    break
            if static_delay_profile:
                break

        route_candidate = (
            not suppress_edge_profile
            and (
                low_orbit_route_profile
                or precise_cluster_profile
                or static_prize_route_profile
                or static_close_route_profile
                or mid_orbit_route_profile
                or tail_both_profile
            )
        )
        delay_candidate = (
            (static_delay_profile and not static_local_first_profile and not low_static_route_profile and not route_only_opening_profile)
            or static_far_delay_profile
            or dynamic_equal_static_delay_profile
            or delay_race_profile
            or tail_delay_profile
            or tail_both_profile
            or (
                delay_frontier_profile
                and not static_close_route_profile
                and not low_orbit_route_profile
                and not fast_local_prize_profile
                and not route_only_opening_profile
            )
            or (local_opening_profile and not static_close_route_profile and not route_only_opening_profile)
            or low_orbit_delay_profile
        )

        if tail_both_profile:
            archetype = "both"
        elif route_only_opening_profile or route_candidate:
            archetype = "route"
        elif delay_candidate:
            archetype = "delay"
        elif local_opening_profile:
            archetype = "local"
        elif lean_opening_profile:
            archetype = "lean"
        else:
            archetype = "baseline"

        PROFILE_SIGNATURE = signature
        PROFILE_ARCHETYPE = archetype
        PROFILE_EDGE_AIM = archetype in ("route", "both")
        PROFILE_DELAYED_SNIPE = archetype in ("delay", "both")
        PROFILE_LOCAL_OPENING = archetype == "local"
        PROFILE_LEAN_OPENING = archetype == "lean"
        PROFILE_AGGRESSIVE_DEFENSE = aggressive_defense_profile
        PROFILE_OPENING_ROUTE_GUARD = opening_route_guard_profile

        activate_strategy_profile(step)

    mode_value = os.environ.get("ORBIT_STRATEGY_MODE")
    mode = "" if mode_value is None else mode_value.lower()
    if mode == "baseline":
        EDGE_AIM_ENABLED = False
        DELAYED_SNIPE_ENABLED = False
        LOCAL_OPENING_ENABLED = False
        LEAN_OPENING_ENABLED = False
        AGGRESSIVE_DEFENSE_ENABLED = False
        OPENING_META_ENABLED = False
        OPENING_ROUTE_GUARD_ENABLED = False
        OPENING_COMMITTED_ENABLED = False
    elif mode == "delay":
        EDGE_AIM_ENABLED = False
        DELAYED_SNIPE_ENABLED = True
        LOCAL_OPENING_ENABLED = False
        LEAN_OPENING_ENABLED = False
        AGGRESSIVE_DEFENSE_ENABLED = False
        OPENING_META_ENABLED = False
        OPENING_ROUTE_GUARD_ENABLED = False
        OPENING_COMMITTED_ENABLED = False
    elif mode == "route":
        EDGE_AIM_ENABLED = True
        DELAYED_SNIPE_ENABLED = False
        LOCAL_OPENING_ENABLED = False
        LEAN_OPENING_ENABLED = False
        AGGRESSIVE_DEFENSE_ENABLED = False
        OPENING_META_ENABLED = False
        OPENING_ROUTE_GUARD_ENABLED = False
        OPENING_COMMITTED_ENABLED = False
    elif mode == "both":
        EDGE_AIM_ENABLED = True
        DELAYED_SNIPE_ENABLED = True
        LOCAL_OPENING_ENABLED = False
        LEAN_OPENING_ENABLED = False
        AGGRESSIVE_DEFENSE_ENABLED = False
        OPENING_META_ENABLED = False
        OPENING_ROUTE_GUARD_ENABLED = False
        OPENING_COMMITTED_ENABLED = False
    elif mode == "meta":
        EDGE_AIM_ENABLED = True
        DELAYED_SNIPE_ENABLED = True
        LOCAL_OPENING_ENABLED = False
        LEAN_OPENING_ENABLED = False
        AGGRESSIVE_DEFENSE_ENABLED = False
        OPENING_META_ENABLED = step < OPENING_TURN_LIMIT
        OPENING_ROUTE_GUARD_ENABLED = step < OPENING_TURN_LIMIT
        OPENING_COMMITTED_ENABLED = False
    elif mode in ("profile", "auto", ""):
        pass

    update_profile_capture_memory(player, step, planets)


def build_world(obs):
    player = _read(obs, "player", 0)
    step = _read(obs, "step", 0) or 0
    raw_planets = _read(obs, "planets", []) or []
    raw_fleets = _read(obs, "fleets", []) or []
    ang_vel = _read(obs, "angular_velocity", 0.0) or 0.0
    raw_init = _read(obs, "initial_planets", []) or []
    comets = _read(obs, "comets", []) or []
    comet_ids = set(_read(obs, "comet_planet_ids", []) or [])

    planets = [Planet(*planet) for planet in raw_planets]
    fleets = [Fleet(*fleet) for fleet in raw_fleets]
    initial_planets = [Planet(*planet) for planet in raw_init]
    initial_by_id = {planet.id: planet for planet in initial_planets}

    return WorldModel(
        player=player,
        step=step,
        planets=planets,
        fleets=fleets,
        initial_by_id=initial_by_id,
        ang_vel=ang_vel,
        comets=comets,
        comet_ids=comet_ids,
    )


def orbit_log_enabled(step):
    mode = os.environ.get("ORBIT_LOG", "1").strip().lower()
    if mode in ("0", "false", "off", "no"):
        return False
    try:
        every = max(1, int(os.environ.get("ORBIT_LOG_EVERY", "1")))
    except ValueError:
        every = 1
    return int(step) % every == 0


def orbit_phase(world):
    if world.is_very_late:
        return "very_late"
    if world.is_late:
        return "late"
    if world.is_opening:
        return "opening"
    return "mid"


def compact_flags():
    flags = []
    if EDGE_AIM_ENABLED:
        flags.append("edge")
    if DELAYED_SNIPE_ENABLED:
        flags.append("delay")
    if LOCAL_OPENING_ENABLED:
        flags.append("local")
    if LEAN_OPENING_ENABLED:
        flags.append("lean")
    if OPENING_META_ENABLED:
        flags.append("meta")
    if AGGRESSIVE_DEFENSE_ENABLED:
        flags.append("def")
    if OPENING_ROUTE_GUARD_ENABLED:
        flags.append("guard")
    return flags


def first_current_ray_hit(world, src_id, angle):
    src = world.planet_by_id.get(src_id)
    if src is None:
        return None
    start_x, start_y = launch_point(src.x, src.y, src.radius, angle)
    best = None
    for planet in world.planets:
        if planet.id == src_id:
            continue
        hit = ray_circle_hit_distance(start_x, start_y, angle, planet.x, planet.y, planet.radius)
        if hit is None:
            continue
        if best is None or hit < best[0]:
            best = (hit, planet.id)
    return None if best is None else best[1]


def top_planet_rows(planets, limit=4):
    rows = []
    for planet in sorted(
        planets,
        key=lambda item: (-int(item.production), -int(item.ships), item.id),
    )[:limit]:
        rows.append([planet.id, int(planet.production), int(planet.ships)])
    return rows


def threatened_planet_rows(world, horizon=30, limit=4):
    rows = []
    for planet in world.my_planets:
        fall_turn = world.fall_turn_map.get(planet.id)
        if fall_turn is None or fall_turn > horizon:
            continue
        rows.append([planet.id, int(fall_turn), int(planet.production), int(planet.ships)])
    rows.sort(key=lambda item: (item[1], -item[2], item[0]))
    return rows[:limit]


def budget_pressure_rows(world, limit=4):
    rows = []
    for planet in world.my_planets:
        keep_needed = int(world.keep_needed_map.get(planet.id, 0))
        first_enemy = world.first_enemy_map.get(planet.id)
        fall_turn = world.fall_turn_map.get(planet.id)
        if keep_needed <= 0 and first_enemy is None and fall_turn is None:
            continue
        rows.append(
            [
                planet.id,
                int(planet.production),
                int(planet.ships),
                keep_needed,
                -1 if first_enemy is None else int(first_enemy),
                -1 if fall_turn is None else int(fall_turn),
                1 if world.holds_full_map.get(planet.id, True) else 0,
            ]
        )
    rows.sort(
        key=lambda item: (
            -(item[3] / max(1, item[2])),
            item[5] if item[5] >= 0 else 10**9,
            -item[1],
            item[0],
        )
    )
    return rows[:limit]


def target_snapshot_rows(world, limit=5):
    rows = []
    for target in world.planets:
        if target.owner == world.player:
            continue
        my_dist = nearest_distance_to_set(target.x, target.y, world.my_planets)
        enemy_dist = nearest_distance_to_set(target.x, target.y, world.enemy_planets)
        score = (
            float(target.production) * 10.0
            + (8.0 if target.owner not in (-1, world.player) else 0.0)
        ) / (1.0 + my_dist * 0.25 + max(0, int(target.ships)) * 0.10)
        rows.append(
            [
                target.id,
                int(target.owner),
                int(target.production),
                int(target.ships),
                round(my_dist, 1),
                round(enemy_dist, 1),
                round(score, 3),
            ]
        )
    rows.sort(key=lambda item: (-item[6], -item[2], item[4], item[0]))
    return rows[:limit]


def classify_empty_action(world, elapsed_ms, soft_budget):
    if world.my_planets and elapsed_ms >= max(0.0, (soft_budget or 0.0) * 1000.0 - 35.0):
        return "near_soft_deadline"
    debug = getattr(world, "debug_info", {}) or {}
    counts = debug.get("counts", {})
    if debug.get("stage", "").startswith("opening") and debug.get("opening_return") == 0:
        return "opening_wait"
    budget_rows = debug.get("budget") or []
    if world.my_planets and budget_rows and all(int(row[4]) <= 0 for row in budget_rows):
        if (
            not threatened_planet_rows(world)
            and all(int(row[3]) >= int(row[2]) for row in budget_rows)
        ):
            return "full_reserve_no_immediate_risk"
        if any(row[5] is not None and int(row[5]) <= DOOMED_EVAC_TURN_LIMIT for row in budget_rows):
            return "locked_under_attack"
        return "no_attack_budget"
    if counts.get("multi_pair_eta_fail", 0) or counts.get("multi_trio_eta_fail", 0):
        return "multi_eta_filtered"
    if counts.get("multi_pair_total_fail", 0) or counts.get("multi_trio_total_fail", 0):
        return "multi_not_enough_mass"
    if counts.get("partial_option", 0) and not debug.get("mission_count", 0):
        return "partial_options_unassembled"
    if counts.get("candidate_pair", 0) and not debug.get("mission_count", 0):
        return "all_candidates_filtered"
    if debug.get("mission_count", 0) and not counts.get("accepted_mission", 0):
        return "missions_rejected"
    return "no_selected_action"


def log_agent_turn(world, actions, elapsed_ms, obs=None, config=None, soft_budget=None):
    if not orbit_log_enabled(world.step):
        return
    try:
        my_fleet_ships = sum(int(fleet.ships) for fleet in world.fleets if fleet.owner == world.player)
        my_planet_ships = sum(int(planet.ships) for planet in world.my_planets)
        enemy_planet_ships = sum(int(planet.ships) for planet in world.enemy_planets)
        neutral_prod = sum(int(planet.production) for planet in world.neutral_planets)
        enemy_owner_stats = [
            [owner, int(world.owner_production.get(owner, 0)), int(world.owner_strength.get(owner, 0))]
            for owner in sorted(world.owner_strength)
            if owner not in (-1, world.player)
        ]
        action_rows = []
        action_detail_rows = []
        for src_id, angle, ships in actions:
            first_hit = first_current_ray_hit(world, int(src_id), float(angle))
            action_rows.append(
                [
                    int(src_id),
                    first_hit,
                    int(ships),
                    round(float(angle), 4),
                ]
            )
            hit = world.planet_by_id.get(first_hit)
            action_detail_rows.append(
                [
                    int(src_id),
                    first_hit,
                    None if hit is None else int(hit.owner),
                    None if hit is None else int(hit.production),
                    None if hit is None else int(hit.ships),
                    int(ships),
                    round(float(angle), 4),
                ]
            )

        payload = {
            "v": 1,
            "step": int(world.step),
            "player": int(world.player),
            "phase": orbit_phase(world),
            "players": int(world.num_players),
            "profile": PROFILE_ARCHETYPE,
            "flags": compact_flags(),
            "home": [int(home_id) for home_id in PROFILE_HOME_IDS],
            "my": [
                len(world.my_planets),
                int(world.my_prod),
                int(world.my_total),
                int(my_planet_ships),
                int(my_fleet_ships),
            ],
            "enemy": [
                len(world.enemy_planets),
                int(world.enemy_prod),
                int(world.enemy_total),
                int(enemy_planet_ships),
                enemy_owner_stats,
            ],
            "neutral": [len(world.neutral_planets), int(neutral_prod)],
            "core": top_planet_rows(world.my_planets),
            "risk": threatened_planet_rows(world),
            "pressure": budget_pressure_rows(world),
            "targets": target_snapshot_rows(world),
            "act": action_rows,
            "actx": action_detail_rows,
            "sent": sum(int(action[2]) for action in actions),
            "ms": round(float(elapsed_ms), 2),
            "soft_ms": round(float((soft_budget or 0.0) * 1000.0), 2),
            "over": _read(obs, "remainingOverageTime", None) if obs is not None else None,
            "empty": classify_empty_action(world, elapsed_ms, soft_budget) if not actions else "",
            "timing": timing_rows(world),
            "dbg": getattr(world, "debug_info", {}),
        }
        print(
            ORBIT_LOG_PREFIX + json.dumps(payload, separators=(",", ":")),
            file=sys.stderr,
            flush=True,
        )
    except Exception as exc:
        fallback = {
            "v": 1,
            "step": int(getattr(world, "step", -1)),
            "player": int(getattr(world, "player", -1)),
            "log_error": type(exc).__name__,
        }
        print(
            ORBIT_LOG_PREFIX + json.dumps(fallback, separators=(",", ":")),
            file=sys.stderr,
            flush=True,
        )


def agent(obs, config=None):
    start_time = time.perf_counter()
    phase_start = time.perf_counter()
    configure_strategy_profile(obs)
    configure_ms = (time.perf_counter() - phase_start) * 1000.0
    phase_start = time.perf_counter()
    world = build_world(obs)
    timing_add(world, "phase.configure_strategy_profile", configure_ms)
    timing_add(world, "phase.build_world", (time.perf_counter() - phase_start) * 1000.0)
    if not world.my_planets:
        actions = []
        log_agent_turn(world, actions, (time.perf_counter() - start_time) * 1000.0, obs=obs, config=config, soft_budget=0.0)
        return actions
    act_timeout = _read(config, "actTimeout", SOFT_ACT_DEADLINE) if config is not None else SOFT_ACT_DEADLINE
    soft_budget = min(SOFT_ACT_DEADLINE, max(0.55, act_timeout))
    deadline = None if os.environ.get("ORBIT_NO_DEADLINE") == "1" else start_time + soft_budget
    phase_start = time.perf_counter()
    actions = plan_moves(world, deadline=deadline)
    timing_add(world, "phase.plan_moves", (time.perf_counter() - phase_start) * 1000.0)
    log_agent_turn(world, actions, (time.perf_counter() - start_time) * 1000.0, obs=obs, config=config, soft_budget=soft_budget)
    return actions


__all__ = ["agent", "build_world"]
