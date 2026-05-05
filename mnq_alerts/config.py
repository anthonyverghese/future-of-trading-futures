"""
config.py — All tunable settings. Credentials are loaded from .env.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Market Timing (Eastern) ────────────────────────────────────────────────────

MARKET_OPEN_HOUR = 9
MARKET_OPEN_MIN = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MIN = 0
IB_END_HOUR = 10  # Initial Balance includes the 10:30 bar; window is [9:30, 10:31)
IB_END_MIN = 31

# ── Databento ──────────────────────────────────────────────────────────────────

DATABENTO_API_KEY = os.getenv("DATABENTO_API_KEY", "")
DATABENTO_DATASET = "GLBX.MDP3"  # CME Globex futures
DATABENTO_SYMBOL = "MNQ.c.0"  # Continuous front-month MNQ (auto-rolled)

# ── Alerts ─────────────────────────────────────────────────────────────────────

ALERT_THRESHOLD_POINTS = 7  # Notify when MNQ is within this many points of a level
ALERT_EXIT_POINTS = 20  # Points away from reference to reset the alert zone
CHECK_INTERVAL_SECONDS = 30  # How often to poll during RTH

# ── Outcome Evaluation ───────────────────────────────────────────────────────

HIT_THRESHOLD = 1.0  # Points — price within this distance counts as "hit the line"
TARGET_POINTS = 8.0  # Points price must move in recommended direction (target)
STOP_POINTS = 20.0  # Points against — stopped out before target = incorrect
EVAL_WINDOW_MINS = 15  # Minutes after hitting line to evaluate outcome

# ── Pushover ───────────────────────────────────────────────────────────────────
# https://pushover.net — API token from pushover.net/apps, User Key from dashboard

PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN", "")
PUSHOVER_USER_KEY = os.getenv("PUSHOVER_USER_KEY", "")

# ── IBKR Trading ──────────────────────────────────────────────────────────────
# Set IBKR_TRADING_ENABLED=true in .env to enable live order submission.
# Requires IB Gateway or TWS running with API access on IBKR_HOST:IBKR_PORT.

IBKR_TRADING_ENABLED = os.getenv("IBKR_TRADING_ENABLED", "false").lower() == "true"
IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", "4002"))  # 4002=Gateway paper, 4001=Gateway live
IBKR_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", "1"))
IBKR_ACCOUNT = os.getenv(
    "IBKR_ACCOUNT", ""
)  # Paper account ID (e.g. "DU1234567") — verified at startup

# Bot trading parameters (validated via walk-forward over 318 days, VWAP-corrected).
BOT_ENTRY_THRESHOLD = (
    1.0  # Bot trades when price is within 1 pt of level (vs 7 for human)
)
BOT_TARGET_POINTS = 8.0  # Default target (used if level not in PER_LEVEL_TS)
BOT_STOP_POINTS = 20.0  # Default stop
# Per-level target/stop: MFE-based, trained on first 200 days, validated on 131 held-out.
# Interior fibs have bigger bounces (T10-T12), extensions have smaller (T6).
BOT_PER_LEVEL_TS = {
    "FIB_EXT_HI_1.272": (6, 20),
    "FIB_EXT_LO_1.272": (6, 20),
    "FIB_0.236": (8, 25),
    "FIB_0.618": (12, 20),
    "FIB_0.764": (10, 25),
    "IBH": (6, 20),
}
# Per-level max entries per day (data-driven from WR-by-entry-count analysis).
# On Mondays, caps are doubled (BOT_MONDAY_DOUBLE_CAPS).
BOT_PER_LEVEL_MAX_ENTRIES = {
    "FIB_0.236": 18,  # gets stronger with more tests, increased from 12 (validated 2026-04-29)
    "FIB_0.618": 3,   # collapses at test #4 (57.5% WR)
    "FIB_0.764": 5,   # degrades after #5
    "FIB_EXT_HI_1.272": 6,  # weakening in recent quarters, cap for safety
    "FIB_EXT_LO_1.272": 6,  # weak at #7-9
    "IBH": 7,
}
BOT_MONDAY_DOUBLE_CAPS = True  # Double per-level caps on Mondays (+$2.18/day vs baseline)
BOT_INCLUDE_VWAP = False  # VWAP excluded: drags in weak regimes (66% WR)
BOT_INCLUDE_IBL = False  # IBL excluded: -$1.5/day, 72.2% WR (weakest level)
BOT_INCLUDE_IBH = True  # IBH re-included: SELL only (+$4.05/day, 80.8% WR). BUY blocked via BOT_DIRECTION_FILTER.
BOT_INCLUDE_INTERIOR_FIBS = True  # Interior fib retracements (0.236, 0.618, 0.764)
BOT_EXCLUDE_LEVELS = {
    "FIB_0.5",          # test #1 is negative, no consistent edge
    "FIB_EXT_LO_1.272", # added 2026-05-05 — slippage-aware backtest at
                        # buffer=1.0pt: -$0.99/tr (-$0.95/day, MaxDD impact
                        # significant). At target/2 it was -$0.76/tr. The
                        # 81% WR on this level is above the 76.9% T6/S20
                        # breakeven, but slippage tips it negative.
                        # Dropping it: $/day +$15.66, MaxDD $980 (down
                        # from $1,110 with this level included).
}
BOT_DIRECTION_FILTER = {"IBH": "down"}  # IBH: SELL only (80.8% WR, BUY drags it down)
BOT_MIN_SCORE = -99  # Unscored: scoring hurts OOS (validated 2026-04-26)
BOT_TREND_LOOKBACK_MIN = 60  # Minutes to look back for trend calculation
BOT_VOL_FILTER_MIN_RANGE_PCT = 0.0015  # Skip entry when 30m range < 0.15% of price
BOT_MAX_ENTRIES_PER_LEVEL = 12  # Default max (overridden by BOT_PER_LEVEL_MAX_ENTRIES)
BOT_GLOBAL_COOLDOWN_AFTER_LOSS_SECS = 0  # Disabled (2026-05-02): hurts P&L at $200 limit.
BOT_MOMENTUM_THRESHOLD = 0.0  # Disabled (2026-05-04): hurts P&L -$1.74/day with v2 accurate sim.
# Was 5.0. Momentum filter blocked profitable trades more often than bad ones.
BOT_MOMENTUM_LOOKBACK_MIN = 5  # Minutes to look back for momentum calculation
BOT_FAILED_FILL_COOLDOWN_SECS = 60  # Per-level cooldown after limit order not filled
BOT_FILL_TIMEOUT_SECS = 3.0  # How long to wait for entry limit to fill before cancelling
# Entry limit buffer in pts. Limit price is line ± buffer (above for BUY,
# below for SELL). Was target/2 (3-6pt across levels) which gave 99% fill
# rate but allowed up to 6pt of slippage from line on bad fills.
# Switched to a fixed 1.0pt after a slippage-modeled buffer sweep
# (336 days, 100ms latency, tick-data fill model) showed:
#   target/2: +$14.96/day, fill 99.4%, MaxDD $1,711
#   buffer=1: +$17.89/day, fill 87.6%, MaxDD $1,295   ← +$2.93/day, lower DD
# Tighter buffer drops the worst-fill trades (strong-rejection scenarios
# fill at the limit price) in exchange for keeping wins closer to target.
BOT_ENTRY_LIMIT_BUFFER_PTS = 1.0
# Was 30s. With higher loss limit, recovery trades after losses are profitable.
# Removing cooldown + suppression + adaptive caps = +$49.46/day (vs +$41.73 with all on).
DAILY_LOSS_LIMIT_USD = 200.0  # Stop trading for the day after losing this much
# Increased from $100→$200 (2026-05-02): bot recovers from losses (84.5% WR
# after 2+L). Higher limit captures recovery trades. +$1.93/day, 27 bad days
# (vs 51 at $100). MaxDD increases $512→$683.
BOT_TIMEOUT_SECS = (
    15 * 60
)  # Close position if neither target nor stop hits in this window
# Matches bot_risk_backtest.py WINDOW_SECS so live = backtest
BOT_EOD_FLATTEN_BUFFER_MIN = 2  # Flatten any open position this many minutes
# before MARKET_CLOSE (so 15:58 ET with 4pm close) to avoid overnight margin.
# New bot entries are also blocked once this cutoff passes.

# ── Display ─────────────────────────────────────────────────────────────────────
# Override the auto-detected local timezone for log timestamps.
# Useful on EC2 where the system timezone is UTC.
# Example: DISPLAY_TZ=America/Los_Angeles

DISPLAY_TZ = os.getenv("DISPLAY_TZ", "")
