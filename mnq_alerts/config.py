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
IB_END_HOUR = 10  # Initial Balance ends at 10:30 AM
IB_END_MIN = 30

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

# Bot trading parameters (validated via bot_risk_backtest.py over 214 days).
BOT_TARGET_POINTS = 12.0  # Take profit distance from line price
BOT_STOP_POINTS = 25.0  # Stop loss distance from line price
DAILY_LOSS_LIMIT_USD = 150.0  # Stop trading for the day after losing this much
MAX_CONSECUTIVE_LOSSES = 3  # Stop trading for the day after N straight losses

# ── Display ─────────────────────────────────────────────────────────────────────
# Override the auto-detected local timezone for log timestamps.
# Useful on EC2 where the system timezone is UTC.
# Example: DISPLAY_TZ=America/Los_Angeles

DISPLAY_TZ = os.getenv("DISPLAY_TZ", "")
