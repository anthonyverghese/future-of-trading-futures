"""
config.py — All tunable settings. Credentials are loaded from .env.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Market Timing (Eastern) ────────────────────────────────────────────────────

MARKET_OPEN_HOUR  = 9
MARKET_OPEN_MIN   = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MIN  = 0
IB_END_HOUR       = 10  # Initial Balance ends at 10:30 AM
IB_END_MIN        = 30

# ── Databento ──────────────────────────────────────────────────────────────────

DATABENTO_API_KEY = os.getenv("DATABENTO_API_KEY", "")
DATABENTO_DATASET = "GLBX.MDP3"   # CME Globex futures
DATABENTO_SYMBOL  = "MNQ.c.0"     # Continuous front-month MNQ (auto-rolled)

# ── Alerts ─────────────────────────────────────────────────────────────────────

ALERT_THRESHOLD_POINTS = 10  # Notify when MNQ is within this many points of a level
CHECK_INTERVAL_SECONDS = 30  # How often to poll during RTH

# ── Pushover ───────────────────────────────────────────────────────────────────
# https://pushover.net — API token from pushover.net/apps, User Key from dashboard

PUSHOVER_TOKEN    = os.getenv("PUSHOVER_TOKEN", "")
PUSHOVER_USER_KEY = os.getenv("PUSHOVER_USER_KEY", "")

# ── Display ─────────────────────────────────────────────────────────────────────
# Override the auto-detected local timezone for log timestamps.
# Useful on EC2 where the system timezone is UTC.
# Example: DISPLAY_TZ=America/Los_Angeles

DISPLAY_TZ = os.getenv("DISPLAY_TZ", "")
