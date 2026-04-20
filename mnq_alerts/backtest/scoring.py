"""Scoring functions and weight training.

Single source of truth for all scoring factors, buckets, and weights.
Both human and bot scoring use the same factor definitions — the
difference is in the weight values.

Factors (28 total):
  5 level quality + 9 direction combos + 1 time + 2 tick rate +
  4 entry count + 5 session move + 2 streak + 1 volatility +
  3 bot-only (approach speed, tick density)
"""

from __future__ import annotations

from dataclasses import dataclass

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from score_optimizer import suggest_weight


@dataclass
class EntryFactors:
    """All scoring factors at an entry point."""
    level: str
    direction: str
    entry_count: int
    et_mins: int          # ET minutes since midnight
    tick_rate: float      # trades/min in 3-min window
    session_move: float   # points from day open
    range_30m: float      # 30-min range in points
    approach_speed: float # pts/sec in last 10s
    tick_density: float   # ticks/sec in last 10s


# Human weights from score_optimizer.py Weights class (proven, hand-tuned).
HUMAN_WEIGHTS = {
    "lv_ibh": -1, "lv_ibl": 1, "lv_fib_hi": 2, "lv_fib_lo": 1, "lv_vwap": -1,
    "co_fib_hi_up": 2, "co_fib_lo_down": 1, "co_ibl_down": 1, "co_vwap_up": 1,
    "co_ibh_up": -1, "co_ibl_up": -1, "co_fib_lo_up": -1, "co_fib_hi_down": -1,
    "co_vwap_down": -1,
    "tp": 2,
    "ts": 2, "tl": 0,
    "e1": -1, "e2": 1, "e3": -1, "e5": 1,
    "mg": 2, "mr": 2, "ms": 1, "mz": -3, "mb": 0,
    "sw": 3, "sl": -2,
    "vh": -2,
    "af": 0, "av": 0, "dl": 0,
}


def score_entry(f: EntryFactors, w: dict, cw: int = 0, cl: int = 0) -> int:
    """Score an entry using the given weights. Works for both human and bot."""
    s = 0

    # Level quality.
    lv_map = {"IBH": "lv_ibh", "IBL": "lv_ibl", "FIB_EXT_HI_1.272": "lv_fib_hi",
              "FIB_EXT_LO_1.272": "lv_fib_lo", "VWAP": "lv_vwap"}
    s += w.get(lv_map.get(f.level, ""), 0)

    # Direction combo.
    combo_map = {
        ("FIB_EXT_HI_1.272", "up"): "co_fib_hi_up",
        ("FIB_EXT_LO_1.272", "down"): "co_fib_lo_down",
        ("IBL", "down"): "co_ibl_down",
        ("VWAP", "up"): "co_vwap_up",
        ("IBH", "up"): "co_ibh_up",
        ("IBL", "up"): "co_ibl_up",
        ("FIB_EXT_LO_1.272", "up"): "co_fib_lo_up",
        ("FIB_EXT_HI_1.272", "down"): "co_fib_hi_down",
        ("VWAP", "down"): "co_vwap_down",
    }
    s += w.get(combo_map.get((f.level, f.direction), ""), 0)

    # Time of day — power hour (>= 15:00 ET = 900 mins).
    if f.et_mins >= 900:
        s += w.get("tp", 0)

    # Tick rate — sweet spot 1750-2000 (human bucket), low <500.
    if 1750 <= f.tick_rate < 2000:
        s += w.get("ts", 0)
    elif f.tick_rate < 500:
        s += w.get("tl", 0)

    # Entry count.
    if f.entry_count == 1: s += w.get("e1", 0)
    elif f.entry_count == 2: s += w.get("e2", 0)
    elif f.entry_count == 3: s += w.get("e3", 0)
    elif f.entry_count == 5: s += w.get("e5", 0)

    # Session move (point-based buckets, same as human).
    m = f.session_move
    if 10 < m <= 20: s += w.get("mg", 0)
    elif -20 < m <= -10: s += w.get("mr", 0)
    elif m <= -50: s += w.get("ms", 0)
    elif 0 < m <= 10: s += w.get("mz", 0)
    elif m > 50: s += w.get("mb", 0)

    # Streak.
    if cw >= 2: s += w.get("sw", 0)
    elif cl >= 2: s += w.get("sl", 0)

    # Volatility — 30m range > 75 pts.
    if f.range_30m > 75.0:
        s += w.get("vh", 0)

    # Bot-only: approach speed.
    if f.approach_speed > 3.0: s += w.get("av", 0)
    elif f.approach_speed > 1.5: s += w.get("af", 0)

    # Bot-only: tick density.
    if f.tick_density < 5: s += w.get("dl", 0)

    return s


def train_weights(
    entries_outcomes: list[tuple[EntryFactors, str, int, int]],
) -> dict:
    """Train weights from (factors, outcome, cw, cl) tuples.

    Uses suggest_weight (round(delta/2.5)) per factor bucket.
    """
    if not entries_outcomes:
        return {}
    total = len(entries_outcomes)
    wc = sum(1 for _, o, _, _ in entries_outcomes if o == "win")
    bl = wc / total * 100
    sw = suggest_weight

    def wr(fn):
        sub = [i for i in range(total) if fn(
            entries_outcomes[i][0], entries_outcomes[i][2], entries_outcomes[i][3]
        )]
        if len(sub) < 30:
            return bl
        return sum(1 for i in sub if entries_outcomes[i][1] == "win") / len(sub) * 100

    w = {}
    for lv, k in [("IBH", "lv_ibh"), ("IBL", "lv_ibl"), ("FIB_EXT_HI_1.272", "lv_fib_hi"),
                   ("FIB_EXT_LO_1.272", "lv_fib_lo"), ("VWAP", "lv_vwap")]:
        w[k] = sw(wr(lambda f, cw, cl, l=lv: f.level == l), bl)

    for lv, d, k in [("FIB_EXT_HI_1.272", "up", "co_fib_hi_up"),
                      ("FIB_EXT_LO_1.272", "down", "co_fib_lo_down"),
                      ("IBL", "down", "co_ibl_down"), ("VWAP", "up", "co_vwap_up"),
                      ("IBH", "up", "co_ibh_up"), ("IBL", "up", "co_ibl_up"),
                      ("FIB_EXT_LO_1.272", "up", "co_fib_lo_up"),
                      ("FIB_EXT_HI_1.272", "down", "co_fib_hi_down"),
                      ("VWAP", "down", "co_vwap_down")]:
        w[k] = sw(wr(lambda f, cw, cl, l=lv, dr=d: f.level == l and f.direction == dr), bl)

    w["tp"] = sw(wr(lambda f, cw, cl: f.et_mins >= 900), bl)
    w["ts"] = sw(wr(lambda f, cw, cl: 1750 <= f.tick_rate < 2000), bl)
    w["tl"] = sw(wr(lambda f, cw, cl: f.tick_rate < 500), bl)
    w["e1"] = sw(wr(lambda f, cw, cl: f.entry_count == 1), bl)
    w["e2"] = sw(wr(lambda f, cw, cl: f.entry_count == 2), bl)
    w["e3"] = sw(wr(lambda f, cw, cl: f.entry_count == 3), bl)
    w["e5"] = sw(wr(lambda f, cw, cl: f.entry_count == 5), bl)
    w["mg"] = sw(wr(lambda f, cw, cl: 10 < f.session_move <= 20), bl)
    w["mr"] = sw(wr(lambda f, cw, cl: -20 < f.session_move <= -10), bl)
    w["ms"] = sw(wr(lambda f, cw, cl: f.session_move <= -50), bl)
    w["mz"] = sw(wr(lambda f, cw, cl: 0 < f.session_move <= 10), bl)
    w["mb"] = sw(wr(lambda f, cw, cl: f.session_move > 50), bl)
    w["sw"] = sw(wr(lambda f, cw, cl: cw >= 2), bl)
    w["sl"] = sw(wr(lambda f, cw, cl: cl >= 2), bl)
    w["vh"] = sw(wr(lambda f, cw, cl: f.range_30m > 75), bl)
    w["af"] = sw(wr(lambda f, cw, cl: 1.5 < f.approach_speed <= 3.0), bl)
    w["av"] = sw(wr(lambda f, cw, cl: f.approach_speed > 3.0), bl)
    w["dl"] = sw(wr(lambda f, cw, cl: f.tick_density < 5), bl)
    return w
