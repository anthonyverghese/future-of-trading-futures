"""
trend_filter_backtest.py — Walk-forward backtest for trend/volatility filters.

Tests adding trend and volatility awareness to the human alert scoring:
  1. Trend filter: penalize BUY when price falling, SELL when price rising
  2. Volatility filter: penalize all signals during extreme price swings
  3. Combined: both together
  4. Score penalty vs hard filter comparison

Motivated by Apr 8, 2026: afternoon selloff triggered 7 incorrect alerts
as the human app kept firing BUY at support levels being broken.

Usage:
    python -u trend_filter_backtest.py
"""

from __future__ import annotations

import datetime
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(__file__))

from targeted_backtest import (
    DayCache,
    load_cached_days,
    load_day,
    preprocess_day,
    simulate_day,
    ET,
)
from score_optimizer import EnrichedAlert, Weights, compute_tick_rate, score_alert

_ET = pytz.timezone("America/New_York")

INITIAL_TRAIN_DAYS = 60
STEP_DAYS = 20


# ══════════════════════════════════════════════════════════════════════════════
# ENRICHMENT — Add trend and volatility metrics to each alert
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrendAlert(EnrichedAlert):
    """EnrichedAlert extended with trend and volatility metrics."""

    trend_30m: float = 0.0  # price change over last 30 min (positive = up)
    trend_15m: float = 0.0  # price change over last 15 min
    trend_60m: float = 0.0  # price change over last 60 min
    range_30m: float = 0.0  # high - low over last 30 min (volatility proxy)
    range_60m: float = 0.0  # high - low over last 60 min
    # Day-level regime metrics (same for all alerts on one day)
    day_range: float = 0.0  # full intraday high - low (up to alert time)
    day_max_drawdown: float = 0.0  # max drop from intraday high to low
    is_steep_day: bool = False  # True if day_range exceeds threshold


def compute_trend_metrics(df: pd.DataFrame, alert_ts: pd.Timestamp) -> dict:
    """Compute trend and volatility metrics at alert time."""
    metrics = {
        "trend_15m": 0.0,
        "trend_30m": 0.0,
        "trend_60m": 0.0,
        "range_30m": 0.0,
        "range_60m": 0.0,
    }

    for minutes, key in [(15, "trend_15m"), (30, "trend_30m"), (60, "trend_60m")]:
        window_start = alert_ts - pd.Timedelta(minutes=minutes)
        mask = (df.index >= window_start) & (df.index <= alert_ts)
        window = df.loc[mask, "price"]
        if len(window) >= 2:
            metrics[key] = float(window.iloc[-1] - window.iloc[0])

    for minutes, key in [(30, "range_30m"), (60, "range_60m")]:
        window_start = alert_ts - pd.Timedelta(minutes=minutes)
        mask = (df.index >= window_start) & (df.index <= alert_ts)
        window = df.loc[mask, "price"]
        if len(window) >= 2:
            metrics[key] = float(window.max() - window.min())

    return metrics


def load_trend_alerts(dates: list[datetime.date]) -> list[TrendAlert]:
    """Load all alerts with trend/volatility metrics."""
    all_alerts: list[TrendAlert] = []
    cw = 0
    cl = 0

    for i, date in enumerate(dates):
        try:
            df = load_day(date)
            dc = preprocess_day(df, date)
            if dc is None:
                continue
        except Exception:
            continue

        day_alerts = simulate_day(dc)
        first_price = float(dc.post_ib_prices[0])
        day_alerts.sort(key=lambda a: a.alert_time)

        # Compute day-level regime metrics
        rth_prices = dc.full_prices
        day_high = float(np.max(rth_prices))
        day_low = float(np.min(rth_prices))
        day_range = day_high - day_low
        # Max drawdown: largest peak-to-trough within the day
        running_max = np.maximum.accumulate(rth_prices)
        drawdowns = running_max - rth_prices
        day_max_dd = float(np.max(drawdowns))
        # "Steep day" = intraday range > 200 pts (~1% on MNQ ~20k)
        # This threshold catches news/event days vs normal ranging
        steep_threshold = 200.0
        is_steep = day_range > steep_threshold

        for a in day_alerts:
            if a.outcome not in ("correct", "incorrect"):
                continue

            if hasattr(a.alert_time, "astimezone") and a.alert_time.tzinfo:
                now_et = a.alert_time.astimezone(
                    datetime.timezone(datetime.timedelta(hours=-4))
                ).time()
            else:
                now_et = None

            tick_rate = compute_tick_rate(dc.full_df, pd.Timestamp(a.alert_time))
            session_move = a.entry_price - first_price
            trend = compute_trend_metrics(dc.full_df, pd.Timestamp(a.alert_time))

            ta = TrendAlert(
                date=date,
                level=a.level,
                direction=a.direction,
                entry_count=a.level_test_count,
                outcome=a.outcome,
                entry_price=a.entry_price,
                line_price=a.line_price,
                alert_time=a.alert_time,
                now_et=now_et,
                tick_rate=tick_rate,
                session_move_pts=session_move,
                consecutive_wins=cw,
                consecutive_losses=cl,
                trend_15m=trend["trend_15m"],
                trend_30m=trend["trend_30m"],
                trend_60m=trend["trend_60m"],
                range_30m=trend["range_30m"],
                range_60m=trend["range_60m"],
                day_range=day_range,
                day_max_drawdown=day_max_dd,
                is_steep_day=is_steep,
            )
            all_alerts.append(ta)

            if a.outcome == "correct":
                cw += 1
                cl = 0
            else:
                cl += 1
                cw = 0

        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{len(dates)} days...", flush=True)

    return all_alerts


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS — Win rates by trend and volatility buckets
# ══════════════════════════════════════════════════════════════════════════════


def analyze_buckets(alerts: list[TrendAlert]) -> None:
    """Print win rate analysis by trend and volatility buckets."""
    print(f"\n  {'='*90}")
    print(f"  FACTOR ANALYSIS — {len(alerts)} decided alerts")
    print(f"  {'='*90}")

    def wr(subset: list) -> str:
        if not subset:
            return "  (no data)"
        w = sum(1 for a in subset if a.outcome == "correct")
        n = len(subset)
        return f"{w:>5}W {n-w:>4}L  {n:>5} trades  {w/n*100:>5.1f}%"

    # Trend against direction
    print(f"\n  TREND vs DIRECTION (is the recent trend against the signal?):")
    print(f"  {'Bucket':<55} {'W':>5} {'L':>4}  {'Total':>5}  {'WR':>6}")
    for lookback, attr in [
        ("15m", "trend_15m"),
        ("30m", "trend_30m"),
        ("60m", "trend_60m"),
    ]:
        for label, fn in [
            (
                f"  BUY + {lookback} trend < -30 (falling into buy)",
                lambda a, at=attr: a.direction == "up" and getattr(a, at) < -30,
            ),
            (
                f"  BUY + {lookback} trend < -50 (strongly falling)",
                lambda a, at=attr: a.direction == "up" and getattr(a, at) < -50,
            ),
            (
                f"  BUY + {lookback} trend < -75 (crashing into buy)",
                lambda a, at=attr: a.direction == "up" and getattr(a, at) < -75,
            ),
            (
                f"  SELL + {lookback} trend > +30 (rising into sell)",
                lambda a, at=attr: a.direction == "down" and getattr(a, at) > 30,
            ),
            (
                f"  SELL + {lookback} trend > +50 (strongly rising)",
                lambda a, at=attr: a.direction == "down" and getattr(a, at) > 50,
            ),
            (
                f"  SELL + {lookback} trend > +75 (surging into sell)",
                lambda a, at=attr: a.direction == "down" and getattr(a, at) > 75,
            ),
        ]:
            subset = [a for a in alerts if fn(a)]
            print(f"  {label:<55} {wr(subset)}")
        print()

    # Volatility
    print(f"  VOLATILITY (recent range):")
    for lookback, attr in [("30m", "range_30m"), ("60m", "range_60m")]:
        for label, fn in [
            (f"  {lookback} range > 50 pts", lambda a, at=attr: getattr(a, at) > 50),
            (f"  {lookback} range > 75 pts", lambda a, at=attr: getattr(a, at) > 75),
            (f"  {lookback} range > 100 pts", lambda a, at=attr: getattr(a, at) > 100),
            (f"  {lookback} range > 150 pts", lambda a, at=attr: getattr(a, at) > 150),
            (
                f"  {lookback} range <= 50 pts (calm)",
                lambda a, at=attr: getattr(a, at) <= 50,
            ),
        ]:
            subset = [a for a in alerts if fn(a)]
            print(f"  {label:<55} {wr(subset)}")
        print()

    # Day regime analysis
    print(f"  DAY REGIME (steep vs normal days):")
    steep_dates = set(a.date for a in alerts if a.is_steep_day)
    normal_dates = set(a.date for a in alerts if not a.is_steep_day)
    print(f"    Steep days: {len(steep_dates)}, Normal days: {len(normal_dates)}")
    for label, fn in [
        ("Normal days (range <= 200 pts)", lambda a: not a.is_steep_day),
        ("Steep days (range > 200 pts)", lambda a: a.is_steep_day),
        ("Steep days (range > 150 pts)", lambda a: a.day_range > 150),
        ("Steep days (range > 250 pts)", lambda a: a.day_range > 250),
        ("Steep days (range > 300 pts)", lambda a: a.day_range > 300),
        ("Days with max DD > 100 pts", lambda a: a.day_max_drawdown > 100),
        ("Days with max DD > 150 pts", lambda a: a.day_max_drawdown > 150),
        ("Days with max DD > 200 pts", lambda a: a.day_max_drawdown > 200),
    ]:
        subset = [a for a in alerts if fn(a)]
        print(f"  {label:<55} {wr(subset)}")
    print()

    # On steep days, how do counter-trend signals perform?
    print(f"  STEEP DAY BREAKDOWN:")
    steep_alerts = [a for a in alerts if a.is_steep_day]
    for label, fn in [
        ("All alerts on steep days", lambda a: True),
        (
            "BUY on steep days with 30m trend < -30",
            lambda a: a.direction == "up" and a.trend_30m < -30,
        ),
        (
            "SELL on steep days with 30m trend > +30",
            lambda a: a.direction == "down" and a.trend_30m > 30,
        ),
        (
            "Aligned with trend on steep days",
            lambda a: (a.direction == "up" and a.trend_30m > 0)
            or (a.direction == "down" and a.trend_30m < 0),
        ),
        (
            "Against trend on steep days",
            lambda a: (a.direction == "up" and a.trend_30m < 0)
            or (a.direction == "down" and a.trend_30m > 0),
        ),
    ]:
        subset = [a for a in steep_alerts if fn(a)]
        print(f"  {label:<55} {wr(subset)}")
    print()

    # Combined: trend against direction AND high volatility
    print(f"  COMBINED — counter-trend in volatile conditions:")
    for label, fn in [
        (
            "BUY + 30m trend < -30 + 30m range > 75",
            lambda a: a.direction == "up" and a.trend_30m < -30 and a.range_30m > 75,
        ),
        (
            "BUY + 30m trend < -50 + 60m range > 100",
            lambda a: a.direction == "up" and a.trend_30m < -50 and a.range_60m > 100,
        ),
        (
            "SELL + 30m trend > +30 + 30m range > 75",
            lambda a: a.direction == "down" and a.trend_30m > 30 and a.range_30m > 75,
        ),
        (
            "SELL + 30m trend > +50 + 60m range > 100",
            lambda a: a.direction == "down" and a.trend_30m > 50 and a.range_60m > 100,
        ),
        (
            "Any direction against 30m trend > 40 + range > 75",
            lambda a: (
                (a.direction == "up" and a.trend_30m < -40)
                or (a.direction == "down" and a.trend_30m > 40)
            )
            and a.range_30m > 75,
        ),
    ]:
        subset = [a for a in alerts if fn(a)]
        print(f"  {label:<55} {wr(subset)}")


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD — Test trend filters with adaptive weight tuning
# ══════════════════════════════════════════════════════════════════════════════


def score_with_trend(
    a: TrendAlert,
    w: Weights,
    trend_lookback: str = "30m",
    trend_threshold: float = 40.0,
    trend_penalty: int = -3,
    vol_lookback: str = "30m",
    vol_threshold: float = 100.0,
    vol_penalty: int = -2,
    use_trend: bool = True,
    use_vol: bool = True,
    # Regime-aware: raise min_score on steep days
    steep_min_score_boost: int = 0,
) -> int:
    """Score an alert with optional trend and volatility penalties.

    steep_min_score_boost: added to the effective min_score on steep days.
    This is applied as a negative score adjustment so the walk-forward
    can use a single min_score threshold. E.g., boost=2 means steep days
    effectively need score >= 6 instead of >= 4.
    """
    base = score_alert(a, w)

    if use_trend:
        trend_val = getattr(a, f"trend_{trend_lookback}", 0.0)
        # Penalize BUY when price has been falling, SELL when rising
        if a.direction == "up" and trend_val < -trend_threshold:
            base += trend_penalty
        elif a.direction == "down" and trend_val > trend_threshold:
            base += trend_penalty

    if use_vol:
        vol_val = getattr(a, f"range_{vol_lookback}", 0.0)
        if vol_val > vol_threshold:
            base += vol_penalty

    # Regime-aware: effectively raise the bar on steep days
    if steep_min_score_boost > 0 and a.is_steep_day:
        base -= steep_min_score_boost

    return base


def walk_forward_with_trend(
    all_alerts: list[TrendAlert],
    valid_days: list[datetime.date],
    configs: list[dict],
) -> None:
    """Walk-forward validation for trend filter configs."""
    by_date: dict[datetime.date, list[TrendAlert]] = {}
    for a in all_alerts:
        by_date.setdefault(a.date, []).append(a)

    n = len(valid_days)
    test_day_count = n - INITIAL_TRAIN_DAYS

    print(f"\n  {'='*110}")
    print(f"  WALK-FORWARD TREND FILTER — {n} days, {INITIAL_TRAIN_DAYS}+{STEP_DAYS}")
    print(f"  {'='*110}")

    # Baseline: current scoring, no trend filter
    print(
        f"\n  {'Config':<55} {'W':>5} {'L':>5} {'Tot':>5} {'WR%':>6} {'/day':>5} {'Delta':>6}"
    )
    print(f"  {'-'*55} {'-'*5} {'-'*5} {'-'*5} {'-'*6} {'-'*5} {'-'*6}")

    for cfg in configs:
        label = cfg.pop("_label")
        min_score = cfg.pop("_min_score", 4)
        all_scored: list[tuple[TrendAlert, int]] = []

        k = INITIAL_TRAIN_DAYS
        while k < n:
            train_days = valid_days[:k]
            test_days = valid_days[k : k + STEP_DAYS]
            if not test_days:
                break

            # Train weights on training data (same as existing walk-forward)
            train_alerts = [a for d in train_days for a in by_date.get(d, [])]
            w = _fit_weights_from_trend_alerts(train_alerts)

            # Score test alerts with trend filter
            test_alerts = [a for d in test_days for a in by_date.get(d, [])]
            for a in test_alerts:
                s = score_with_trend(a, w, **cfg)
                all_scored.append((a, s))

            k += STEP_DAYS

        # Filter by min_score and report
        passing = [(a, s) for a, s in all_scored if s >= min_score]
        if passing:
            w_count = sum(1 for a, _ in passing if a.outcome == "correct")
            l_count = sum(1 for a, _ in passing if a.outcome == "incorrect")
            total = w_count + l_count
            wr = w_count / total * 100 if total else 0
            per_day = total / max(1, test_day_count)
            # Baseline comparison
            baseline_wr = _baseline_wr.get("wr", 0)
            delta = wr - baseline_wr
            print(
                f"  {label:<55} {w_count:>5} {l_count:>5} {total:>5} "
                f"{wr:>5.1f}% {per_day:>5.1f} {delta:>+5.1f}%"
            )
        else:
            print(f"  {label:<55} NO TRADES")

        # Restore popped keys for potential reuse
        cfg["_label"] = label
        cfg["_min_score"] = min_score


# Global baseline for delta calculation
_baseline_wr: dict[str, float] = {}


def _fit_weights_from_trend_alerts(alerts: list[TrendAlert]) -> Weights:
    """Fit scoring weights from trend alerts (delegates to existing fit logic)."""
    from walk_forward import fit_weights

    # Convert TrendAlert list to EnrichedAlert list (parent class)
    return fit_weights(alerts)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()

    print("Step 1: Loading data...")
    all_days = sorted(load_cached_days())
    print(f"  {len(all_days)} cached trading days")

    print("Step 2: Loading alerts with trend metrics...")
    all_alerts = load_trend_alerts(all_days)
    print(f"  {len(all_alerts)} decided alerts loaded in {time.time()-t0:.1f}s")

    valid_days = sorted(set(a.date for a in all_alerts))
    print(f"  {len(valid_days)} days with alerts")

    # Step 3: Factor analysis (full dataset, for understanding)
    analyze_buckets(all_alerts)

    # Step 4: Walk-forward — test configurations
    by_date: dict[datetime.date, list[TrendAlert]] = {}
    for a in all_alerts:
        by_date.setdefault(a.date, []).append(a)

    # First compute baseline WR for delta column
    n = len(valid_days)
    baseline_scored: list[tuple[TrendAlert, int]] = []
    k = INITIAL_TRAIN_DAYS
    while k < n:
        train_days = valid_days[:k]
        test_days = valid_days[k : k + STEP_DAYS]
        if not test_days:
            break
        train_alerts = [a for d in train_days for a in by_date.get(d, [])]
        w = _fit_weights_from_trend_alerts(train_alerts)
        test_alerts = [a for d in test_days for a in by_date.get(d, [])]
        for a in test_alerts:
            s = score_alert(a, w)
            baseline_scored.append((a, s))
        k += STEP_DAYS

    passing_baseline = [(a, s) for a, s in baseline_scored if s >= 4]
    bw = sum(1 for a, _ in passing_baseline if a.outcome == "correct")
    bl = sum(1 for a, _ in passing_baseline if a.outcome == "incorrect")
    baseline_wr = bw / (bw + bl) * 100 if (bw + bl) else 0
    baseline_total = bw + bl
    baseline_per_day = baseline_total / max(1, n - INITIAL_TRAIN_DAYS)
    _baseline_wr["wr"] = baseline_wr
    print(
        f"\n  BASELINE (score>=4, no trend filter): "
        f"{bw}W/{bl}L = {baseline_wr:.1f}% WR, {baseline_per_day:.1f}/day"
    )

    # Test configs — sweep trend parameters
    configs = []

    # No filter baseline
    configs.append(
        {
            "_label": "BASELINE: no trend/vol filter",
            "_min_score": 4,
            "use_trend": False,
            "use_vol": False,
        }
    )

    # Trend filter only — sweep lookback and threshold
    for lookback in ["15m", "30m", "60m"]:
        for threshold in [30, 40, 50, 75]:
            for penalty in [-2, -3, -4, -5]:
                configs.append(
                    {
                        "_label": f"Trend {lookback} >{threshold}pts penalty={penalty}",
                        "_min_score": 4,
                        "trend_lookback": lookback,
                        "trend_threshold": threshold,
                        "trend_penalty": penalty,
                        "use_trend": True,
                        "use_vol": False,
                    }
                )

    # Volatility filter only
    for vol_lb in ["30m", "60m"]:
        for vol_thresh in [75, 100, 125, 150]:
            for vol_pen in [-2, -3, -4]:
                configs.append(
                    {
                        "_label": f"Vol {vol_lb} >{vol_thresh}pts penalty={vol_pen}",
                        "_min_score": 4,
                        "trend_lookback": "30m",
                        "trend_threshold": 40,
                        "trend_penalty": -3,
                        "use_trend": False,
                        "use_vol": True,
                        "vol_lookback": vol_lb,
                        "vol_threshold": vol_thresh,
                        "vol_penalty": vol_pen,
                    }
                )

    # Combined trend + volatility
    for t_lb, t_thresh, t_pen in [
        ("30m", 40, -3),
        ("30m", 50, -3),
        ("30m", 30, -4),
        ("60m", 50, -3),
    ]:
        for v_lb, v_thresh, v_pen in [
            ("30m", 75, -2),
            ("60m", 100, -2),
            ("30m", 100, -3),
        ]:
            configs.append(
                {
                    "_label": f"T:{t_lb}>{t_thresh}({t_pen}) + V:{v_lb}>{v_thresh}({v_pen})",
                    "_min_score": 4,
                    "trend_lookback": t_lb,
                    "trend_threshold": t_thresh,
                    "trend_penalty": t_pen,
                    "use_trend": True,
                    "use_vol": True,
                    "vol_lookback": v_lb,
                    "vol_threshold": v_thresh,
                    "vol_penalty": v_pen,
                }
            )

    # Regime-aware: raise min_score on steep days (different behavior by day type)
    for boost in [1, 2, 3, 4]:
        configs.append(
            {
                "_label": f"Regime: steep days score boost +{boost} (no trend/vol)",
                "_min_score": 4,
                "use_trend": False,
                "use_vol": False,
                "steep_min_score_boost": boost,
            }
        )

    # Regime + trend combined: best of both worlds
    for boost in [2, 3]:
        for t_lb, t_thresh, t_pen in [("30m", 40, -3), ("30m", 50, -4)]:
            configs.append(
                {
                    "_label": f"Regime boost +{boost} + T:{t_lb}>{t_thresh}({t_pen})",
                    "_min_score": 4,
                    "trend_lookback": t_lb,
                    "trend_threshold": t_thresh,
                    "trend_penalty": t_pen,
                    "use_trend": True,
                    "use_vol": False,
                    "steep_min_score_boost": boost,
                }
            )

    # Regime + volatility
    for boost in [2, 3]:
        for v_lb, v_thresh, v_pen in [("30m", 75, -2), ("60m", 100, -2)]:
            configs.append(
                {
                    "_label": f"Regime boost +{boost} + V:{v_lb}>{v_thresh}({v_pen})",
                    "_min_score": 4,
                    "use_trend": False,
                    "use_vol": True,
                    "vol_lookback": v_lb,
                    "vol_threshold": v_thresh,
                    "vol_penalty": v_pen,
                    "steep_min_score_boost": boost,
                }
            )

    # Regime + trend + vol
    for boost in [2, 3]:
        configs.append(
            {
                "_label": f"Regime +{boost} + T:30m>40(-3) + V:60m>100(-2)",
                "_min_score": 4,
                "trend_lookback": "30m",
                "trend_threshold": 40,
                "trend_penalty": -3,
                "use_trend": True,
                "use_vol": True,
                "vol_lookback": "60m",
                "vol_threshold": 100,
                "vol_penalty": -2,
                "steep_min_score_boost": boost,
            }
        )

    walk_forward_with_trend(all_alerts, valid_days, configs)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
