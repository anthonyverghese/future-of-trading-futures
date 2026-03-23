"""
backtest.py — Backtests the MNQ alert system over the last 45 trading days.

Fetches historical trade data from Databento, simulates the exact alert
logic (VWAP, IBH/IBL, zone states with reference-price locking), evaluates
outcomes (correct/incorrect/inconclusive), trains multiple classifiers to
predict incorrect outcomes, and prints a results table with a
"Correctly Avoided" column.

Usage:
    python mnq_alerts/backtest.py

Requires a Databento subscription with historical data access.
"""

from __future__ import annotations

import datetime
import os
import sys
from dataclasses import dataclass, field

import databento as db
import joblib
import numpy as np
import pandas as pd
import pytz
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from config import DATABENTO_API_KEY

MODEL_PATH = os.path.join(os.path.dirname(__file__), "alert_model.joblib")

ET = pytz.timezone("America/New_York")
PT = pytz.timezone("America/Los_Angeles")

DATASET = "GLBX.MDP3"
SYMBOL = "MNQ.c.0"
MARKET_OPEN = datetime.time(9, 30)
IB_END = datetime.time(10, 30)
MARKET_CLOSE = datetime.time(16, 0)

ALERT_THRESHOLD = 7.0  # points — zone entry (must match live config)
EXIT_THRESHOLD = 20.0  # points — zone exit (must match live config)
HIT_THRESHOLD = 1.0  # points — price within this = "touched the line"
TARGET_POINTS = 8.0  # points in recommended direction = correct
STOP_POINTS = 20.0  # points against — stopped out before target = incorrect
WINDOW_SECS = 15 * 60  # 15-minute evaluation window
FEATURE_SECS = 3 * 60  # 3-minute approach window BEFORE alert fires
CONFLUENCE_PTS = 10.0  # pts — line within this of a prior day level = confluence


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class Alert:
    date: datetime.date
    alert_time: datetime.datetime
    level: str  # VWAP, IBH, or IBL
    line_price: float  # level price locked at zone entry
    entry_price: float  # MNQ price when alert fired
    direction: str  # 'up' = BUY rec, 'down' = SELL rec
    level_test_count: int = 1  # which test of this level today (1 = first touch)
    prior_confluence: bool = (
        False  # True if line is within CONFLUENCE_PTS of a prior day level
    )
    hit_time: datetime.datetime | None = None
    outcome_time: datetime.datetime | None = None
    outcome: str = "inconclusive"
    features: dict = field(default_factory=dict)
    cv_pred: int | None = None  # 0=predicted incorrect (avoid), 1=predicted correct


class ZoneState:
    """Alert zone state — mirrors the live LevelState logic exactly."""

    def __init__(self, name: str, price: float) -> None:
        self.name = name
        self.price = price
        self.in_zone = False
        self.ref: float | None = None
        self.entry_count = 0  # cumulative zone entries today for this level

    def update(
        self, current_price: float, new_level_price: float | None = None
    ) -> bool:
        """Returns True if an alert should fire (zone just entered).

        new_level_price updates self.price (used for drifting VWAP).
        Exit is checked against self.ref (locked at entry) so VWAP drift
        does not reset or re-trigger the alert.
        """
        if new_level_price is not None:
            self.price = new_level_price

        if self.in_zone:
            if abs(current_price - self.ref) > EXIT_THRESHOLD:
                self.in_zone = False
                self.ref = None
            return False

        if abs(current_price - self.price) <= ALERT_THRESHOLD:
            self.in_zone = True
            self.ref = self.price
            self.entry_count += 1
            return True

        return False


# ── Data fetching ─────────────────────────────────────────────────────────────


def get_trading_days(n: int = 45, offset: int = 0) -> list[datetime.date]:
    """Return n trading weekdays ending offset trading days before yesterday.

    offset=0  → most recent 45 days (default)
    offset=45 → the 45 days prior to that (out-of-sample validation period)
    """
    days: list[datetime.date] = []
    d = datetime.datetime.now(ET).date() - datetime.timedelta(days=1)
    # Skip `offset` trading days first.
    skipped = 0
    while skipped < offset:
        if d.weekday() < 5:
            skipped += 1
        d -= datetime.timedelta(days=1)
    # Then collect n trading days.
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= datetime.timedelta(days=1)
    return sorted(days)


CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")


def _cache_path(date: datetime.date) -> str:
    return os.path.join(CACHE_DIR, f"MNQ_{date}.parquet")


def fetch_trades(client: db.Historical, date: datetime.date) -> pd.DataFrame:
    """Fetch RTH trades for one day. Returns DataFrame with ET DatetimeIndex.

    Results are cached to data_cache/MNQ_<date>.parquet so subsequent runs
    load from disk instead of re-downloading from Databento.
    """
    path = _cache_path(date)
    if os.path.exists(path):
        print(f"    [cache] loading {date} from disk", flush=True)
        return pd.read_parquet(path)

    start = ET.localize(datetime.datetime.combine(date, MARKET_OPEN)).isoformat()
    end = ET.localize(datetime.datetime.combine(date, MARKET_CLOSE)).isoformat()

    store = client.timeseries.get_range(
        dataset=DATASET,
        schema="trades",
        stype_in="continuous",
        symbols=[SYMBOL],
        start=start,
        end=end,
    )

    rows = []
    for rec in store:
        if not isinstance(rec, db.TradeMsg):
            continue
        ts = pd.Timestamp(rec.ts_event, unit="ns", tz="UTC").tz_convert(ET)
        price = rec.price / 1_000_000_000
        size = int(rec.size)
        rows.append((ts, price, size))
        if len(rows) % 10_000 == 0:
            print(f"    ... {len(rows):,} trades downloaded", flush=True)

    if not rows:
        return pd.DataFrame(columns=["price", "size"])

    df = (
        pd.DataFrame(rows, columns=["ts", "price", "size"]).set_index("ts").sort_index()
    )

    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_parquet(path)
    print(f"    [cache] saved {date} ({len(df):,} trades)", flush=True)
    return df


# ── Simulation ────────────────────────────────────────────────────────────────


def simulate_and_evaluate(
    df: pd.DataFrame,
    date: datetime.date,
    prior_levels: list[float] | None = None,
) -> list[Alert]:
    """
    Simulate alert system + evaluate outcomes for one trading day.
    prior_levels: list of key prices from the prior session (IBH, IBL, close).
    Returns a list of Alert objects with outcomes and features populated.
    """
    if df.empty:
        return []

    # Compute IBH / IBL from 9:30–10:30 trade prices.
    ib = df[(df.index.time >= MARKET_OPEN) & (df.index.time < IB_END)]
    if ib.empty:
        return []
    ibh = float(ib["price"].max())
    ibl = float(ib["price"].min())

    # Compute running VWAP, session high/low from 9:30 (all O(n), no look-ahead).
    df = df.copy()
    df["vwap"] = (df["price"] * df["size"]).cumsum() / df["size"].cumsum()
    df["session_high"] = df["price"].cummax()
    df["session_low"] = df["price"].cummin()
    day_open = float(df["price"].iloc[0])

    # ── Alert zone simulation (post-IB only) ─────────────────────────────────
    ib_range = ibh - ibl

    # Fibonacci levels derived from IB range (fixed after 10:30 AM, like IBH/IBL).
    # Retracements sit between IBL and IBH; extensions sit beyond.
    fib_levels = {
        "FIB_EXT_LO_1.618": ibl - 0.618 * ib_range,  # 161.8% extension below IBL
        "FIB_EXT_LO_1.272": ibl - 0.272 * ib_range,  # 127.2% extension below IBL
        "FIB_RET_0.236": ibl + 0.236 * ib_range,  # 23.6% retracement
        "FIB_RET_0.382": ibl + 0.382 * ib_range,  # 38.2% retracement
        "FIB_RET_0.500": ibl + 0.500 * ib_range,  # 50% (midpoint)
        "FIB_RET_0.618": ibl + 0.618 * ib_range,  # 61.8% retracement
        "FIB_RET_0.786": ibl + 0.786 * ib_range,  # 78.6% retracement
        "FIB_EXT_HI_1.272": ibh + 0.272 * ib_range,  # 127.2% extension above IBH
        "FIB_EXT_HI_1.618": ibh + 0.618 * ib_range,  # 161.8% extension above IBH
    }

    zones = {
        "IBH": ZoneState("IBH", ibh),
        "IBL": ZoneState("IBL", ibl),
        "VWAP": ZoneState("VWAP", float(df["vwap"].iloc[0])),
    }
    for fib_name, fib_price in fib_levels.items():
        zones[fib_name] = ZoneState(fib_name, fib_price)

    alerts: list[Alert] = []
    post_ib = df[df.index.time >= IB_END]

    # itertuples is ~8x faster than iterrows for large DataFrames.
    for tick_num, row in enumerate(post_ib.itertuples()):
        if tick_num % 10_000 == 0:
            print(
                f"    [sim] {row.Index.strftime('%Y-%m-%d %H:%M:%S')} ET  "
                f"({tick_num:,} ticks)",
                flush=True,
            )

        ts = row.Index
        price = row.price
        vwap = row.vwap
        # Fib levels are fixed (like IBH/IBL), only VWAP drifts.
        level_prices = {"IBH": ibh, "IBL": ibl, "VWAP": vwap}
        for fib_name, fib_price in fib_levels.items():
            level_prices[fib_name] = fib_price

        for name, zone in zones.items():
            if zone.update(price, new_level_price=level_prices[name]):
                direction = "up" if price > zone.ref else "down"
                confluence = bool(
                    prior_levels
                    and any(abs(zone.ref - p) <= CONFLUENCE_PTS for p in prior_levels)
                )
                alerts.append(
                    Alert(
                        date=date,
                        alert_time=ts.to_pydatetime(warn=False),
                        level=name,
                        line_price=zone.ref,
                        entry_price=price,
                        direction=direction,
                        level_test_count=zone.entry_count,
                        prior_confluence=confluence,
                    )
                )

    # ── Outcome evaluation (vectorized — no iterrows) ─────────────────────────
    prices = df["price"]

    for alert in alerts:
        alert_ts = pd.Timestamp(alert.alert_time)
        window_end = alert_ts + pd.Timedelta(seconds=WINDOW_SECS)

        # Phase 1: first tick where price touches the line within 15 min.
        hit_seg = prices[(prices.index > alert_ts) & (prices.index <= window_end)]
        hit_mask = abs(hit_seg - alert.line_price) <= HIT_THRESHOLD
        if not hit_mask.any():
            alert.outcome = "inconclusive"
            continue

        hit_ts = hit_mask.idxmax()
        alert.hit_time = hit_ts.to_pydatetime(warn=False)

        # Phase 2: did price hit +TARGET_POINTS before -STOP_POINTS within 15 min?
        # Correct   = target reached before stop hit within the window.
        # Incorrect = stop hit first, OR neither hit within window (time expired).
        eval_end = hit_ts + pd.Timedelta(seconds=WINDOW_SECS)
        eval_seg = prices[(prices.index > hit_ts) & (prices.index <= eval_end)]

        if alert.direction == "up":
            target_mask = eval_seg >= alert.line_price + TARGET_POINTS
            stop_mask = eval_seg <= alert.line_price - STOP_POINTS
        else:
            target_mask = eval_seg <= alert.line_price - TARGET_POINTS
            stop_mask = eval_seg >= alert.line_price + STOP_POINTS

        target_hit = target_mask.any()
        stop_hit = stop_mask.any()

        if target_hit and stop_hit:
            # Both triggered — whichever came first wins.
            target_ts = eval_seg.index[target_mask][0]
            stop_ts = eval_seg.index[stop_mask][0]
            if target_ts <= stop_ts:
                alert.outcome = "correct"
                alert.outcome_time = target_ts.to_pydatetime(warn=False)
            else:
                alert.outcome = "incorrect"
                alert.outcome_time = stop_ts.to_pydatetime(warn=False)
        elif target_hit:
            alert.outcome = "correct"
            alert.outcome_time = eval_seg.index[target_mask][0].to_pydatetime(
                warn=False
            )
        else:
            # Stop hit or time expired — both count as incorrect.
            alert.outcome = "incorrect"
            if stop_hit:
                alert.outcome_time = eval_seg.index[stop_mask][0].to_pydatetime(
                    warn=False
                )

    # ── Feature extraction (2-min approach window BEFORE alert fires) ────────
    # Window = [alert_ts - 2min, alert_ts]. All data is available at the
    # moment the alert fires — zero look-ahead bias.
    #
    # Note: inconclusive alerts (price entered zone but never touched line)
    # are excluded from feature extraction and model training because we have
    # no outcome label. This means the model is trained only on alerts where
    # price did touch the line — a selection bias. Approach patterns that
    # predict failure-to-touch are invisible to the model.
    for alert in alerts:
        if alert.outcome not in ("correct", "incorrect"):
            continue  # inconclusive: no outcome label to train on

        alert_ts = pd.Timestamp(alert.alert_time)
        window_start = alert_ts - pd.Timedelta(seconds=FEATURE_SECS)
        window = df[(df.index >= window_start) & (df.index <= alert_ts)]

        if len(window) < 3:
            # Too few ticks in approach window (e.g. alert fired very close
            # to IB end) — skip rather than compute degenerate features.
            continue

        prices_arr = window["price"].values
        sizes_arr = window["size"].values
        n = len(prices_arr)

        # Split window into two equal halves.
        mid = n // 2
        first_prices = prices_arr[:mid] if mid >= 2 else prices_arr[:1]
        second_prices = prices_arr[mid:] if n - mid >= 2 else prices_arr[-1:]
        first_sizes = sizes_arr[:mid]
        second_sizes = sizes_arr[mid:]

        # ── Approach momentum features ────────────────────────────────────────
        # Sign convention: positive = price moving TOWARD the line (approach),
        #                  negative = price moving AWAY from the line (retreat).
        # direction="up": price is above the line; approaching = moving DOWN.
        # direction="down": price is below the line; approaching = moving UP.
        is_up = alert.direction == "up"

        def toward(val: float) -> float:
            """Positive = moving toward the line. Call only within this iteration."""
            return -val if is_up else val

        overall_change = prices_arr[-1] - prices_arr[0]
        approach_momentum = toward(overall_change)  # >0 = consistent approach

        # Linear regression slope — more robust to noise than endpoint diff.
        x = np.arange(n, dtype=float)
        slope = float(np.polyfit(x, prices_arr, 1)[0])
        approach_slope = toward(slope)

        # Sub-window momentum: first half (early approach) and second half (late).
        first_change = (
            (first_prices[-1] - first_prices[0]) if len(first_prices) >= 2 else 0.0
        )
        second_change = (
            (second_prices[-1] - second_prices[0]) if len(second_prices) >= 2 else 0.0
        )
        approach_first = toward(first_change)
        approach_second = toward(second_change)

        # Acceleration: is approach strengthening (+) or stalling/reversing (-)?
        approach_accel = approach_second - approach_first

        # Volatility and normalized approach momentum.
        volatility = float(np.std(prices_arr))
        norm_approach = approach_momentum / volatility if volatility > 1e-9 else 0.0

        # ── Pullback feature ──────────────────────────────────────────────────
        # Max pullback: how far price moved AGAINST the approach direction
        # relative to where it started. Measures interruption quality.
        #
        # direction="up" (descending toward line): pullback = price going higher
        #   than the opening of the approach window.
        #   max_pullback = max(0, max(prices) - prices[0])
        #   → 0 for a clean descent; positive if price spiked up mid-approach.
        #
        # direction="down" (ascending toward line): pullback = price going lower
        #   than the opening of the approach window.
        #   max_pullback = max(0, prices[0] - min(prices))
        #   → 0 for a clean ascent; positive if price dipped down mid-approach.
        if is_up:
            max_pullback = max(0.0, float(np.max(prices_arr)) - prices_arr[0])
        else:
            max_pullback = max(0.0, prices_arr[0] - float(np.min(prices_arr)))

        # ── Volume and activity features ──────────────────────────────────────
        first_vol = int(np.sum(first_sizes))
        second_vol = int(np.sum(second_sizes))
        volume_trend = second_vol - first_vol  # positive = participation increasing
        tick_rate = n / (FEATURE_SECS / 60)  # trades per minute in the window

        # ── Session context features ───────────────────────────────────────────
        # Look up session high/low at alert time (all computed cumulatively, no look-ahead).
        row_at_alert = df.loc[:alert_ts, ["session_high", "session_low"]].iloc[-1]
        session_high_now = float(row_at_alert["session_high"])
        session_low_now = float(row_at_alert["session_low"])

        # How far has MNQ moved from today's open? Positive = above open (green day).
        session_move_pts = alert.entry_price - day_open

        # Distance from today's session extremes at alert time.
        dist_from_high = session_high_now - alert.entry_price  # 0 = at session high
        dist_from_low = alert.entry_price - session_low_now  # 0 = at session low

        # minutes since market open (continuous — top feature last run)
        alert_time_mins = alert.alert_time.hour * 60 + alert.alert_time.minute
        alert_mins = alert_time_mins - (9 * 60 + 30)

        alert.features = {
            # Approach momentum (positive = moving toward line)
            "approach_momentum": approach_momentum,
            "approach_slope": approach_slope,
            "approach_first": approach_first,
            "approach_second": approach_second,
            "approach_accel": approach_accel,
            "norm_approach": norm_approach,
            # Approach quality
            "volatility": volatility,
            "max_pullback": max_pullback,
            # Volume / activity
            "volume_trend": volume_trend,
            "tick_rate": tick_rate,
            # Time of day (continuous only — buckets showed 0% importance)
            "time_of_day_mins": alert_mins,
            # Session / market context
            "session_move_pts": session_move_pts,
            "dist_from_high": dist_from_high,
            "dist_from_low": dist_from_low,
            # Level quality
            "level_test_count": alert.level_test_count,
            "prior_confluence": 1 if alert.prior_confluence else 0,
            # Alert context
            "entry_distance": abs(alert.entry_price - alert.line_price),
        }

    return alerts


# ── Model training ────────────────────────────────────────────────────────────


def build_model(all_alerts: list[Alert]) -> None:
    """
    Compare multiple classifiers, select the best by ROC-AUC, tune the
    decision threshold to maximize win rate (recall of 'incorrect' class),
    and attach cross-val predictions to alerts for the 'Correctly Avoided' table.
    """
    labeled = [
        a for a in all_alerts if a.outcome in ("correct", "incorrect") and a.features
    ]

    print(f"\n{'─' * 65}")
    print("  MODEL TRAINING")
    print(f"{'─' * 65}")

    n_correct = sum(1 for a in labeled if a.outcome == "correct")
    n_incorrect = sum(1 for a in labeled if a.outcome == "incorrect")
    print(f"  Samples: {len(labeled)}  ({n_correct} correct, {n_incorrect} incorrect)")

    if len(labeled) < 6 or min(n_correct, n_incorrect) < 2:
        print("  Insufficient samples for reliable model — skipping.")
        return

    X = pd.DataFrame([a.features for a in labeled])
    y = np.array([1 if a.outcome == "correct" else 0 for a in labeled])

    n_splits = min(5, min(n_correct, n_incorrect))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # ── Model comparison ──────────────────────────────────────────────────────
    # LogisticRegression needs scaling; tree models do not.
    # Conservative hyperparameters throughout — small dataset, avoid overfitting.
    candidates: dict[str, object] = {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.3,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=4,
            random_state=42,
        ),
    }

    print(f"\n  Model comparison ({n_splits}-fold cross-val, ROC-AUC):")
    best_auc = 0.0
    best_name = ""
    best_model = None

    for name, model in candidates.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        mean_auc, std_auc = scores.mean(), scores.std()
        flag = "  ← best" if mean_auc > best_auc else ""
        print(f"    {name:<25}  AUC {mean_auc:.3f} ± {std_auc:.3f}{flag}")
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_name = name
            best_model = model

    print(f"\n  Selected: {best_name} (AUC {best_auc:.3f})")
    if len(labeled) < 30:
        print(
            f"  Note: {len(labeled)} samples — treat results as directional, not definitive."
        )

    # ── Threshold optimisation ────────────────────────────────────────────────
    # Get cross-val probabilities (honest out-of-sample estimates).
    proba = cross_val_predict(best_model, X, y, cv=cv, method="predict_proba")
    p_incorrect = proba[:, 0]  # P(outcome = incorrect)

    # Show the win-rate trade-off across thresholds so you can pick your comfort level.
    # Lower threshold = more aggressive at flagging trades as incorrect (skip them).
    print(f"\n  Threshold sweep  (predict 'skip trade' when P(incorrect) > threshold):")
    print(
        f"  {'Threshold':>10}  {'Avoided':>8}  {'Missed good':>12}  {'Win rate':>10}  {'Trades taken':>13}"
    )
    print(f"  {'-'*10}  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*13}")

    best_wr = n_correct / (n_correct + n_incorrect)  # baseline (no model)
    best_threshold = 0.5
    for thr in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]:
        skip = p_incorrect > thr
        ca = int((skip & (y == 0)).sum())  # correctly avoided (bad trades skipped)
        ia = int((skip & (y == 1)).sum())  # incorrectly avoided (good trades skipped)
        rem_correct = n_correct - ia
        rem_incorrect = n_incorrect - ca
        total = rem_correct + rem_incorrect
        wr = rem_correct / total if total > 0 else 0.0
        taken = total
        print(f"  {thr:>10.2f}  {ca:>8}  {ia:>12}  {wr:>9.1%}  {taken:>13}")
        if (
            wr > best_wr and rem_correct >= n_correct * 0.6
        ):  # don't skip too many good trades
            best_wr = wr
            best_threshold = thr

    print(
        f"\n  Auto-selected threshold: {best_threshold:.2f}  "
        f"(maximises win rate while keeping ≥60% of correct trades)"
    )

    # Attach cv_pred using the chosen threshold.
    for alert, p_inc in zip(labeled, p_incorrect):
        alert.cv_pred = 0 if p_inc > best_threshold else 1

    # ── Feature importances ───────────────────────────────────────────────────
    best_model.fit(X, y)  # type: ignore[union-attr]
    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=X.columns)
    elif hasattr(best_model, "named_steps"):
        clf = best_model.named_steps["clf"]
        if hasattr(clf, "coef_"):
            imp = pd.Series(np.abs(clf.coef_[0]), index=X.columns)
        else:
            imp = pd.Series(np.zeros(len(X.columns)), index=X.columns)
    else:
        imp = pd.Series(np.zeros(len(X.columns)), index=X.columns)

    imp = imp.sort_values(ascending=False)
    print(f"\n  Feature importances (final model trained on all data):")
    for feat, val in imp.items():
        bar = "█" * max(1, int(val / imp.max() * 30))
        print(f"    {feat:<25} {val / imp.sum():>5.1%}  {bar}")

    # ── Feature means by outcome ──────────────────────────────────────────────
    top5 = list(imp.index[:5])
    correct_rows = X[y == 1]
    incorrect_rows = X[y == 0]
    print(f"\n  Top-5 feature means by outcome:")
    print(f"  {'Feature':<25}  {'Correct':>10}  {'Incorrect':>10}  {'Δ (C-I)':>10}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*10}")
    for feat in top5:
        c_mean = correct_rows[feat].mean()
        i_mean = incorrect_rows[feat].mean()
        delta = c_mean - i_mean
        print(f"  {feat:<25}  {c_mean:>10.3f}  {i_mean:>10.3f}  {delta:>+10.3f}")

    # ── Save model ────────────────────────────────────────────────────────────
    joblib.dump(
        {"model": best_model, "threshold": best_threshold, "features": list(X.columns)},
        MODEL_PATH,
    )
    print(f"\n  Model saved to {MODEL_PATH}")
    print(f"  Load with: joblib.load('{MODEL_PATH}')")

    # ── Classification report at chosen threshold ─────────────────────────────
    final_preds = (p_incorrect > best_threshold).astype(int)
    final_preds = (
        1 - final_preds
    )  # flip: 0=incorrect, 1=correct, but report expects 1=skip
    # Reframe: 1 = "take trade" (correct), 0 = "skip trade" (incorrect)
    print(f"\n  Classification report at threshold {best_threshold:.2f}:")
    print(
        classification_report(
            y,
            1 - (p_incorrect > best_threshold).astype(int),
            target_names=["incorrect (skip)", "correct (take)"],
            zero_division=0,
        )
    )


# ── Results table ─────────────────────────────────────────────────────────────


def print_results(all_alerts: list[Alert], days: list[datetime.date]) -> None:
    rows = []
    for date in days:
        day = [a for a in all_alerts if a.date == date]
        correct = sum(1 for a in day if a.outcome == "correct")
        incorrect = sum(1 for a in day if a.outcome == "incorrect")
        inconc = sum(1 for a in day if a.outcome == "inconclusive")
        avoided = sum(1 for a in day if a.outcome == "incorrect" and a.cv_pred == 0)
        rows.append(
            {
                "Date": str(date),
                "Alerts": len(day),
                "Correct": correct,
                "Incorrect": incorrect,
                "Inconclusive": inconc,
                "Correctly Avoided": avoided,
            }
        )

    df = pd.DataFrame(rows)

    # Add Win% column (correct / decided, blank if no decided alerts).
    def win_pct(row: pd.Series) -> str:
        decided = row["Correct"] + row["Incorrect"]
        if decided == 0:
            return "  —"
        return f"{row['Correct'] / decided:.0%}"

    df["Win%"] = df.apply(win_pct, axis=1)

    total_decided_total = int(df["Correct"].sum() + df["Incorrect"].sum())
    total_row = {
        "Date": "TOTAL",
        "Alerts": df["Alerts"].sum(),
        "Correct": df["Correct"].sum(),
        "Incorrect": df["Incorrect"].sum(),
        "Inconclusive": df["Inconclusive"].sum(),
        "Correctly Avoided": df["Correctly Avoided"].sum(),
        "Win%": (
            f"{df['Correct'].sum() / total_decided_total:.0%}"
            if total_decided_total
            else "—"
        ),
    }
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    n_days = len([r for r in rows if r["Correct"] + r["Incorrect"] > 0])
    print(f"\n{'═' * 90}")
    print(
        f"  BACKTEST RESULTS  ({days[0]} → {days[-1]},  {n_days} active trading days)"
    )
    print(f"{'═' * 90}")
    print(df.to_string(index=False))

    total_correct = int(total_row["Correct"])
    total_incorrect = int(total_row["Incorrect"])
    total_avoided = int(total_row["Correctly Avoided"])
    total_decided = total_correct + total_incorrect

    # Good trades the model incorrectly told you to skip.
    good_skipped = sum(
        1 for a in all_alerts if a.outcome == "correct" and a.cv_pred == 0
    )

    print(f"\n{'─' * 65}")
    if total_decided > 0:
        raw_rate = total_correct / total_decided
        print(
            f"  Win rate — raw (no model)    : {raw_rate:.1%}  "
            f"({total_correct}W / {total_incorrect}L)"
        )

        # Subtract both avoided bad trades AND wrongly skipped good trades.
        adj_correct = total_correct - good_skipped
        adj_incorrect = total_incorrect - total_avoided
        adj_decided = adj_correct + adj_incorrect
        adj_rate = adj_correct / adj_decided if adj_decided > 0 else 0.0
        print(
            f"  Win rate — model-filtered    : {adj_rate:.1%}  "
            f"({adj_correct}W / {adj_incorrect}L  |  "
            f"avoided {total_avoided} bad, skipped {good_skipped} good)"
        )
        print(
            f"\n  'Correctly Avoided' uses cross-val predictions — no in-sample overfitting."
        )
    print(f"{'─' * 65}")

    # ── Breakdown analysis ────────────────────────────────────────────────────
    # Shows raw win rates by key groupings so we can decide which hard filters
    # or feature changes are worth making — without running another backtest.
    decided = [a for a in all_alerts if a.outcome in ("correct", "incorrect")]
    if not decided:
        return

    def win_rate_table(groups: list[tuple[str, list[Alert]]]) -> None:
        print(f"  {'Group':<28}  {'W':>5}  {'L':>5}  {'Total':>7}  {'Win%':>6}")
        print(f"  {'-'*28}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*6}")
        for label, alerts in groups:
            w = sum(1 for a in alerts if a.outcome == "correct")
            l = sum(1 for a in alerts if a.outcome == "incorrect")
            t = w + l
            wr = w / t if t > 0 else 0.0
            warn = "  ⚠ n<30" if 0 < t < 30 else ""
            print(f"  {label:<28}  {w:>5}  {l:>5}  {t:>7}  {wr:>5.1%}{warn}")

    # By time bucket (using features dict; fall back to alert_time if no features)
    def time_bucket(a: Alert) -> str:
        mins = a.alert_time.hour * 60 + a.alert_time.minute
        if (10 * 60 + 30) <= mins < (11 * 60 + 30):
            return "10:30–11:30 ET (first hour)"
        elif (11 * 60 + 30) <= mins < (13 * 60):
            return "11:30–13:00 ET (lunch)"
        elif (13 * 60) <= mins < (15 * 60):
            return "13:00–15:00 ET (afternoon)"
        elif (15 * 60) <= mins < (16 * 60):
            return "15:00–16:00 ET (power hour)"
        else:
            return "other"

    print(f"\n{'─' * 55}")
    print("  WIN RATE BY TIME OF DAY")
    print(f"{'─' * 55}")
    buckets = [
        "10:30–11:30 ET (first hour)",
        "11:30–13:00 ET (lunch)",
        "13:00–15:00 ET (afternoon)",
        "15:00–16:00 ET (power hour)",
    ]
    win_rate_table([(b, [a for a in decided if time_bucket(a) == b]) for b in buckets])

    print(f"\n{'─' * 55}")
    print("  WIN RATE BY LEVEL")
    print(f"{'─' * 55}")
    win_rate_table(
        [
            (lvl, [a for a in decided if a.level == lvl])
            for lvl in ["IBH", "IBL", "VWAP"]
        ]
    )

    print(f"\n{'─' * 55}")
    print("  WIN RATE BY LEVEL TEST COUNT (how many times zone entered today)")
    print(f"{'─' * 55}")
    max_count = max((a.level_test_count for a in decided), default=1)
    groups = []
    for n in range(1, min(max_count + 1, 6)):
        label = f"Test #{n}" if n < 5 else "Test #5+"
        subset = [
            a
            for a in decided
            if (a.level_test_count == n if n < 5 else a.level_test_count >= 5)
        ]
        groups.append((label, subset))
    win_rate_table(groups)

    print(f"\n{'─' * 55}")
    print("  WIN RATE BY DAY DIRECTION (MNQ vs open at alert time)")
    print(f"{'─' * 55}")
    green_day = [a for a in decided if a.features.get("session_move_pts", 0) > 0]
    red_day = [a for a in decided if a.features.get("session_move_pts", 0) <= 0]
    win_rate_table(
        [
            ("Green day (price above open)", green_day),
            ("Red day (price at/below open)", red_day),
        ]
    )

    print(f"\n{'─' * 55}")
    print("  WIN RATE BY DIRECTION")
    print(f"{'─' * 55}")
    win_rate_table(
        [
            (
                "BUY  (price above line → support)",
                [a for a in decided if a.direction == "up"],
            ),
            (
                "SELL (price below line → resistance)",
                [a for a in decided if a.direction == "down"],
            ),
        ]
    )

    print(f"\n{'─' * 55}")
    print("  WIN RATE BY ENTRY DISTANCE (how far inside the 7-pt zone)")
    print(f"{'─' * 55}")
    win_rate_table(
        [
            (
                "0–5 pts from line (close)",
                [a for a in decided if abs(a.entry_price - a.line_price) <= 5],
            ),
            (
                "5–7 pts from line (outer edge)",
                [a for a in decided if abs(a.entry_price - a.line_price) > 5],
            ),
        ]
    )

    print(f"\n{'─' * 55}")
    print(
        f"  WIN RATE BY PRIOR DAY CONFLUENCE (line within {CONFLUENCE_PTS:.0f} pts of prior IBH/IBL/close)"
    )
    print(f"{'─' * 55}")
    win_rate_table(
        [
            ("Confluence with prior level", [a for a in decided if a.prior_confluence]),
            (
                "No prior level confluence",
                [a for a in decided if not a.prior_confluence],
            ),
        ]
    )

    print(f"\n{'─' * 55}")
    print("  WIN RATE BY APPROACH STRENGTH (norm_approach feature, top/bottom half)")
    print(f"{'─' * 55}")
    featured = [a for a in decided if a.features]
    if featured:
        median_approach = sorted(a.features.get("norm_approach", 0) for a in featured)[
            len(featured) // 2
        ]
        win_rate_table(
            [
                (
                    "Strong approach (top half)",
                    [
                        a
                        for a in featured
                        if a.features.get("norm_approach", 0) >= median_approach
                    ],
                ),
                (
                    "Weak approach (bottom half)",
                    [
                        a
                        for a in featured
                        if a.features.get("norm_approach", 0) < median_approach
                    ],
                ),
            ]
        )

    print(f"\n{'─' * 55}")
    print("  WIN RATE BY DAY OF WEEK")
    print(f"{'─' * 55}")
    win_rate_table(
        [
            ("Monday", [a for a in decided if a.alert_time.weekday() == 0]),
            ("Tuesday", [a for a in decided if a.alert_time.weekday() == 1]),
            ("Wednesday", [a for a in decided if a.alert_time.weekday() == 2]),
            ("Thursday", [a for a in decided if a.alert_time.weekday() == 3]),
            ("Friday", [a for a in decided if a.alert_time.weekday() == 4]),
        ]
    )

    print(f"\n{'─' * 55}")
    print("  WIN RATE BY VWAP DIRECTION (VWAP alerts only)")
    print(f"{'─' * 55}")
    vwap = [a for a in decided if a.level == "VWAP"]
    win_rate_table(
        [
            (
                "VWAP BUY  (price above VWAP → support)",
                [a for a in vwap if a.direction == "up"],
            ),
            (
                "VWAP SELL (price below VWAP → resistance)",
                [a for a in vwap if a.direction == "down"],
            ),
        ]
    )

    print(f"\n{'─' * 55}")
    print("  WIN RATE BY SESSION MOVE MAGNITUDE (MNQ pts from open at alert time)")
    print(f"{'─' * 55}")

    def session_bucket(a: Alert) -> str:
        m = a.features.get("session_move_pts", 0)
        if m > 50:
            return "Strongly green  (>+50 pts)"
        elif m > 0:
            return "Mildly green    (0 to +50 pts)"
        elif m > -50:
            return "Mildly red      (0 to -50 pts)"
        else:
            return "Strongly red    (<-50 pts)"

    buckets_session = [
        "Strongly green  (>+50 pts)",
        "Mildly green    (0 to +50 pts)",
        "Mildly red      (0 to -50 pts)",
        "Strongly red    (<-50 pts)",
    ]
    win_rate_table(
        [
            (b, [a for a in decided if a.features and session_bucket(a) == b])
            for b in buckets_session
        ]
    )

    # ── NEW TEST 1: Filter stacking analysis ─────────────────────────────────
    # Shows the cumulative effect of adding each filter.
    print(f"\n{'═' * 70}")
    print("  FILTER STACKING ANALYSIS")
    print(f"  (What happens to win rate as we add each filter?)")
    print(f"{'═' * 70}")

    all_decided = [a for a in all_alerts if a.outcome in ("correct", "incorrect")]

    filters = [
        ("Baseline (all alerts)", lambda a: True),
        ("+ Skip first test (#1)", lambda a: a.level_test_count >= 2),
        (
            "+ Skip first hour (10:30-11:30)",
            lambda a: a.level_test_count >= 2
            and not (
                (10 * 60 + 30)
                <= a.alert_time.hour * 60 + a.alert_time.minute
                < (11 * 60 + 30)
            ),
        ),
        (
            "+ Skip power hour (15:00-16:00)",
            lambda a: a.level_test_count >= 2
            and not (
                (10 * 60 + 30)
                <= a.alert_time.hour * 60 + a.alert_time.minute
                < (11 * 60 + 30)
            )
            and not (
                (15 * 60) <= a.alert_time.hour * 60 + a.alert_time.minute < (16 * 60)
            ),
        ),
        (
            "+ Skip test #2 (keep #3+)",
            lambda a: a.level_test_count >= 3
            and not (
                (10 * 60 + 30)
                <= a.alert_time.hour * 60 + a.alert_time.minute
                < (11 * 60 + 30)
            )
            and not (
                (15 * 60) <= a.alert_time.hour * 60 + a.alert_time.minute < (16 * 60)
            ),
        ),
    ]

    print(f"  {'Filter':<38}  {'W':>5}  {'L':>5}  {'Total':>7}  {'Win%':>6}")
    print(f"  {'-'*38}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*6}")
    for label, fn in filters:
        subset = [a for a in all_decided if fn(a)]
        w = sum(1 for a in subset if a.outcome == "correct")
        l = sum(1 for a in subset if a.outcome == "incorrect")
        t = w + l
        wr = w / t if t > 0 else 0.0
        print(f"  {label:<38}  {w:>5}  {l:>5}  {t:>7}  {wr:>5.1%}")

    # ── NEW TEST 2: Tick rate analysis ────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  TICK RATE ANALYSIS")
    print(f"  (Higher tick rate = more market activity during approach)")
    print(f"{'═' * 70}")

    with_tick = [a for a in decided if a.features and "tick_rate" in a.features]
    if with_tick:
        rates = sorted(a.features["tick_rate"] for a in with_tick)
        q25 = rates[len(rates) // 4]
        q50 = rates[len(rates) // 2]
        q75 = rates[3 * len(rates) // 4]
        win_rate_table(
            [
                (
                    f"Q1: tick_rate < {q25:.0f}",
                    [a for a in with_tick if a.features["tick_rate"] < q25],
                ),
                (
                    f"Q2: {q25:.0f} – {q50:.0f}",
                    [a for a in with_tick if q25 <= a.features["tick_rate"] < q50],
                ),
                (
                    f"Q3: {q50:.0f} – {q75:.0f}",
                    [a for a in with_tick if q50 <= a.features["tick_rate"] < q75],
                ),
                (
                    f"Q4: tick_rate ≥ {q75:.0f}",
                    [a for a in with_tick if a.features["tick_rate"] >= q75],
                ),
            ]
        )

        # Sweep minimum tick rate thresholds.
        print(f"\n  Minimum tick rate filter sweep:")
        print(
            f"  {'Min tick_rate':>14}  {'W':>5}  {'L':>5}  {'Total':>7}  {'Win%':>6}  {'Removed':>8}"
        )
        print(f"  {'-'*14}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*8}")
        for threshold in [0, 500, 750, 1000, 1250, 1500, 1750, 2000]:
            subset = [a for a in with_tick if a.features["tick_rate"] >= threshold]
            w = sum(1 for a in subset if a.outcome == "correct")
            l = sum(1 for a in subset if a.outcome == "incorrect")
            t = w + l
            wr = w / t if t > 0 else 0.0
            removed = len(with_tick) - len(subset)
            warn = "  ⚠ n<30" if 0 < t < 30 else ""
            print(
                f"  {threshold:>14}  {w:>5}  {l:>5}  {t:>7}  {wr:>5.1%}  {removed:>8}{warn}"
            )

    # ── NEW TEST 3: Trend alignment ──────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  TREND ALIGNMENT ANALYSIS")
    print(f"  (Does trading WITH the session trend beat trading AGAINST it?)")
    print(f"{'═' * 70}")

    with_features = [a for a in decided if a.features]
    if with_features:

        def trend_aligned(a: Alert) -> bool:
            """True if trade direction matches session direction (with-trend)."""
            session_move = a.features.get("session_move_pts", 0)
            if a.direction == "up" and session_move > 0:
                return True  # BUY on green day = with trend
            if a.direction == "down" and session_move < 0:
                return True  # SELL on red day = with trend
            return False

        aligned = [a for a in with_features if trend_aligned(a)]
        counter = [a for a in with_features if not trend_aligned(a)]
        win_rate_table(
            [
                ("With trend", aligned),
                ("Against trend", counter),
            ]
        )

    # ── NEW TEST 4: Multi-factor cross-tabs ──────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  CROSS-TAB: TIME OF DAY × TEST COUNT")
    print(f"{'═' * 70}")

    time_labels = ["11:30–13:00 ET (lunch)", "13:00–15:00 ET (afternoon)"]
    test_labels = ["Test #2", "Test #3+"]
    cross_groups = []
    for tl in time_labels:
        for tc in test_labels:
            subset = [
                a
                for a in decided
                if time_bucket(a) == tl
                and (
                    a.level_test_count == 2
                    if tc == "Test #2"
                    else a.level_test_count >= 3
                )
            ]
            cross_groups.append((f"{tl.split(' ET')[0]} / {tc}", subset))
    win_rate_table(cross_groups)

    print(f"\n{'═' * 70}")
    print("  CROSS-TAB: TIME OF DAY × TREND ALIGNMENT")
    print(f"{'═' * 70}")

    if with_features:
        cross_groups2 = []
        for tl in time_labels:
            for trend_label, trend_fn in [
                ("with trend", trend_aligned),
                ("against trend", lambda a: not trend_aligned(a)),
            ]:
                subset = [
                    a for a in with_features if time_bucket(a) == tl and trend_fn(a)
                ]
                cross_groups2.append((f"{tl.split(' ET')[0]} / {trend_label}", subset))
        win_rate_table(cross_groups2)

    print(f"\n{'═' * 70}")
    print("  CROSS-TAB: LEVEL × TIME OF DAY")
    print(f"{'═' * 70}")

    cross_groups3 = []
    for lvl in ["IBH", "IBL", "VWAP"]:
        for tl in time_labels:
            subset = [a for a in decided if a.level == lvl and time_bucket(a) == tl]
            cross_groups3.append((f"{lvl} / {tl.split(' ET')[0]}", subset))
    win_rate_table(cross_groups3)

    print(f"\n{'═' * 70}")
    print("  CROSS-TAB: LEVEL × TREND ALIGNMENT")
    print(f"{'═' * 70}")

    if with_features:
        cross_groups4 = []
        for lvl in ["IBH", "IBL", "VWAP"]:
            for trend_label, trend_fn in [
                ("with trend", trend_aligned),
                ("against trend", lambda a: not trend_aligned(a)),
            ]:
                subset = [a for a in with_features if a.level == lvl and trend_fn(a)]
                cross_groups4.append((f"{lvl} / {trend_label}", subset))
        win_rate_table(cross_groups4)


# ── Parameter sweep ──────────────────────────────────────────────────────────


def parameter_sweep(
    all_alerts: list[Alert],
    day_dfs: dict[datetime.date, pd.DataFrame],
) -> None:
    """Re-evaluate outcomes with different target/window combinations."""
    combos = [
        (8, 10 * 60, "+8 pts / 10 min"),
        (8, 15 * 60, "+8 pts / 15 min (current)"),
        (10, 10 * 60, "+10 pts / 10 min"),
        (10, 15 * 60, "+10 pts / 15 min"),
        (10, 20 * 60, "+10 pts / 20 min"),
        (12, 15 * 60, "+12 pts / 15 min"),
        (12, 20 * 60, "+12 pts / 20 min"),
        (15, 15 * 60, "+15 pts / 15 min"),
        (15, 20 * 60, "+15 pts / 20 min"),
        (20, 15 * 60, "+20 pts / 15 min"),
        (20, 20 * 60, "+20 pts / 20 min"),
        (20, 30 * 60, "+20 pts / 30 min"),
        (25, 20 * 60, "+25 pts / 20 min"),
        (25, 30 * 60, "+25 pts / 30 min"),
    ]

    print(f"\n{'═' * 70}")
    print("  PARAMETER SWEEP: TARGET × WINDOW")
    print(f"  (Re-evaluates outcomes for every alert under different parameters)")
    print(f"{'═' * 70}")
    print(
        f"  {'Params':<30}  {'W':>5}  {'L':>5}  {'Inc':>5}  {'Total':>7}  {'Win%':>6}  {'EV/trade':>9}"
    )
    print(f"  {'-'*30}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*9}")

    for target, window, label in combos:
        correct = incorrect = inconclusive = 0

        for alert in all_alerts:
            df = day_dfs.get(alert.date)
            if df is None:
                continue

            prices = df["price"]
            alert_ts = pd.Timestamp(alert.alert_time)
            window_end = alert_ts + pd.Timedelta(
                seconds=WINDOW_SECS
            )  # use original window for hit phase

            # Phase 1: did price touch the line within original 15 min?
            hit_seg = prices[(prices.index > alert_ts) & (prices.index <= window_end)]
            hit_mask = abs(hit_seg - alert.line_price) <= HIT_THRESHOLD
            if not hit_mask.any():
                inconclusive += 1
                continue

            hit_ts = hit_mask.idxmax()

            # Phase 2: re-evaluate with the sweep parameters.
            # Check if target or stop (20 pts) is hit first.
            eval_end = hit_ts + pd.Timedelta(seconds=window)
            eval_seg = prices[(prices.index > hit_ts) & (prices.index <= eval_end)]

            if alert.direction == "up":
                target_mask = eval_seg >= alert.line_price + target
                stop_mask = eval_seg <= alert.line_price - STOP_POINTS
            else:
                target_mask = eval_seg <= alert.line_price - target
                stop_mask = eval_seg >= alert.line_price + STOP_POINTS

            target_hit = target_mask.any()
            stop_hit = stop_mask.any()

            if target_hit and stop_hit:
                target_ts = eval_seg.index[target_mask][0]
                stop_ts = eval_seg.index[stop_mask][0]
                if target_ts <= stop_ts:
                    correct += 1
                else:
                    incorrect += 1
            elif target_hit:
                correct += 1
            else:
                incorrect += 1

        decided = correct + incorrect
        wr = correct / decided if decided > 0 else 0.0
        ev = wr * target - (1 - wr) * STOP_POINTS if decided > 0 else 0.0
        marker = " ← current" if "(current)" in label else ""
        print(
            f"  {label:<30}  {correct:>5}  {incorrect:>5}  {inconclusive:>5}  "
            f"{decided:>7}  {wr:>5.1%}  {ev:>+8.1f}{marker}"
        )


# ── Stop loss sweep ──────────────────────────────────────────────────────────


def stop_loss_sweep(
    all_alerts: list[Alert],
    day_dfs: dict[datetime.date, pd.DataFrame],
) -> None:
    """Sweep different stop loss distances to see impact on win rate."""
    print(f"\n{'═' * 70}")
    print("  STOP LOSS SWEEP")
    print(
        f"  (Fixed +{TARGET_POINTS} pt target / {WINDOW_SECS // 60} min window, varying stop distance)"
    )
    print(f"{'═' * 70}")
    print(
        f"  {'Stop (pts)':<12}  {'W':>5}  {'L':>5}  {'Total':>7}  {'Win%':>6}  {'Avg P/L':>8}"
    )
    print(f"  {'-'*12}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*8}")

    for stop in [8, 10, 12, 15, 20, 25, 30, 999]:
        correct = incorrect = 0
        for alert in all_alerts:
            df = day_dfs.get(alert.date)
            if df is None:
                continue
            prices = df["price"]
            alert_ts = pd.Timestamp(alert.alert_time)
            window_end = alert_ts + pd.Timedelta(seconds=WINDOW_SECS)

            hit_seg = prices[(prices.index > alert_ts) & (prices.index <= window_end)]
            hit_mask = abs(hit_seg - alert.line_price) <= HIT_THRESHOLD
            if not hit_mask.any():
                continue

            hit_ts = hit_mask.idxmax()
            eval_end = hit_ts + pd.Timedelta(seconds=WINDOW_SECS)
            eval_seg = prices[(prices.index > hit_ts) & (prices.index <= eval_end)]

            if alert.direction == "up":
                target_mask = eval_seg >= alert.line_price + TARGET_POINTS
                stop_mask = eval_seg <= alert.line_price - stop
            else:
                target_mask = eval_seg <= alert.line_price - TARGET_POINTS
                stop_mask = eval_seg >= alert.line_price + stop

            target_hit = target_mask.any()
            stop_hit = stop_mask.any()

            if target_hit and stop_hit:
                if eval_seg.index[target_mask][0] <= eval_seg.index[stop_mask][0]:
                    correct += 1
                else:
                    incorrect += 1
            elif target_hit:
                correct += 1
            else:
                incorrect += 1

        decided = correct + incorrect
        wr = correct / decided if decided > 0 else 0.0
        # Expected value per trade: win_rate * target - (1 - win_rate) * stop
        effective_stop = stop if stop < 999 else 0
        ev = wr * TARGET_POINTS - (1 - wr) * effective_stop if stop < 999 else 0
        stop_label = f"-{stop} pts" if stop < 999 else "No stop"
        marker = " ← current" if stop == STOP_POINTS else ""
        print(
            f"  {stop_label:<12}  {correct:>5}  {incorrect:>5}  {decided:>7}  "
            f"{wr:>5.1%}  {ev:>+7.1f}{marker}"
        )


# ── Combinatorial filter sweep ──────────────────────────────────────────────


def combinatorial_filter_sweep(all_alerts: list[Alert]) -> None:
    """Test every combination of filters and find the best that keeps ≥ 50 trades."""
    # All alerts already have first-test and first-hour filtered out by simulate_and_evaluate.
    # We sweep additional filters on top of that.
    decided = [a for a in all_alerts if a.outcome in ("correct", "incorrect")]

    time_filters = {
        "all_times": lambda a: True,
        "no_lunch": lambda a: not (
            (11 * 60 + 30) <= a.alert_time.hour * 60 + a.alert_time.minute < (13 * 60)
        ),
        "afternoon_only": lambda a: (13 * 60)
        <= a.alert_time.hour * 60 + a.alert_time.minute
        < (15 * 60),
        "no_power_hr": lambda a: not (
            (15 * 60) <= a.alert_time.hour * 60 + a.alert_time.minute < (16 * 60)
        ),
        "lunch+afternoon": lambda a: (11 * 60 + 30)
        <= a.alert_time.hour * 60 + a.alert_time.minute
        < (15 * 60),
    }

    level_filters = {
        "all_levels": lambda a: True,
        "IBL_only": lambda a: a.level == "IBL",
        "no_IBH": lambda a: a.level != "IBH",
        "IB_only": lambda a: a.level in ("IBH", "IBL"),
        "VWAP_only": lambda a: a.level == "VWAP",
    }

    tick_filters = {
        "any_tick": lambda a: True,
        "tick≥1000": lambda a: a.features.get("tick_rate", 0) >= 1000,
        "tick≥1500": lambda a: a.features.get("tick_rate", 0) >= 1500,
        "tick≥1750": lambda a: a.features.get("tick_rate", 0) >= 1750,
        "tick≥2000": lambda a: a.features.get("tick_rate", 0) >= 2000,
    }

    test_count_filters = {
        "test≥2": lambda a: a.level_test_count >= 2,
        "test≥3": lambda a: a.level_test_count >= 3,
        "test=3-4": lambda a: 3 <= a.level_test_count <= 4,
    }

    session_filters = {
        "any_session": lambda a: True,
        "mildly_red": lambda a: -50 < a.features.get("session_move_pts", 0) <= 0,
        "not_strong_green": lambda a: a.features.get("session_move_pts", 0) <= 50,
    }

    print(f"\n{'═' * 90}")
    print("  COMBINATORIAL FILTER SWEEP")
    print(
        f"  (Testing all filter combinations, showing top 25 by win rate with ≥ 50 trades)"
    )
    print(f"{'═' * 90}")

    results: list[tuple[str, int, int, int, float]] = []

    for t_name, t_fn in time_filters.items():
        for l_name, l_fn in level_filters.items():
            for tk_name, tk_fn in tick_filters.items():
                for tc_name, tc_fn in test_count_filters.items():
                    for s_name, s_fn in session_filters.items():
                        subset = [
                            a
                            for a in decided
                            if t_fn(a) and l_fn(a) and tk_fn(a) and tc_fn(a) and s_fn(a)
                        ]
                        w = sum(1 for a in subset if a.outcome == "correct")
                        l = sum(1 for a in subset if a.outcome == "incorrect")
                        t = w + l
                        if t < 50:
                            continue
                        wr = w / t
                        combo_name = (
                            f"{t_name} | {l_name} | {tk_name} | {tc_name} | {s_name}"
                        )
                        results.append((combo_name, w, l, t, wr))

    results.sort(key=lambda x: (-x[4], -x[3]))

    print(f"  {'Filters':<75}  {'W':>4}  {'L':>4}  {'N':>5}  {'Win%':>6}")
    print(f"  {'-'*75}  {'-'*4}  {'-'*4}  {'-'*5}  {'-'*6}")
    for combo_name, w, l, t, wr in results[:25]:
        print(f"  {combo_name:<75}  {w:>4}  {l:>4}  {t:>5}  {wr:>5.1%}")

    if results:
        best = results[0]
        print(f"\n  Best combo: {best[0]}")
        print(f"  → {best[1]}W / {best[2]}L = {best[4]:.1%}  ({best[3]} trades)")


# ── Level-specific strategies ───────────────────────────────────────────────


def level_specific_analysis(all_alerts: list[Alert]) -> None:
    """Apply different filter thresholds per level to maximize overall win rate."""
    decided = [a for a in all_alerts if a.outcome in ("correct", "incorrect")]

    print(f"\n{'═' * 70}")
    print("  LEVEL-SPECIFIC STRATEGY ANALYSIS")
    print(f"  (Different filters per level — IBL is strong, IBH needs help)")
    print(f"{'═' * 70}")

    def time_mins(a: Alert) -> int:
        return a.alert_time.hour * 60 + a.alert_time.minute

    # Define per-level filter options to try.
    level_options: dict[str, list[tuple[str, object]]] = {
        "IBH": [
            ("IBH: all", lambda a: True),
            ("IBH: afternoon only", lambda a: (13 * 60) <= time_mins(a) < (15 * 60)),
            ("IBH: tick≥1750", lambda a: a.features.get("tick_rate", 0) >= 1750),
            ("IBH: test≥3", lambda a: a.level_test_count >= 3),
            (
                "IBH: aftn+tick≥1750",
                lambda a: (13 * 60) <= time_mins(a) < (15 * 60)
                and a.features.get("tick_rate", 0) >= 1750,
            ),
            ("IBH: disabled", lambda a: False),
        ],
        "IBL": [
            ("IBL: all", lambda a: True),
            ("IBL: tick≥1000", lambda a: a.features.get("tick_rate", 0) >= 1000),
            ("IBL: tick≥1750", lambda a: a.features.get("tick_rate", 0) >= 1750),
        ],
        "VWAP": [
            ("VWAP: all", lambda a: True),
            ("VWAP: afternoon only", lambda a: (13 * 60) <= time_mins(a) < (15 * 60)),
            ("VWAP: tick≥1750", lambda a: a.features.get("tick_rate", 0) >= 1750),
            (
                "VWAP: no_lunch",
                lambda a: not ((11 * 60 + 30) <= time_mins(a) < (13 * 60)),
            ),
            (
                "VWAP: aftn+tick≥1750",
                lambda a: (13 * 60) <= time_mins(a) < (15 * 60)
                and a.features.get("tick_rate", 0) >= 1750,
            ),
        ],
    }

    # Try every combination of per-level filters.
    results: list[tuple[str, int, int, int, float]] = []

    for ibh_label, ibh_fn in level_options["IBH"]:
        for ibl_label, ibl_fn in level_options["IBL"]:
            for vwap_label, vwap_fn in level_options["VWAP"]:
                subset = []
                for a in decided:
                    if a.level == "IBH" and ibh_fn(a):
                        subset.append(a)
                    elif a.level == "IBL" and ibl_fn(a):
                        subset.append(a)
                    elif a.level == "VWAP" and vwap_fn(a):
                        subset.append(a)

                w = sum(1 for a in subset if a.outcome == "correct")
                l = sum(1 for a in subset if a.outcome == "incorrect")
                t = w + l
                if t < 30:
                    continue
                wr = w / t
                combo = f"{ibh_label} + {ibl_label} + {vwap_label}"
                results.append((combo, w, l, t, wr))

    results.sort(key=lambda x: (-x[4], -x[3]))

    print(f"\n  Top 20 level-specific combos (min 30 trades):")
    print(f"  {'Strategy':<70}  {'W':>4}  {'L':>4}  {'N':>5}  {'Win%':>6}")
    print(f"  {'-'*70}  {'-'*4}  {'-'*4}  {'-'*5}  {'-'*6}")
    for combo, w, l, t, wr in results[:20]:
        print(f"  {combo:<70}  {w:>4}  {l:>4}  {t:>5}  {wr:>5.1%}")

    if results:
        best = results[0]
        print(f"\n  Best level-specific strategy: {best[0]}")
        print(f"  → {best[1]}W / {best[2]}L = {best[4]:.1%}  ({best[3]} trades)")


# ── Composite scoring ───────────────────────────────────────────────────────


def composite_scoring(all_alerts: list[Alert]) -> None:
    """Assign a composite score to each alert and sweep cutoffs."""
    decided = [
        a for a in all_alerts if a.outcome in ("correct", "incorrect") and a.features
    ]

    print(f"\n{'═' * 70}")
    print("  COMPOSITE SCORING ANALYSIS")
    print(f"  (Score each alert on multiple factors, sweep cutoffs)")
    print(f"{'═' * 70}")

    # Score components — must match _composite_score() in alert_manager.py exactly.
    def score_alert(a: Alert) -> float:
        s = 0.0
        mins = a.alert_time.hour * 60 + a.alert_time.minute

        # Level: IBL is strongest (+3), IBH weakest (-1)
        if a.level == "IBL":
            s += 3
        elif a.level == "IBH":
            s -= 1

        # Direction × Level interaction
        combo = (a.level, a.direction)
        if combo in (
            ("FIB_EXT_HI_1.272", "up"),
            ("FIB_EXT_LO_1.272", "down"),
            ("IBL", "down"),
        ):
            s += 1
        elif combo in (("IBH", "up"),):
            s -= 1

        # Time of day: afternoon best (+2), first hour worst (-3), lunch -1
        if (13 * 60) <= mins < (15 * 60):
            s += 2
        elif (10 * 60 + 30) <= mins < (11 * 60 + 30):
            s -= 3  # first hour post-IB (was hard-filtered, now scored)
        elif (11 * 60 + 30) <= mins < (13 * 60):
            s -= 1  # lunch
        else:
            s += 1  # power hour

        # Tick rate: higher is better
        tr = a.features.get("tick_rate", 0)
        if tr >= 2000:
            s += 2
        elif tr >= 1750:
            s += 1
        elif tr < 1000:
            s -= 2

        # Test count: first test heavily penalized, #3 is sweet spot,
        # heavy decay after #4 to prevent flooding from excessive retests.
        tc = a.level_test_count
        if tc == 1:
            s -= 4  # first test (was hard-filtered, now scored)
        elif tc == 3:
            s += 2
        elif tc == 4:
            s += 1
        elif tc == 5:
            s -= 2
        elif tc >= 6:
            s -= 4  # effectively kills alerts past 5th retest

        # Session context: mildly red is best
        session_move = a.features.get("session_move_pts", 0)
        if -50 < session_move <= 0:
            s += 2
        elif session_move > 50:
            s -= 1

        return s

    # Score all alerts and sweep thresholds.
    scored = [(a, score_alert(a)) for a in decided]
    all_scores = sorted(set(s for _, s in scored))

    print(f"\n  Score distribution:")
    print(f"  {'Score':>6}  {'W':>5}  {'L':>5}  {'N':>5}  {'Win%':>6}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*6}")
    for sc in all_scores:
        at_score = [a for a, s in scored if s == sc]
        w = sum(1 for a in at_score if a.outcome == "correct")
        l = sum(1 for a in at_score if a.outcome == "incorrect")
        t = w + l
        wr = w / t if t > 0 else 0.0
        print(f"  {sc:>6.0f}  {w:>5}  {l:>5}  {t:>5}  {wr:>5.1%}")

    print(f"\n  Cutoff sweep (only take alerts with score ≥ cutoff):")
    print(
        f"  {'Cutoff':>7}  {'W':>5}  {'L':>5}  {'Total':>7}  {'Win%':>6}  {'EV/trade':>9}"
    )
    print(f"  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*9}")

    for cutoff in sorted(all_scores):
        above = [a for a, s in scored if s >= cutoff]
        w = sum(1 for a in above if a.outcome == "correct")
        l = sum(1 for a in above if a.outcome == "incorrect")
        t = w + l
        if t < 10:
            continue
        wr = w / t
        ev = wr * TARGET_POINTS - (1 - wr) * STOP_POINTS
        print(f"  {cutoff:>7.0f}  {w:>5}  {l:>5}  {t:>7}  {wr:>5.1%}  {ev:>+8.1f}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    # --days N    : number of trading days to backtest (default 45)
    # --offset N  : skip N trading days back before starting (default 0)
    # Examples:
    #   python backtest.py                      # most recent 45 days
    #   python backtest.py --offset 45          # prior 45 days
    #   python backtest.py --days 90            # full 90-day combined run
    #   python backtest.py --days 90 --offset 45  # 90 days starting 45 days back
    n_days = 45
    if "--days" in sys.argv:
        n_days = int(sys.argv[sys.argv.index("--days") + 1])

    offset = 0
    if "--offset" in sys.argv:
        offset = int(sys.argv[sys.argv.index("--offset") + 1])

    days = get_trading_days(n_days, offset=offset)
    period = f"(offset {offset}d)" if offset else "(most recent)"
    print(f"{'═' * 65}")
    print(f"  MNQ Backtest  |  {days[0]} → {days[-1]}  {period}")
    print(f"  Alert threshold : ±{ALERT_THRESHOLD} pts")
    print(
        f"  Target          : +{TARGET_POINTS} pts from line within {WINDOW_SECS // 60} min"
    )
    print(
        f"  Incorrect       : line touched but target not reached within {WINDOW_SECS // 60} min"
    )
    print(
        f"  Inconclusive    : line never touched within {WINDOW_SECS // 60} min of alert"
    )
    print(f"{'═' * 65}\n")

    client: db.Historical = db.Historical(key=DATABENTO_API_KEY)
    all_alerts: list[Alert] = []
    day_dfs: dict[datetime.date, pd.DataFrame] = {}  # for parameter sweep

    cum_correct = cum_incorrect = cum_inconc = 0
    prior_levels: list[float] = []  # IBH, IBL, close from the previous session

    for day_num, date in enumerate(days, 1):
        print(f"\n{'─' * 65}")
        print(f"  Day {day_num}/{len(days)}  |  {date}")
        print(f"{'─' * 65}")

        try:
            df = fetch_trades(client, date)
        except Exception as e:
            print(f"  ERROR fetching data: {e}")
            continue

        if df.empty:
            print("  No data for this day.")
            continue

        day_dfs[date] = df
        alerts = simulate_and_evaluate(df, date, prior_levels=prior_levels)
        all_alerts.extend(alerts)

        # Compute this day's key levels to carry forward as prior_levels tomorrow.
        ib = df[(df.index.time >= MARKET_OPEN) & (df.index.time < IB_END)]
        if not ib.empty:
            prior_levels = [
                float(ib["price"].max()),  # prior IBH
                float(ib["price"].min()),  # prior IBL
                float(df["price"].iloc[-1]),  # prior session close
            ]

        c = sum(1 for a in alerts if a.outcome == "correct")
        i = sum(1 for a in alerts if a.outcome == "incorrect")
        n = sum(1 for a in alerts if a.outcome == "inconclusive")
        cum_correct += c
        cum_incorrect += i
        cum_inconc += n

        print(f"  Trades fetched    : {len(df):,}")
        print(
            f"  Alerts today      : {len(alerts)}  "
            f"({c} correct  |  {i} incorrect  |  {n} inconclusive)"
        )

        # Per-alert detail for correct and incorrect outcomes.
        decided = [a for a in alerts if a.outcome in ("correct", "incorrect")]
        if decided:

            def fmt(dt: datetime.datetime | None) -> str:
                if dt is None:
                    return "       —"
                return dt.astimezone(PT).strftime("%H:%M:%S")

            print(
                f"\n  {'Alert(PT)':>9}  {'Level':>5}  {'Line':>8}  {'Entry':>8}  "
                f"{'Dir':>6}  {'Hit(PT)':>8}  {'Done(PT)':>8}  Outcome"
            )
            print(
                f"  {'-'*9}  {'-'*5}  {'-'*8}  {'-'*8}  "
                f"{'-'*6}  {'-'*8}  {'-'*8}  {'-'*9}"
            )
            for a in sorted(decided, key=lambda x: x.alert_time):
                marker = "✓" if a.outcome == "correct" else "✗"
                print(
                    f"  {fmt(a.alert_time):>9}  "
                    f"{a.level:>5}  "
                    f"{a.line_price:>8.2f}  "
                    f"{a.entry_price:>8.2f}  "
                    f"{'↑ BUY' if a.direction == 'up' else '↓ SELL':>6}  "
                    f"{fmt(a.hit_time):>8}  "
                    f"{fmt(a.outcome_time):>8}  "
                    f"{marker} {a.outcome}"
                )
        print()

        print(
            f"  Cumulative total  : "
            f"{cum_correct} correct  |  {cum_incorrect} incorrect  |  {cum_inconc} inconclusive"
        )

    # Split alerts into original levels and fib levels for separate analysis.
    original_alerts = [a for a in all_alerts if not a.level.startswith("FIB_")]
    fib_alerts = [a for a in all_alerts if a.level.startswith("FIB_")]

    build_model(original_alerts)
    print_results(original_alerts, days)
    parameter_sweep(original_alerts, day_dfs)
    stop_loss_sweep(original_alerts, day_dfs)
    combinatorial_filter_sweep(original_alerts)
    level_specific_analysis(original_alerts)
    composite_scoring(original_alerts)
    ib_range_analysis(original_alerts)
    direction_level_interaction(original_alerts)
    day_of_week_scoring(original_alerts)
    enhanced_composite_sweep(original_alerts)
    fib_level_analysis(fib_alerts)
    combined_scoring_analysis(all_alerts, len(days), day_dfs)


# ── IB range width analysis ──────────────────────────────────────────────────


def ib_range_analysis(all_alerts: list[Alert]) -> None:
    """Analyze win rate by IB range width (IBH - IBL in points)."""
    decided = [
        a for a in all_alerts if a.outcome in ("correct", "incorrect") and a.features
    ]

    print(f"\n{'═' * 70}")
    print("  IB RANGE WIDTH ANALYSIS")
    print(f"  (Does a wider/narrower IB range predict better alert outcomes?)")
    print(f"{'═' * 70}")

    # Group alerts by their day, compute IB range for each day.
    from collections import defaultdict

    day_alerts: dict[datetime.date, list[Alert]] = defaultdict(list)
    for a in decided:
        day_alerts[a.date].append(a)

    # We need IBH/IBL per day — extract from alert data.
    day_ib: dict[datetime.date, float] = {}
    for date, alerts in day_alerts.items():
        ibh_alerts = [a for a in alerts if a.level == "IBH"]
        ibl_alerts = [a for a in alerts if a.level == "IBL"]
        if ibh_alerts and ibl_alerts:
            ibh_price = ibh_alerts[0].line_price
            ibl_price = ibl_alerts[0].line_price
            day_ib[date] = ibh_price - ibl_price

    if not day_ib:
        print("  No days with both IBH and IBL alerts — skipping.")
        return

    ranges = sorted(day_ib.values())
    median_range = ranges[len(ranges) // 2]

    # Tag each alert with its day's IB range.
    tagged = [(a, day_ib.get(a.date)) for a in decided]
    tagged = [(a, r) for a, r in tagged if r is not None]

    def win_rate_table(groups: list[tuple[str, list]]) -> None:
        print(f"  {'Group':<35}  {'W':>5}  {'L':>5}  {'Total':>7}  {'Win%':>6}")
        print(f"  {'-'*35}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*6}")
        for label, subset in groups:
            w = sum(1 for a in subset if a.outcome == "correct")
            l = sum(1 for a in subset if a.outcome == "incorrect")
            t = w + l
            wr = w / t if t > 0 else 0.0
            warn = "  ⚠ n<30" if 0 < t < 30 else ""
            print(f"  {label:<35}  {w:>5}  {l:>5}  {t:>7}  {wr:>5.1%}{warn}")

    q25 = ranges[len(ranges) // 4]
    q75 = ranges[3 * len(ranges) // 4]

    win_rate_table(
        [
            (f"Narrow IB (< {q25:.0f} pts)", [a for a, r in tagged if r < q25]),
            (
                f"Normal IB ({q25:.0f}–{q75:.0f} pts)",
                [a for a, r in tagged if q25 <= r <= q75],
            ),
            (f"Wide IB (> {q75:.0f} pts)", [a for a, r in tagged if r > q75]),
        ]
    )

    # Sweep IB range thresholds.
    print(
        f"\n  IB range filter sweep (only take alerts on days with IB range ≥ threshold):"
    )
    print(f"  {'Min IB range':>14}  {'W':>5}  {'L':>5}  {'Total':>7}  {'Win%':>6}")
    print(f"  {'-'*14}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*6}")
    for threshold in [0, 20, 30, 40, 50, 60, 80, 100]:
        subset = [a for a, r in tagged if r >= threshold]
        w = sum(1 for a in subset if a.outcome == "correct")
        l = sum(1 for a in subset if a.outcome == "incorrect")
        t = w + l
        if t < 10:
            continue
        wr = w / t if t > 0 else 0.0
        print(f"  {threshold:>14}  {w:>5}  {l:>5}  {t:>7}  {wr:>5.1%}")


# ── Direction × Level interaction ─────────────────────────────────────────────


def direction_level_interaction(all_alerts: list[Alert]) -> None:
    """Show win rates for every direction × level combination."""
    decided = [a for a in all_alerts if a.outcome in ("correct", "incorrect")]

    print(f"\n{'═' * 70}")
    print("  DIRECTION × LEVEL INTERACTION")
    print(f"  (Which direction works best at each level?)")
    print(f"{'═' * 70}")

    print(f"  {'Combo':<35}  {'W':>5}  {'L':>5}  {'Total':>7}  {'Win%':>6}")
    print(f"  {'-'*35}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*6}")

    for level in ["IBH", "IBL", "VWAP"]:
        for direction, dir_label in [("up", "BUY"), ("down", "SELL")]:
            subset = [
                a for a in decided if a.level == level and a.direction == direction
            ]
            w = sum(1 for a in subset if a.outcome == "correct")
            l = sum(1 for a in subset if a.outcome == "incorrect")
            t = w + l
            wr = w / t if t > 0 else 0.0
            warn = "  ⚠ n<30" if 0 < t < 30 else ""
            print(
                f"  {level} {dir_label:<30}  {w:>5}  {l:>5}  {t:>7}  {wr:>5.1%}{warn}"
            )


# ── Day of week scoring ───────────────────────────────────────────────────────


def day_of_week_scoring(all_alerts: list[Alert]) -> None:
    """Analyze whether day of week should be a score component."""
    decided = [
        a for a in all_alerts if a.outcome in ("correct", "incorrect") and a.features
    ]

    print(f"\n{'═' * 70}")
    print("  DAY OF WEEK AS POTENTIAL SCORE COMPONENT")
    print(f"  (Should we add day-of-week to composite score?)")
    print(f"{'═' * 70}")

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    print(
        f"  {'Day':<12}  {'W':>5}  {'L':>5}  {'Total':>7}  {'Win%':>6}  {'Suggested score adj':>20}"
    )
    print(f"  {'-'*12}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*20}")

    overall_w = sum(1 for a in decided if a.outcome == "correct")
    overall_t = len(decided)
    overall_wr = overall_w / overall_t if overall_t > 0 else 0.0

    for i, day_name in enumerate(days):
        subset = [a for a in decided if a.alert_time.weekday() == i]
        w = sum(1 for a in subset if a.outcome == "correct")
        l = sum(1 for a in subset if a.outcome == "incorrect")
        t = w + l
        wr = w / t if t > 0 else 0.0
        delta = wr - overall_wr
        if t < 30:
            suggestion = "n/a (too few)"
        elif delta > 0.05:
            suggestion = f"+1 ({delta:+.1%})"
        elif delta < -0.05:
            suggestion = f"-1 ({delta:+.1%})"
        else:
            suggestion = f" 0 ({delta:+.1%})"
        warn = "  ⚠ n<30" if 0 < t < 30 else ""
        print(
            f"  {day_name:<12}  {w:>5}  {l:>5}  {t:>7}  {wr:>5.1%}  {suggestion:>20}{warn}"
        )

    # Test what happens if we exclude worst day(s).
    print(f"\n  Impact of excluding worst day(s):")
    day_wrs = []
    for i in range(5):
        subset = [a for a in decided if a.alert_time.weekday() == i]
        w = sum(1 for a in subset if a.outcome == "correct")
        t = len(subset)
        day_wrs.append((i, days[i], w / t if t > 0 else 0.0, t))

    day_wrs.sort(key=lambda x: x[2])  # worst first

    remaining = list(decided)
    print(f"  {'Excluded':<20}  {'W':>5}  {'L':>5}  {'Total':>7}  {'Win%':>6}")
    print(f"  {'-'*20}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*6}")

    # Baseline
    w = sum(1 for a in remaining if a.outcome == "correct")
    l = sum(1 for a in remaining if a.outcome == "incorrect")
    t = w + l
    print(f"  {'(none)':<20}  {w:>5}  {l:>5}  {t:>7}  {w/t:>5.1%}")

    for day_idx, day_name, day_wr, day_count in day_wrs[:3]:
        remaining = [a for a in remaining if a.alert_time.weekday() != day_idx]
        w = sum(1 for a in remaining if a.outcome == "correct")
        l = sum(1 for a in remaining if a.outcome == "incorrect")
        t = w + l
        if t > 0:
            print(
                f"  {f'- {day_name} ({day_wr:.0%})':<20}  {w:>5}  {l:>5}  {t:>7}  {w/t:>5.1%}"
            )


# ── Enhanced composite sweep ─────────────────────────────────────────────────


def enhanced_composite_sweep(all_alerts: list[Alert]) -> None:
    """Test adding new components (day of week, direction×level, IB range) to composite score."""
    decided = [
        a for a in all_alerts if a.outcome in ("correct", "incorrect") and a.features
    ]

    print(f"\n{'═' * 70}")
    print("  ENHANCED COMPOSITE SCORING SWEEP")
    print(f"  (Testing new score components on top of existing 5-factor score)")
    print(f"{'═' * 70}")

    # Current 5-factor score (baseline).
    def base_score(a: Alert) -> int:
        s = 0
        mins = a.alert_time.hour * 60 + a.alert_time.minute
        if a.level == "IBL":
            s += 3
        elif a.level == "IBH":
            s -= 1
        if (13 * 60) <= mins < (15 * 60):
            s += 2
        elif (10 * 60 + 30) <= mins < (11 * 60 + 30):
            s -= 3
        elif (11 * 60 + 30) <= mins < (13 * 60):
            s -= 1
        else:
            s += 1
        tr = a.features.get("tick_rate", 0)
        if tr >= 2000:
            s += 2
        elif tr >= 1750:
            s += 1
        elif tr < 1000:
            s -= 2
        tc = a.level_test_count
        if tc == 1:
            s -= 4
        elif tc == 3:
            s += 2
        elif tc == 4:
            s += 1
        elif tc >= 5:
            s -= 1
        session_move = a.features.get("session_move_pts", 0)
        if -50 < session_move <= 0:
            s += 2
        elif session_move > 50:
            s -= 1
        return s

    # New component candidates.
    def dow_component(a: Alert) -> int:
        """Day of week adjustment."""
        dow = a.alert_time.weekday()
        # Will fill in based on actual data — for now test with Fri bonus.
        if dow == 4:
            return 1  # Friday
        if dow == 3:
            return -1  # Thursday (historically worst)
        return 0

    def direction_level_component(a: Alert) -> int:
        """Bonus for historically strong direction×level combos."""
        if a.level == "IBL" and a.direction == "up":
            return 1  # IBL BUY = support bounce
        if a.level == "IBH" and a.direction == "down":
            return -1  # IBH SELL = fade (historically weak)
        return 0

    def confluence_component(a: Alert) -> int:
        """Bonus for prior day level confluence."""
        return 1 if a.prior_confluence else 0

    # Test each new component individually, then together.
    component_combos = [
        ("Baseline (current 5-factor)", lambda a: 0),
        ("+ Day of week", dow_component),
        ("+ Direction×Level", direction_level_component),
        ("+ Prior confluence", confluence_component),
        (
            "+ All three new components",
            lambda a: dow_component(a)
            + direction_level_component(a)
            + confluence_component(a),
        ),
    ]

    for comp_label, comp_fn in component_combos:
        print(f"\n  {comp_label}:")
        scored = [(a, base_score(a) + comp_fn(a)) for a in decided]
        cutoffs = sorted(set(s for _, s in scored))

        print(
            f"  {'Cutoff':>7}  {'W':>5}  {'L':>5}  {'Total':>7}  {'Win%':>6}  {'EV/trade':>9}"
        )
        print(f"  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*9}")

        for cutoff in cutoffs:
            above = [a for a, s in scored if s >= cutoff]
            w = sum(1 for a in above if a.outcome == "correct")
            l = sum(1 for a in above if a.outcome == "incorrect")
            t = w + l
            if t < 30:
                continue
            wr = w / t
            ev = wr * TARGET_POINTS - (1 - wr) * STOP_POINTS
            print(f"  {cutoff:>7.0f}  {w:>5}  {l:>5}  {t:>7}  {wr:>5.1%}  {ev:>+8.1f}")


# ── Fibonacci level analysis ──────────────────────────────────────────────────


def fib_level_analysis(fib_alerts: list[Alert]) -> None:
    """Analyze win rates for each Fibonacci level derived from the IB range."""
    decided = [a for a in fib_alerts if a.outcome in ("correct", "incorrect")]

    print(f"\n{'═' * 90}")
    print("  FIBONACCI LEVEL ANALYSIS")
    print(
        f"  (Fib retracements & extensions based on IB range, {len(decided)} decided alerts)"
    )
    print(f"{'═' * 90}")

    if not decided:
        print("  No decided fib alerts — skipping.")
        return

    # Overall win rate by fib level.
    fib_names = sorted(set(a.level for a in decided))

    print(
        f"\n  {'Level':<25}  {'W':>5}  {'L':>5}  {'Inc':>5}  {'Total':>7}  {'Win%':>6}  {'EV/trade':>9}"
    )
    print(f"  {'-'*25}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*9}")

    inconc_alerts = [a for a in fib_alerts if a.outcome == "inconclusive"]
    for name in fib_names:
        w = sum(1 for a in decided if a.level == name and a.outcome == "correct")
        l = sum(1 for a in decided if a.level == name and a.outcome == "incorrect")
        inc = sum(1 for a in inconc_alerts if a.level == name)
        t = w + l
        wr = w / t if t > 0 else 0.0
        ev = wr * TARGET_POINTS - (1 - wr) * STOP_POINTS if t > 0 else 0.0
        warn = "  ⚠ n<30" if 0 < t < 30 else ""
        print(
            f"  {name:<25}  {w:>5}  {l:>5}  {inc:>5}  {t:>7}  {wr:>5.1%}  {ev:>+8.1f}{warn}"
        )

    # Compare fib vs original levels.
    print(f"\n  Summary:")
    total_w = sum(1 for a in decided if a.outcome == "correct")
    total_l = sum(1 for a in decided if a.outcome == "incorrect")
    total = total_w + total_l
    if total > 0:
        wr = total_w / total
        ev = wr * TARGET_POINTS - (1 - wr) * STOP_POINTS
        print(
            f"  All fib levels combined: {total_w}W / {total_l}L = {wr:.1%}  "
            f"(EV {ev:+.1f} pts/trade, {total} trades)"
        )

    # Win rate by direction for each fib level.
    print(
        f"\n  {'Level × Direction':<35}  {'W':>5}  {'L':>5}  {'Total':>7}  {'Win%':>6}"
    )
    print(f"  {'-'*35}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*6}")
    for name in fib_names:
        for direction, dir_label in [("up", "BUY"), ("down", "SELL")]:
            subset = [
                a for a in decided if a.level == name and a.direction == direction
            ]
            w = sum(1 for a in subset if a.outcome == "correct")
            l = sum(1 for a in subset if a.outcome == "incorrect")
            t = w + l
            if t == 0:
                continue
            wr = w / t
            warn = "  ⚠ n<30" if t < 30 else ""
            print(
                f"  {name + ' ' + dir_label:<35}  {w:>5}  {l:>5}  {t:>7}  {wr:>5.1%}{warn}"
            )

    # Win rate by time of day for fib levels that have enough data.
    print(f"\n  {'Level × Time':<40}  {'W':>5}  {'L':>5}  {'Total':>7}  {'Win%':>6}")
    print(f"  {'-'*40}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*6}")

    def time_bucket(a: Alert) -> str:
        mins = a.alert_time.hour * 60 + a.alert_time.minute
        if mins < 11 * 60 + 30:
            return "first hour"
        elif mins < 13 * 60:
            return "lunch"
        elif mins < 15 * 60:
            return "afternoon"
        else:
            return "power hour"

    for name in fib_names:
        level_decided = [a for a in decided if a.level == name]
        if len(level_decided) < 30:
            continue
        for tb in ["first hour", "lunch", "afternoon", "power hour"]:
            subset = [a for a in level_decided if time_bucket(a) == tb]
            w = sum(1 for a in subset if a.outcome == "correct")
            l = sum(1 for a in subset if a.outcome == "incorrect")
            t = w + l
            if t == 0:
                continue
            wr = w / t
            warn = "  ⚠ n<30" if t < 30 else ""
            print(
                f"  {name + ' / ' + tb:<40}  {w:>5}  {l:>5}  {t:>7}  {wr:>5.1%}{warn}"
            )

    # Win rate by test count for fib levels.
    print(
        f"\n  {'Level × Test Count':<35}  {'W':>5}  {'L':>5}  {'Total':>7}  {'Win%':>6}"
    )
    print(f"  {'-'*35}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*6}")
    for name in fib_names:
        level_decided = [a for a in decided if a.level == name]
        if len(level_decided) < 30:
            continue
        for tc_label, tc_fn in [
            ("#1", lambda a: a.level_test_count == 1),
            ("#2", lambda a: a.level_test_count == 2),
            ("#3+", lambda a: a.level_test_count >= 3),
        ]:
            subset = [a for a in level_decided if tc_fn(a)]
            w = sum(1 for a in subset if a.outcome == "correct")
            l = sum(1 for a in subset if a.outcome == "incorrect")
            t = w + l
            if t == 0:
                continue
            wr = w / t
            warn = "  ⚠ n<30" if t < 30 else ""
            print(
                f"  {name + ' / test ' + tc_label:<35}  {w:>5}  {l:>5}  {t:>7}  {wr:>5.1%}{warn}"
            )


def combined_scoring_analysis(
    all_alerts: list[Alert], n_days: int, day_dfs: dict | None = None
) -> None:
    """Score ALL alerts (original + fib) with composite scoring and show combined stats."""
    decided = [
        a for a in all_alerts if a.outcome in ("correct", "incorrect") and a.features
    ]

    print(f"\n{'═' * 90}")
    print("  COMBINED SCORING ANALYSIS (ALL LEVELS: IBH/IBL/VWAP + Fib)")
    print(
        f"  (Composite score applied to all {len(decided)} decided alerts across {n_days} days)"
    )
    print(f"{'═' * 90}")

    def score_alert(a: Alert) -> int:
        """Must match _composite_score() in alert_manager.py exactly."""
        s = 0
        mins = a.alert_time.hour * 60 + a.alert_time.minute

        # Level quality
        if a.level == "IBL":
            s += 3
        elif a.level == "IBH":
            s -= 1
        elif a.level == "FIB_EXT_LO_1.272":
            s += 2
        elif a.level == "FIB_EXT_HI_1.272":
            s += 1

        # Direction × Level interaction
        combo = (a.level, a.direction)
        if combo in (
            ("FIB_EXT_HI_1.272", "up"),
            ("FIB_EXT_LO_1.272", "down"),
            ("IBL", "down"),
        ):
            s += 1
        elif combo in (("IBH", "up"),):
            s -= 1

        # Time of day
        if (13 * 60) <= mins < (15 * 60):
            s += 2
        elif (10 * 60 + 30) <= mins < (11 * 60 + 30):
            s -= 3
        elif (11 * 60 + 30) <= mins < (13 * 60):
            s -= 1
        else:
            s += 1

        # Tick rate
        tr = a.features.get("tick_rate", 0)
        if tr >= 2000:
            s += 2
        elif tr >= 1750:
            s += 1
        elif tr < 1000:
            s -= 2

        # Test count
        tc = a.level_test_count
        if tc == 1:
            s -= 4
        elif tc == 3:
            s += 2
        elif tc == 4:
            s += 1
        elif tc >= 5:
            s -= 1

        # Session context
        session_move = a.features.get("session_move_pts", 0)
        if -50 < session_move <= 0:
            s += 2
        elif session_move > 50:
            s -= 1

        # Outcome streak
        cw = a.features.get("consecutive_wins", 0)
        cl = a.features.get("consecutive_losses", 0)
        if cw >= 2:
            s += 2
        elif cl >= 2:
            s -= 3

        return s

    # Score alerts chronologically so streak tracking works correctly.
    # Alerts must be sorted by time so that consecutive_wins/losses reflect
    # the outcomes of earlier alerts when scoring later ones.
    decided_sorted = sorted(decided, key=lambda a: a.alert_time)
    scored = []
    recent_outcomes: list[str] = []
    for a in decided_sorted:
        # Compute streak from prior resolved alerts
        cons_wins = 0
        for o in reversed(recent_outcomes):
            if o == "correct":
                cons_wins += 1
            else:
                break
        cons_losses = 0
        for o in reversed(recent_outcomes):
            if o == "incorrect":
                cons_losses += 1
            else:
                break
        a.features["consecutive_wins"] = cons_wins
        a.features["consecutive_losses"] = cons_losses
        s = score_alert(a)
        scored.append((a, s))
        if a.outcome in ("correct", "incorrect"):
            recent_outcomes.append(a.outcome)

    # Live system levels only (no IBH, no FIB_RET_0.236).
    _LIVE_LEVELS = {"IBL", "VWAP", "FIB_EXT_LO_1.272", "FIB_EXT_HI_1.272"}

    live_filtered = [(a, s) for a, s in scored if a.level in _LIVE_LEVELS and s >= 3]

    # --- All-levels cutoff sweep (for reference) ---
    all_scores = sorted(set(s for _, s in scored))
    print(f"\n  Cutoff sweep (all levels, uniform cutoff):")
    print(
        f"  {'Cutoff':>7}  {'W':>5}  {'L':>5}  {'Total':>7}  {'Win%':>6}  "
        f"{'EV/trade':>9}  {'Alerts/day':>10}  {'$/day @20c':>11}"
    )
    print(
        f"  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*6}  "
        f"{'-'*9}  {'-'*10}  {'-'*11}"
    )

    for cutoff in sorted(all_scores):
        above = [a for a, s in scored if s >= cutoff]
        w = sum(1 for a in above if a.outcome == "correct")
        l = sum(1 for a in above if a.outcome == "incorrect")
        t = w + l
        if t < 10:
            continue
        wr = w / t
        ev_pts = wr * TARGET_POINTS - (1 - wr) * STOP_POINTS
        alerts_per_day = t / n_days
        daily_dollar = ev_pts * 40 * alerts_per_day
        marker = " ← current" if cutoff == 3 else ""
        print(
            f"  {cutoff:>7.0f}  {w:>5}  {l:>5}  {t:>7}  {wr:>5.1%}  "
            f"{ev_pts:>+8.1f}  {alerts_per_day:>10.1f}  {daily_dollar:>+10.0f}{marker}"
        )

    # --- Live system: per-level breakdown ---
    print(
        f"\n  LIVE SYSTEM (4 levels: IBL, VWAP, FIB_EXT_LO_1.272, FIB_EXT_HI_1.272, score ≥ 3):"
    )
    print(
        f"  {'Level':<25}  {'W':>5}  {'L':>5}  {'Total':>7}  {'Win%':>6}  "
        f"{'EV/trade':>9}  {'Alerts/day':>10}"
    )
    print(f"  {'-'*25}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*9}  {'-'*10}")
    level_names = sorted(set(a.level for a, _ in live_filtered))
    for name in level_names:
        level_alerts = [a for a, s in live_filtered if a.level == name]
        w = sum(1 for a in level_alerts if a.outcome == "correct")
        l = sum(1 for a in level_alerts if a.outcome == "incorrect")
        t = w + l
        if t == 0:
            continue
        wr = w / t
        ev_pts = wr * TARGET_POINTS - (1 - wr) * STOP_POINTS
        apd = t / n_days
        print(
            f"  {name:<25}  {w:>5}  {l:>5}  {t:>7}  {wr:>5.1%}  "
            f"{ev_pts:>+8.1f}  {apd:>10.1f}"
        )

    # Grand total for live system
    total_w = sum(1 for a, _ in live_filtered if a.outcome == "correct")
    total_l = sum(1 for a, _ in live_filtered if a.outcome == "incorrect")
    total_t = total_w + total_l
    if total_t > 0:
        total_wr = total_w / total_t
        total_ev = total_wr * TARGET_POINTS - (1 - total_wr) * STOP_POINTS
        total_apd = total_t / n_days
        daily_ev = total_ev * 40 * total_apd
        print(
            f"\n  {'TOTAL':<25}  {total_w:>5}  {total_l:>5}  {total_t:>7}  {total_wr:>5.1%}  "
            f"{total_ev:>+8.1f}  {total_apd:>10.1f}"
        )
        print(f"\n  Expected value at 20 contracts ($2/pt):")
        print(f"    Per trade : {total_ev * 40:>+.2f}")
        print(f"    Per day   : {daily_ev:>+.0f}  ({total_apd:.1f} alerts/day)")

    # Parameter sweep on live-system alerts only
    if day_dfs:
        live_alerts = [a for a, s in live_filtered]
        _live_target_sweep(live_alerts, day_dfs, n_days)
        _experimental_factors(live_alerts, day_dfs, n_days)


def _live_target_sweep(
    live_alerts: list[Alert],
    day_dfs: dict,
    n_days: int,
) -> None:
    """Sweep target sizes on live-system-filtered alerts only."""
    combos = [
        (8, 15 * 60, "+8 pts / 15 min (current)"),
        (10, 15 * 60, "+10 pts / 15 min"),
        (10, 20 * 60, "+10 pts / 20 min"),
        (12, 15 * 60, "+12 pts / 15 min"),
        (12, 20 * 60, "+12 pts / 20 min"),
        (15, 15 * 60, "+15 pts / 15 min"),
        (15, 20 * 60, "+15 pts / 20 min"),
        (20, 15 * 60, "+20 pts / 15 min"),
        (20, 20 * 60, "+20 pts / 20 min"),
        (20, 30 * 60, "+20 pts / 30 min"),
        (25, 20 * 60, "+25 pts / 20 min"),
        (25, 30 * 60, "+25 pts / 30 min"),
    ]

    print(f"\n{'═' * 90}")
    print("  LIVE SYSTEM TARGET SWEEP")
    print(
        f"  (4 levels, score ≥ 3, direction-aware — re-evaluating {len(live_alerts)} alerts)"
    )
    print(f"{'═' * 90}")
    print(
        f"  {'Params':<30}  {'W':>5}  {'L':>5}  {'Total':>7}  {'Win%':>6}  "
        f"{'EV/trade':>9}  {'$/trade @20c':>12}  {'$/day':>8}"
    )
    print(
        f"  {'-'*30}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*6}  "
        f"{'-'*9}  {'-'*12}  {'-'*8}"
    )

    for target, window, label in combos:
        correct = incorrect = 0

        for alert in live_alerts:
            df = day_dfs.get(alert.date)
            if df is None:
                continue

            prices = df["price"]
            alert_ts = pd.Timestamp(alert.alert_time)
            window_end = alert_ts + pd.Timedelta(seconds=WINDOW_SECS)

            hit_seg = prices[(prices.index > alert_ts) & (prices.index <= window_end)]
            hit_mask = abs(hit_seg - alert.line_price) <= HIT_THRESHOLD
            if not hit_mask.any():
                continue  # inconclusive — skip

            hit_ts = hit_mask.idxmax()
            eval_end = hit_ts + pd.Timedelta(seconds=window)
            eval_seg = prices[(prices.index > hit_ts) & (prices.index <= eval_end)]

            if alert.direction == "up":
                target_mask = eval_seg >= alert.line_price + target
                stop_mask = eval_seg <= alert.line_price - STOP_POINTS
            else:
                target_mask = eval_seg <= alert.line_price - target
                stop_mask = eval_seg >= alert.line_price + STOP_POINTS

            target_hit = target_mask.any()
            stop_hit = stop_mask.any()

            if target_hit and stop_hit:
                if eval_seg.index[target_mask][0] <= eval_seg.index[stop_mask][0]:
                    correct += 1
                else:
                    incorrect += 1
            elif target_hit:
                correct += 1
            else:
                incorrect += 1

        decided = correct + incorrect
        if decided == 0:
            continue
        wr = correct / decided
        ev = wr * target - (1 - wr) * STOP_POINTS
        dollar_per_trade = ev * 40  # $2/pt × 20 contracts
        dollar_per_day = dollar_per_trade * (decided / n_days)
        marker = " ← current" if "(current)" in label else ""
        print(
            f"  {label:<30}  {correct:>5}  {incorrect:>5}  {decided:>7}  {wr:>5.1%}  "
            f"{ev:>+8.1f}  {dollar_per_trade:>+11.0f}  {dollar_per_day:>+7.0f}{marker}"
        )


def _experimental_factors(
    live_alerts: list[Alert],
    day_dfs: dict,
    n_days: int,
) -> None:
    """Test 6 experimental factors on live-system-filtered alerts."""
    decided = [
        a for a in live_alerts if a.outcome in ("correct", "incorrect") and a.features
    ]
    if not decided:
        return

    def _print_groups(title: str, groups: list[tuple[str, list[Alert]]]) -> None:
        print(f"\n  {title}")
        print(
            f"  {'Group':<40}  {'W':>5}  {'L':>5}  {'Total':>7}  {'Win%':>6}  {'EV':>6}"
        )
        print(f"  {'-'*40}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*6}")
        for label, subset in groups:
            w = sum(1 for a in subset if a.outcome == "correct")
            l = sum(1 for a in subset if a.outcome == "incorrect")
            t = w + l
            if t == 0:
                continue
            wr = w / t
            ev = wr * TARGET_POINTS - (1 - wr) * STOP_POINTS
            warn = "  ⚠ n<30" if t < 30 else ""
            print(
                f"  {label:<40}  {w:>5}  {l:>5}  {t:>7}  {wr:>5.1%}  {ev:>+5.1f}{warn}"
            )

    print(f"\n{'═' * 90}")
    print("  EXPERIMENTAL FACTOR ANALYSIS (live system alerts only)")
    print(f"  ({len(decided)} decided alerts across {n_days} days)")
    print(f"{'═' * 90}")

    # ── 1. IB Range Width ─────────────────────────────────────────────────────
    # Compute IB range for each day from day_dfs.
    day_ib_range: dict[datetime.date, float] = {}
    for date, df in day_dfs.items():
        ib = df[(df.index.time >= MARKET_OPEN) & (df.index.time < IB_END)]
        if not ib.empty:
            day_ib_range[date] = float(ib["price"].max()) - float(ib["price"].min())

    alerts_with_ib = [
        (a, day_ib_range.get(a.date, 0)) for a in decided if a.date in day_ib_range
    ]
    ib_ranges = sorted(r for _, r in alerts_with_ib)
    if ib_ranges:
        q1 = ib_ranges[len(ib_ranges) // 4]
        q2 = ib_ranges[len(ib_ranges) // 2]
        q3 = ib_ranges[3 * len(ib_ranges) // 4]
        _print_groups(
            "1. IB RANGE WIDTH",
            [
                (f"Narrow (< {q1:.0f} pts)", [a for a, r in alerts_with_ib if r < q1]),
                (
                    f"Normal ({q1:.0f}–{q2:.0f} pts)",
                    [a for a, r in alerts_with_ib if q1 <= r < q2],
                ),
                (
                    f"Wide ({q2:.0f}–{q3:.0f} pts)",
                    [a for a, r in alerts_with_ib if q2 <= r < q3],
                ),
                (
                    f"Very wide (≥ {q3:.0f} pts)",
                    [a for a, r in alerts_with_ib if r >= q3],
                ),
            ],
        )

    # ── 2. Prior Day Level Confluence ─────────────────────────────────────────
    _print_groups(
        "2. PRIOR DAY LEVEL CONFLUENCE",
        [
            ("With confluence", [a for a in decided if a.prior_confluence]),
            ("Without confluence", [a for a in decided if not a.prior_confluence]),
        ],
    )

    # ── 3. Approach Momentum ─────────────────────────────────────────────────
    mom_alerts = [
        (a, a.features.get("approach_momentum", 0))
        for a in decided
        if "approach_momentum" in a.features
    ]
    if mom_alerts:
        moms = sorted(m for _, m in mom_alerts)
        mq1 = moms[len(moms) // 4]
        mq2 = moms[len(moms) // 2]
        mq3 = moms[3 * len(moms) // 4]
        _print_groups(
            "3. APPROACH MOMENTUM (speed toward level)",
            [
                (f"Q1: slow (< {mq1:.1f})", [a for a, m in mom_alerts if m < mq1]),
                (
                    f"Q2: moderate ({mq1:.1f}–{mq2:.1f})",
                    [a for a, m in mom_alerts if mq1 <= m < mq2],
                ),
                (
                    f"Q3: fast ({mq2:.1f}–{mq3:.1f})",
                    [a for a, m in mom_alerts if mq2 <= m < mq3],
                ),
                (
                    f"Q4: very fast (≥ {mq3:.1f})",
                    [a for a, m in mom_alerts if m >= mq3],
                ),
            ],
        )

        # Also test approach volatility
        vol_alerts = [
            (a, a.features.get("volatility", 0))
            for a in decided
            if "volatility" in a.features
        ]
        if vol_alerts:
            vols = sorted(v for _, v in vol_alerts)
            vq2 = vols[len(vols) // 2]
            _print_groups(
                "   APPROACH VOLATILITY (choppiness on approach)",
                [
                    (
                        f"Low volatility (< {vq2:.1f})",
                        [a for a, v in vol_alerts if v < vq2],
                    ),
                    (
                        f"High volatility (≥ {vq2:.1f})",
                        [a for a, v in vol_alerts if v >= vq2],
                    ),
                ],
            )

    # ── 4. Multi-Level Proximity ──────────────────────────────────────────────
    # For each alert, check if another live level was within 20 pts at alert time.
    # We need to reconstruct level prices per day.
    PROXIMITY_PTS = 20
    proximity_results = []
    for a in decided:
        df = day_dfs.get(a.date)
        if df is None:
            continue
        # Reconstruct levels for this day
        ib = df[(df.index.time >= MARKET_OPEN) & (df.index.time < IB_END)]
        if ib.empty:
            continue
        ibh = float(ib["price"].max())
        ibl = float(ib["price"].min())
        ib_range = ibh - ibl

        # VWAP at alert time
        session = df[
            (df.index.time >= MARKET_OPEN) & (df.index <= pd.Timestamp(a.alert_time))
        ]
        if session.empty or session["size"].sum() == 0:
            continue
        vwap = float((session["price"] * session["size"]).sum() / session["size"].sum())

        all_levels = {
            "IBL": ibl,
            "VWAP": vwap,
            "FIB_EXT_LO_1.272": ibl - 0.272 * ib_range,
            "FIB_EXT_HI_1.272": ibh + 0.272 * ib_range,
        }
        # Count how many OTHER levels are within PROXIMITY_PTS of this alert's level
        nearby = sum(
            1
            for name, price in all_levels.items()
            if name != a.level and abs(price - a.line_price) <= PROXIMITY_PTS
        )
        proximity_results.append((a, nearby))

    if proximity_results:
        _print_groups(
            "4. MULTI-LEVEL PROXIMITY (other levels within 20 pts)",
            [
                ("No nearby levels", [a for a, n in proximity_results if n == 0]),
                ("1 nearby level", [a for a, n in proximity_results if n == 1]),
                ("2+ nearby levels", [a for a, n in proximity_results if n >= 2]),
            ],
        )

    # ── 5. Partial Profit Taking ──────────────────────────────────────────────
    # Simulate: half at +4, half rides to +12 (or stopped at -20).
    # Compare average P/L vs current all-in +8.
    print(f"\n  5. PARTIAL PROFIT SIMULATION")
    print(f"  (Compare all-in +8 vs half@+4 + half@+12, same -20 stop)")
    print(
        f"  {'Strategy':<40}  {'Avg P/L':>8}  {'Win%':>6}  {'$/trade @20c':>12}  {'$/day':>8}"
    )
    print(f"  {'-'*40}  {'-'*8}  {'-'*6}  {'-'*12}  {'-'*8}")

    # Current: all-in +8
    current_w = sum(1 for a in decided if a.outcome == "correct")
    current_l = len(decided) - current_w
    current_wr = current_w / len(decided)
    current_ev = current_wr * TARGET_POINTS - (1 - current_wr) * STOP_POINTS
    current_dollar = current_ev * 40
    current_daily = current_dollar * (len(decided) / n_days)
    print(
        f"  {'All-in +8 pts (current)':<40}  {current_ev:>+7.1f}  {current_wr:>5.1%}  {current_dollar:>+11.0f}  {current_daily:>+7.0f}"
    )

    # Partial: half at +4, half at +12
    partial_pnl = []
    for alert in decided:
        df = day_dfs.get(alert.date)
        if df is None:
            continue
        prices = df["price"]
        alert_ts = pd.Timestamp(alert.alert_time)
        window_end = alert_ts + pd.Timedelta(seconds=WINDOW_SECS)

        hit_seg = prices[(prices.index > alert_ts) & (prices.index <= window_end)]
        hit_mask = abs(hit_seg - alert.line_price) <= HIT_THRESHOLD
        if not hit_mask.any():
            continue

        hit_ts = hit_mask.idxmax()
        eval_end = hit_ts + pd.Timedelta(seconds=15 * 60)
        eval_seg = prices[(prices.index > hit_ts) & (prices.index <= eval_end)]
        if eval_seg.empty:
            continue

        if alert.direction == "up":
            t1_mask = eval_seg >= alert.line_price + 4
            t2_mask = eval_seg >= alert.line_price + 12
            stop_mask = eval_seg <= alert.line_price - STOP_POINTS
        else:
            t1_mask = eval_seg <= alert.line_price - 4
            t2_mask = eval_seg <= alert.line_price - 12
            stop_mask = eval_seg >= alert.line_price + STOP_POINTS

        t1_hit = t1_mask.any()
        t2_hit = t2_mask.any()
        stop_hit = stop_mask.any()

        # Determine order of events
        t1_ts = eval_seg.index[t1_mask][0] if t1_hit else None
        t2_ts = eval_seg.index[t2_mask][0] if t2_hit else None
        stop_ts = eval_seg.index[stop_mask][0] if stop_hit else None

        # Half 1: +4 target
        half1 = 0.0
        if t1_hit and (stop_ts is None or t1_ts <= stop_ts):
            half1 = 4.0
        elif stop_hit:
            half1 = -20.0
        else:
            half1 = -20.0  # time expired = loss

        # Half 2: +12 target (only if not stopped before +4)
        half2 = 0.0
        if t2_hit and (stop_ts is None or t2_ts <= stop_ts):
            half2 = 12.0
        elif stop_hit:
            half2 = -20.0
        else:
            half2 = -20.0

        partial_pnl.append((half1 + half2) / 2)  # average of two halves

    if partial_pnl:
        avg_partial = sum(partial_pnl) / len(partial_pnl)
        partial_wr = sum(1 for p in partial_pnl if p > 0) / len(partial_pnl)
        partial_dollar = avg_partial * 40
        partial_daily = partial_dollar * (len(partial_pnl) / n_days)
        print(
            f"  {'Half@+4, half@+12':<40}  {avg_partial:>+7.1f}  {partial_wr:>5.1%}  {partial_dollar:>+11.0f}  {partial_daily:>+7.0f}"
        )

    # Also test half@+4, half@+16
    partial_pnl2 = []
    for alert in decided:
        df = day_dfs.get(alert.date)
        if df is None:
            continue
        prices = df["price"]
        alert_ts = pd.Timestamp(alert.alert_time)
        window_end = alert_ts + pd.Timedelta(seconds=WINDOW_SECS)

        hit_seg = prices[(prices.index > alert_ts) & (prices.index <= window_end)]
        hit_mask = abs(hit_seg - alert.line_price) <= HIT_THRESHOLD
        if not hit_mask.any():
            continue

        hit_ts = hit_mask.idxmax()
        eval_end = hit_ts + pd.Timedelta(seconds=20 * 60)
        eval_seg = prices[(prices.index > hit_ts) & (prices.index <= eval_end)]
        if eval_seg.empty:
            continue

        if alert.direction == "up":
            t1_mask = eval_seg >= alert.line_price + 4
            t2_mask = eval_seg >= alert.line_price + 16
            stop_mask = eval_seg <= alert.line_price - STOP_POINTS
        else:
            t1_mask = eval_seg <= alert.line_price - 4
            t2_mask = eval_seg <= alert.line_price - 16
            stop_mask = eval_seg >= alert.line_price + STOP_POINTS

        t1_hit = t1_mask.any()
        t2_hit = t2_mask.any()
        stop_hit = stop_mask.any()

        t1_ts = eval_seg.index[t1_mask][0] if t1_hit else None
        t2_ts = eval_seg.index[t2_mask][0] if t2_hit else None
        stop_ts = eval_seg.index[stop_mask][0] if stop_hit else None

        half1 = 4.0 if (t1_hit and (stop_ts is None or t1_ts <= stop_ts)) else -20.0
        half2 = 16.0 if (t2_hit and (stop_ts is None or t2_ts <= stop_ts)) else -20.0
        partial_pnl2.append((half1 + half2) / 2)

    if partial_pnl2:
        avg_p2 = sum(partial_pnl2) / len(partial_pnl2)
        wr_p2 = sum(1 for p in partial_pnl2 if p > 0) / len(partial_pnl2)
        dollar_p2 = avg_p2 * 40
        daily_p2 = dollar_p2 * (len(partial_pnl2) / n_days)
        print(
            f"  {'Half@+4, half@+16 (20min)':<40}  {avg_p2:>+7.1f}  {wr_p2:>5.1%}  {dollar_p2:>+11.0f}  {daily_p2:>+7.0f}"
        )

    # ── 6. Consecutive Outcome Streaks ────────────────────────────────────────
    # Sort alerts chronologically and check if outcomes are auto-correlated.
    sorted_alerts = sorted(decided, key=lambda a: a.alert_time)
    streak_data = []  # (previous N outcomes, this outcome)
    for i, a in enumerate(sorted_alerts):
        if i < 3:
            continue
        prev3 = [sorted_alerts[j].outcome for j in range(i - 3, i)]
        streak_data.append((prev3, a))

    if streak_data:
        # After 3 consecutive correct
        after_3w = [a for prev, a in streak_data if all(p == "correct" for p in prev)]
        after_3l = [a for prev, a in streak_data if all(p == "incorrect" for p in prev)]
        after_2w = [a for prev, a in streak_data if prev[-2:] == ["correct", "correct"]]
        after_2l = [
            a for prev, a in streak_data if prev[-2:] == ["incorrect", "incorrect"]
        ]
        after_mixed = [
            a
            for prev, a in streak_data
            if not (
                all(p == "correct" for p in prev) or all(p == "incorrect" for p in prev)
            )
        ]

        _print_groups(
            "6. CONSECUTIVE OUTCOME ANALYSIS (does streaking matter?)",
            [
                ("After 3 consecutive wins", after_3w),
                ("After 2 consecutive wins", after_2w),
                ("After mixed results", after_mixed),
                ("After 2 consecutive losses", after_2l),
                ("After 3 consecutive losses", after_3l),
            ],
        )


if __name__ == "__main__":
    main()
