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

DATASET       = "GLBX.MDP3"
SYMBOL        = "MNQ.c.0"
MARKET_OPEN   = datetime.time(9,  30)
IB_END        = datetime.time(10, 30)
MARKET_CLOSE  = datetime.time(16,  0)

ALERT_THRESHOLD = 7.0    # points — zone entry (must match live config)
EXIT_THRESHOLD  = 20.0   # points — zone exit (must match live config)
HIT_THRESHOLD   = 1.0    # points — price within this = "touched the line"
TARGET_POINTS   = 10.0   # points in recommended direction = correct
WINDOW_SECS     = 15 * 60  # 15-minute evaluation window
FEATURE_SECS    = 3  * 60  # 3-minute approach window BEFORE alert fires


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Alert:
    date:             datetime.date
    alert_time:       datetime.datetime
    level:            str           # VWAP, IBH, or IBL
    line_price:       float         # level price locked at zone entry
    entry_price:      float         # MNQ price when alert fired
    direction:        str           # 'up' = BUY rec, 'down' = SELL rec
    level_test_count: int  = 1      # which test of this level today (1 = first touch)
    hit_time:         datetime.datetime | None = None
    outcome_time:     datetime.datetime | None = None
    outcome:          str = "inconclusive"
    features:         dict = field(default_factory=dict)
    cv_pred:          int | None = None   # 0=predicted incorrect (avoid), 1=predicted correct


class ZoneState:
    """Alert zone state — mirrors the live LevelState logic exactly."""

    def __init__(self, name: str, price: float) -> None:
        self.name        = name
        self.price       = price
        self.in_zone     = False
        self.ref: float | None = None
        self.entry_count = 0   # cumulative zone entries today for this level

    def update(self, current_price: float, new_level_price: float | None = None) -> bool:
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
            self.in_zone     = True
            self.ref         = self.price
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
    end   = ET.localize(datetime.datetime.combine(date, MARKET_CLOSE)).isoformat()

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
        ts    = pd.Timestamp(rec.ts_event, unit="ns", tz="UTC").tz_convert(ET)
        price = rec.price / 1_000_000_000
        size  = int(rec.size)
        rows.append((ts, price, size))
        if len(rows) % 10_000 == 0:
            print(f"    ... {len(rows):,} trades downloaded", flush=True)

    if not rows:
        return pd.DataFrame(columns=["price", "size"])

    df = pd.DataFrame(rows, columns=["ts", "price", "size"]).set_index("ts").sort_index()

    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_parquet(path)
    print(f"    [cache] saved {date} ({len(df):,} trades)", flush=True)
    return df


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_and_evaluate(df: pd.DataFrame, date: datetime.date) -> list[Alert]:
    """
    Simulate alert system + evaluate outcomes for one trading day.
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
    df["vwap"]         = (df["price"] * df["size"]).cumsum() / df["size"].cumsum()
    df["session_high"] = df["price"].cummax()
    df["session_low"]  = df["price"].cummin()
    day_open = float(df["price"].iloc[0])

    # ── Alert zone simulation (post-IB only) ─────────────────────────────────
    zones = {
        "IBH":  ZoneState("IBH",  ibh),
        "IBL":  ZoneState("IBL",  ibl),
        "VWAP": ZoneState("VWAP", float(df["vwap"].iloc[0])),
    }

    alerts: list[Alert] = []
    post_ib = df[df.index.time >= IB_END]

    # itertuples is ~8x faster than iterrows for large DataFrames.
    for tick_num, row in enumerate(post_ib.itertuples()):
        if tick_num % 10_000 == 0:
            print(f"    [sim] {row.Index.strftime('%Y-%m-%d %H:%M:%S')} ET  "
                  f"({tick_num:,} ticks)", flush=True)

        ts    = row.Index
        price = row.price
        vwap  = row.vwap
        level_prices = {"IBH": ibh, "IBL": ibl, "VWAP": vwap}

        for name, zone in zones.items():
            if zone.update(price, new_level_price=level_prices[name]):
                alert_time_mins = ts.hour * 60 + ts.minute

                # Hard filter 1: skip alerts in the first post-IB hour.
                # Backtest shows 62.1% win rate (10:30–11:30 ET) — below break-even.
                if (10 * 60 + 30) <= alert_time_mins < (11 * 60 + 30):
                    continue

                # Hard filter 2: skip the first test of any level.
                # Backtest shows 52.8% win rate on test #1 — well below break-even.
                if zone.entry_count == 1:
                    continue

                direction = "up" if price > zone.ref else "down"
                alerts.append(Alert(
                    date=date,
                    alert_time=ts.to_pydatetime(warn=False),
                    level=name,
                    line_price=zone.ref,
                    entry_price=price,
                    direction=direction,
                    level_test_count=zone.entry_count,
                ))

    # ── Outcome evaluation (vectorized — no iterrows) ─────────────────────────
    prices = df["price"]

    for alert in alerts:
        alert_ts   = pd.Timestamp(alert.alert_time)
        window_end = alert_ts + pd.Timedelta(seconds=WINDOW_SECS)

        # Phase 1: first tick where price touches the line within 15 min.
        hit_seg  = prices[(prices.index > alert_ts) & (prices.index <= window_end)]
        hit_mask = abs(hit_seg - alert.line_price) <= HIT_THRESHOLD
        if not hit_mask.any():
            alert.outcome = "inconclusive"
            continue

        hit_ts = hit_mask.idxmax()
        alert.hit_time = hit_ts.to_pydatetime(warn=False)

        # Phase 2: did price hit +TARGET_POINTS within 15 min of touching the line?
        # Correct   = target reached within the window.
        # Incorrect = line was touched but target not reached — regardless of whether
        #             a stop was hit or time simply expired.
        eval_end = hit_ts + pd.Timedelta(seconds=WINDOW_SECS)
        eval_seg = prices[(prices.index > hit_ts) & (prices.index <= eval_end)]

        if alert.direction == "up":
            target_mask = eval_seg >= alert.line_price + TARGET_POINTS
        else:
            target_mask = eval_seg <= alert.line_price - TARGET_POINTS

        if target_mask.any():
            alert.outcome      = "correct"
            alert.outcome_time = eval_seg.index[target_mask][0].to_pydatetime(warn=False)
        else:
            alert.outcome = "incorrect"

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

        alert_ts     = pd.Timestamp(alert.alert_time)
        window_start = alert_ts - pd.Timedelta(seconds=FEATURE_SECS)
        window       = df[(df.index >= window_start) & (df.index <= alert_ts)]

        if len(window) < 3:
            # Too few ticks in approach window (e.g. alert fired very close
            # to IB end) — skip rather than compute degenerate features.
            continue

        prices_arr = window["price"].values
        sizes_arr  = window["size"].values
        n          = len(prices_arr)

        # Split window into two equal halves.
        mid = n // 2
        first_prices  = prices_arr[:mid]  if mid >= 2 else prices_arr[:1]
        second_prices = prices_arr[mid:]  if n - mid >= 2 else prices_arr[-1:]
        first_sizes   = sizes_arr[:mid]
        second_sizes  = sizes_arr[mid:]

        # ── Approach momentum features ────────────────────────────────────────
        # Sign convention: positive = price moving TOWARD the line (approach),
        #                  negative = price moving AWAY from the line (retreat).
        # direction="up": price is above the line; approaching = moving DOWN.
        # direction="down": price is below the line; approaching = moving UP.
        is_up = alert.direction == "up"

        def toward(val: float) -> float:
            """Positive = moving toward the line. Call only within this iteration."""
            return -val if is_up else val

        overall_change    = prices_arr[-1] - prices_arr[0]
        approach_momentum = toward(overall_change)   # >0 = consistent approach

        # Linear regression slope — more robust to noise than endpoint diff.
        x = np.arange(n, dtype=float)
        slope = float(np.polyfit(x, prices_arr, 1)[0])
        approach_slope = toward(slope)

        # Sub-window momentum: first half (early approach) and second half (late).
        first_change  = (first_prices[-1]  - first_prices[0])  if len(first_prices)  >= 2 else 0.0
        second_change = (second_prices[-1] - second_prices[0]) if len(second_prices) >= 2 else 0.0
        approach_first  = toward(first_change)
        approach_second = toward(second_change)

        # Acceleration: is approach strengthening (+) or stalling/reversing (-)?
        approach_accel = approach_second - approach_first

        # Volatility and normalized approach momentum.
        volatility    = float(np.std(prices_arr))
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
        first_vol  = int(np.sum(first_sizes))
        second_vol = int(np.sum(second_sizes))
        volume_trend = second_vol - first_vol   # positive = participation increasing
        tick_rate    = n / (FEATURE_SECS / 60)  # trades per minute in the window

        # ── Session context features ───────────────────────────────────────────
        # Look up session high/low at alert time (all computed cumulatively, no look-ahead).
        row_at_alert      = df.loc[:alert_ts, ["session_high", "session_low"]].iloc[-1]
        session_high_now  = float(row_at_alert["session_high"])
        session_low_now   = float(row_at_alert["session_low"])

        # How far has MNQ moved from today's open? Positive = above open (green day).
        session_move_pts = alert.entry_price - day_open

        # Distance from today's session extremes at alert time.
        dist_from_high = session_high_now - alert.entry_price   # 0 = at session high
        dist_from_low  = alert.entry_price - session_low_now    # 0 = at session low

        # minutes since market open (continuous — top feature last run)
        alert_time_mins = alert.alert_time.hour * 60 + alert.alert_time.minute
        alert_mins      = alert_time_mins - (9 * 60 + 30)

        alert.features = {
            # Approach momentum (positive = moving toward line)
            "approach_momentum":  approach_momentum,
            "approach_slope":     approach_slope,
            "approach_first":     approach_first,
            "approach_second":    approach_second,
            "approach_accel":     approach_accel,
            "norm_approach":      norm_approach,
            # Approach quality
            "volatility":         volatility,
            "max_pullback":       max_pullback,
            # Volume / activity
            "volume_trend":       volume_trend,
            "tick_rate":          tick_rate,
            # Time of day (continuous only — buckets showed 0% importance)
            "time_of_day_mins":   alert_mins,
            # Session / market context
            "session_move_pts":   session_move_pts,
            "dist_from_high":     dist_from_high,
            "dist_from_low":      dist_from_low,
            # Level quality
            "level_test_count":   alert.level_test_count,
            # Alert context
            "entry_distance":     abs(alert.entry_price - alert.line_price),
        }

    return alerts


# ── Model training ────────────────────────────────────────────────────────────

def build_model(all_alerts: list[Alert]) -> None:
    """
    Compare multiple classifiers, select the best by ROC-AUC, tune the
    decision threshold to maximize win rate (recall of 'incorrect' class),
    and attach cross-val predictions to alerts for the 'Correctly Avoided' table.
    """
    labeled = [a for a in all_alerts
               if a.outcome in ("correct", "incorrect") and a.features]

    print(f"\n{'─' * 65}")
    print("  MODEL TRAINING")
    print(f"{'─' * 65}")

    n_correct   = sum(1 for a in labeled if a.outcome == "correct")
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
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.3, class_weight="balanced", max_iter=1000, random_state=42,
            )),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=3,
            max_features="sqrt", class_weight="balanced", random_state=42,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=4, random_state=42,
        ),
    }

    print(f"\n  Model comparison ({n_splits}-fold cross-val, ROC-AUC):")
    best_auc   = 0.0
    best_name  = ""
    best_model = None

    for name, model in candidates.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        mean_auc, std_auc = scores.mean(), scores.std()
        flag = "  ← best" if mean_auc > best_auc else ""
        print(f"    {name:<25}  AUC {mean_auc:.3f} ± {std_auc:.3f}{flag}")
        if mean_auc > best_auc:
            best_auc   = mean_auc
            best_name  = name
            best_model = model

    print(f"\n  Selected: {best_name} (AUC {best_auc:.3f})")
    if len(labeled) < 30:
        print(f"  Note: {len(labeled)} samples — treat results as directional, not definitive.")

    # ── Threshold optimisation ────────────────────────────────────────────────
    # Get cross-val probabilities (honest out-of-sample estimates).
    proba      = cross_val_predict(best_model, X, y, cv=cv, method="predict_proba")
    p_incorrect = proba[:, 0]   # P(outcome = incorrect)

    # Show the win-rate trade-off across thresholds so you can pick your comfort level.
    # Lower threshold = more aggressive at flagging trades as incorrect (skip them).
    print(f"\n  Threshold sweep  (predict 'skip trade' when P(incorrect) > threshold):")
    print(f"  {'Threshold':>10}  {'Avoided':>8}  {'Missed good':>12}  {'Win rate':>10}  {'Trades taken':>13}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*13}")

    best_wr        = n_correct / (n_correct + n_incorrect)   # baseline (no model)
    best_threshold = 0.5
    for thr in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]:
        skip         = p_incorrect > thr
        ca           = int((skip & (y == 0)).sum())   # correctly avoided (bad trades skipped)
        ia           = int((skip & (y == 1)).sum())   # incorrectly avoided (good trades skipped)
        rem_correct  = n_correct   - ia
        rem_incorrect = n_incorrect - ca
        total        = rem_correct + rem_incorrect
        wr           = rem_correct / total if total > 0 else 0.0
        taken        = total
        print(f"  {thr:>10.2f}  {ca:>8}  {ia:>12}  {wr:>9.1%}  {taken:>13}")
        if wr > best_wr and rem_correct >= n_correct * 0.6:   # don't skip too many good trades
            best_wr        = wr
            best_threshold = thr

    print(f"\n  Auto-selected threshold: {best_threshold:.2f}  "
          f"(maximises win rate while keeping ≥60% of correct trades)")

    # Attach cv_pred using the chosen threshold.
    for alert, p_inc in zip(labeled, p_incorrect):
        alert.cv_pred = 0 if p_inc > best_threshold else 1

    # ── Feature importances ───────────────────────────────────────────────────
    best_model.fit(X, y)   # type: ignore[union-attr]
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

    # ── Save model ────────────────────────────────────────────────────────────
    joblib.dump({"model": best_model, "threshold": best_threshold, "features": list(X.columns)},
                MODEL_PATH)
    print(f"\n  Model saved to {MODEL_PATH}")
    print(f"  Load with: joblib.load('{MODEL_PATH}')")

    # ── Classification report at chosen threshold ─────────────────────────────
    final_preds = (p_incorrect > best_threshold).astype(int)
    final_preds = 1 - final_preds   # flip: 0=incorrect, 1=correct, but report expects 1=skip
    # Reframe: 1 = "take trade" (correct), 0 = "skip trade" (incorrect)
    print(f"\n  Classification report at threshold {best_threshold:.2f}:")
    print(classification_report(
        y, 1 - (p_incorrect > best_threshold).astype(int),
        target_names=["incorrect (skip)", "correct (take)"],
        zero_division=0,
    ))


# ── Results table ─────────────────────────────────────────────────────────────

def print_results(all_alerts: list[Alert], days: list[datetime.date]) -> None:
    rows = []
    for date in days:
        day       = [a for a in all_alerts if a.date == date]
        correct   = sum(1 for a in day if a.outcome == "correct")
        incorrect = sum(1 for a in day if a.outcome == "incorrect")
        inconc    = sum(1 for a in day if a.outcome == "inconclusive")
        avoided   = sum(1 for a in day if a.outcome == "incorrect" and a.cv_pred == 0)
        rows.append({
            "Date":              str(date),
            "Alerts":            len(day),
            "Correct":           correct,
            "Incorrect":         incorrect,
            "Inconclusive":      inconc,
            "Correctly Avoided": avoided,
        })

    df = pd.DataFrame(rows)
    total_row = {
        "Date":              "TOTAL",
        "Alerts":            df["Alerts"].sum(),
        "Correct":           df["Correct"].sum(),
        "Incorrect":         df["Incorrect"].sum(),
        "Inconclusive":      df["Inconclusive"].sum(),
        "Correctly Avoided": df["Correctly Avoided"].sum(),
    }
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    print(f"\n{'═' * 80}")
    print("  BACKTEST RESULTS — Last 45 Trading Days")
    print(f"{'═' * 80}")
    print(df.to_string(index=False))

    total_correct   = int(total_row["Correct"])
    total_incorrect = int(total_row["Incorrect"])
    total_avoided   = int(total_row["Correctly Avoided"])
    total_decided   = total_correct + total_incorrect

    # Good trades the model incorrectly told you to skip.
    good_skipped = sum(
        1 for a in all_alerts
        if a.outcome == "correct" and a.cv_pred == 0
    )

    print(f"\n{'─' * 65}")
    if total_decided > 0:
        raw_rate = total_correct / total_decided
        print(f"  Win rate — raw (no model)    : {raw_rate:.1%}  "
              f"({total_correct}W / {total_incorrect}L)")

        # Subtract both avoided bad trades AND wrongly skipped good trades.
        adj_correct   = total_correct   - good_skipped
        adj_incorrect = total_incorrect - total_avoided
        adj_decided   = adj_correct + adj_incorrect
        adj_rate      = adj_correct / adj_decided if adj_decided > 0 else 0.0
        print(f"  Win rate — model-filtered    : {adj_rate:.1%}  "
              f"({adj_correct}W / {adj_incorrect}L  |  "
              f"avoided {total_avoided} bad, skipped {good_skipped} good)")
        print(f"\n  'Correctly Avoided' uses cross-val predictions — no in-sample overfitting.")
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
            print(f"  {label:<28}  {w:>5}  {l:>5}  {t:>7}  {wr:>5.1%}")

    # By time bucket (using features dict; fall back to alert_time if no features)
    def time_bucket(a: Alert) -> str:
        mins = a.alert_time.hour * 60 + a.alert_time.minute
        if   (10*60+30) <= mins < (11*60+30): return "10:30–11:30 ET (first hour)"
        elif (11*60+30) <= mins < (13*60):    return "11:30–13:00 ET (lunch)"
        elif (13*60)    <= mins < (15*60):    return "13:00–15:00 ET (afternoon)"
        elif (15*60)    <= mins < (16*60):    return "15:00–16:00 ET (power hour)"
        else:                                  return "other"

    print(f"\n{'─' * 55}")
    print("  WIN RATE BY TIME OF DAY")
    print(f"{'─' * 55}")
    buckets = ["10:30–11:30 ET (first hour)", "11:30–13:00 ET (lunch)",
               "13:00–15:00 ET (afternoon)", "15:00–16:00 ET (power hour)"]
    win_rate_table([(b, [a for a in decided if time_bucket(a) == b]) for b in buckets])

    print(f"\n{'─' * 55}")
    print("  WIN RATE BY LEVEL")
    print(f"{'─' * 55}")
    win_rate_table([(lvl, [a for a in decided if a.level == lvl])
                    for lvl in ["IBH", "IBL", "VWAP"]])

    print(f"\n{'─' * 55}")
    print("  WIN RATE BY LEVEL TEST COUNT (how many times zone entered today)")
    print(f"{'─' * 55}")
    max_count = max((a.level_test_count for a in decided), default=1)
    groups = []
    for n in range(1, min(max_count + 1, 6)):
        label = f"Test #{n}" if n < 5 else "Test #5+"
        subset = [a for a in decided if (a.level_test_count == n if n < 5 else a.level_test_count >= 5)]
        groups.append((label, subset))
    win_rate_table(groups)

    print(f"\n{'─' * 55}")
    print("  WIN RATE BY DAY DIRECTION (MNQ vs open at alert time)")
    print(f"{'─' * 55}")
    green_day = [a for a in decided if a.features.get("session_move_pts", 0) > 0]
    red_day   = [a for a in decided if a.features.get("session_move_pts", 0) <= 0]
    win_rate_table([
        ("Green day (price above open)", green_day),
        ("Red day (price at/below open)", red_day),
    ])

    print(f"\n{'─' * 55}")
    print("  WIN RATE BY DIRECTION")
    print(f"{'─' * 55}")
    win_rate_table([
        ("BUY  (price above line → support)", [a for a in decided if a.direction == "up"]),
        ("SELL (price below line → resistance)", [a for a in decided if a.direction == "down"]),
    ])


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # Pass --offset 45 to run on the prior 45-day period (out-of-sample).
    # Pass --offset 0 (default) for the most recent 45 days.
    offset = 0
    if "--offset" in sys.argv:
        offset = int(sys.argv[sys.argv.index("--offset") + 1])

    days = get_trading_days(45, offset=offset)
    period = f"(offset {offset}d)" if offset else "(most recent)"
    print(f"{'═' * 65}")
    print(f"  MNQ Backtest  |  {days[0]} → {days[-1]}  {period}")
    print(f"  Alert threshold : ±{ALERT_THRESHOLD} pts")
    print(f"  Target          : +{TARGET_POINTS} pts from line within {WINDOW_SECS // 60} min")
    print(f"  Incorrect       : line touched but target not reached within {WINDOW_SECS // 60} min")
    print(f"  Inconclusive    : line never touched within {WINDOW_SECS // 60} min of alert")
    print(f"{'═' * 65}\n")

    client: db.Historical = db.Historical(key=DATABENTO_API_KEY)
    all_alerts: list[Alert] = []

    cum_correct = cum_incorrect = cum_inconc = 0

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

        alerts = simulate_and_evaluate(df, date)
        all_alerts.extend(alerts)

        c = sum(1 for a in alerts if a.outcome == "correct")
        i = sum(1 for a in alerts if a.outcome == "incorrect")
        n = sum(1 for a in alerts if a.outcome == "inconclusive")
        cum_correct   += c
        cum_incorrect += i
        cum_inconc    += n

        print(f"  Trades fetched    : {len(df):,}")
        print(f"  Alerts today      : {len(alerts)}  "
              f"({c} correct  |  {i} incorrect  |  {n} inconclusive)")

        # Per-alert detail for correct and incorrect outcomes.
        decided = [a for a in alerts if a.outcome in ("correct", "incorrect")]
        if decided:
            def fmt(dt: datetime.datetime | None) -> str:
                if dt is None:
                    return "       —"
                return dt.astimezone(PT).strftime("%H:%M:%S")

            print(f"\n  {'Alert(PT)':>9}  {'Level':>5}  {'Line':>8}  {'Entry':>8}  "
                  f"{'Dir':>6}  {'Hit(PT)':>8}  {'Done(PT)':>8}  Outcome")
            print(f"  {'-'*9}  {'-'*5}  {'-'*8}  {'-'*8}  "
                  f"{'-'*6}  {'-'*8}  {'-'*8}  {'-'*9}")
            for a in sorted(decided, key=lambda x: x.alert_time):
                marker = "✓" if a.outcome == "correct" else "✗"
                print(f"  {fmt(a.alert_time):>9}  "
                      f"{a.level:>5}  "
                      f"{a.line_price:>8.2f}  "
                      f"{a.entry_price:>8.2f}  "
                      f"{'↑ BUY' if a.direction == 'up' else '↓ SELL':>6}  "
                      f"{fmt(a.hit_time):>8}  "
                      f"{fmt(a.outcome_time):>8}  "
                      f"{marker} {a.outcome}")
        print()

        print(f"  Cumulative total  : "
              f"{cum_correct} correct  |  {cum_incorrect} incorrect  |  {cum_inconc} inconclusive")

    build_model(all_alerts)
    print_results(all_alerts, days)


if __name__ == "__main__":
    main()
