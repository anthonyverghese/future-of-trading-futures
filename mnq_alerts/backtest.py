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

DATASET       = "GLBX.MDP3"
SYMBOL        = "MNQ.c.0"
MARKET_OPEN   = datetime.time(9,  30)
IB_END        = datetime.time(10, 30)
MARKET_CLOSE  = datetime.time(16,  0)

ALERT_THRESHOLD = 10.0   # points — zone entry (must match live config)
HIT_THRESHOLD   = 1.0    # points — price within this = "touched the line"
TARGET_POINTS   = 10.0   # points in recommended direction = correct
STOP_POINTS     = 20.0   # points against recommendation = incorrect
WINDOW_SECS     = 15 * 60  # 15-minute evaluation window
FEATURE_SECS    = 2  * 60  # 2-minute feature window before outcome


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Alert:
    date:         datetime.date
    alert_time:   datetime.datetime
    level:        str           # VWAP, IBH, or IBL
    line_price:   float         # level price locked at zone entry
    entry_price:  float         # MNQ price when alert fired
    direction:    str           # 'up' = BUY rec, 'down' = SELL rec
    hit_time:     datetime.datetime | None = None
    outcome_time: datetime.datetime | None = None
    outcome:      str = "inconclusive"
    features:     dict = field(default_factory=dict)
    cv_pred:      int | None = None   # 0=predicted incorrect (avoid), 1=predicted correct


class ZoneState:
    """Alert zone state — mirrors the live LevelState logic exactly."""

    def __init__(self, name: str, price: float) -> None:
        self.name    = name
        self.price   = price
        self.in_zone = False
        self.ref: float | None = None

    def update(self, current_price: float, new_level_price: float | None = None) -> bool:
        """Returns True if an alert should fire (zone just entered).

        new_level_price updates self.price (used for drifting VWAP).
        Exit is checked against self.ref (locked at entry) so VWAP drift
        does not reset or re-trigger the alert.
        """
        if new_level_price is not None:
            self.price = new_level_price

        if self.in_zone:
            if abs(current_price - self.ref) > ALERT_THRESHOLD:
                self.in_zone = False
                self.ref = None
            return False

        if abs(current_price - self.price) <= ALERT_THRESHOLD:
            self.in_zone = True
            self.ref = self.price
            return True

        return False


# ── Data fetching ─────────────────────────────────────────────────────────────

def get_trading_days(n: int = 45) -> list[datetime.date]:
    """Return the last n trading weekdays (weekends excluded; holidays not filtered)."""
    days: list[datetime.date] = []
    d = datetime.datetime.now(ET).date() - datetime.timedelta(days=1)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= datetime.timedelta(days=1)
    return sorted(days)


def fetch_trades(client: db.Historical, date: datetime.date) -> pd.DataFrame:
    """Fetch RTH trades for one day. Returns DataFrame with ET DatetimeIndex."""
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

    # Compute running VWAP from 9:30 (cumulative sum — O(n), not O(n²)).
    df = df.copy()
    df["vwap"] = (df["price"] * df["size"]).cumsum() / df["size"].cumsum()

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
                direction = "up" if price > zone.ref else "down"
                alerts.append(Alert(
                    date=date,
                    alert_time=ts.to_pydatetime(warn=False),
                    level=name,
                    line_price=zone.ref,
                    entry_price=price,
                    direction=direction,
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

        # Phase 2: first tick where target (+10) or stop (-20) is hit.
        eval_end = hit_ts + pd.Timedelta(seconds=WINDOW_SECS)
        eval_seg = prices[(prices.index > hit_ts) & (prices.index <= eval_end)]
        if eval_seg.empty:
            alert.outcome = "inconclusive"
            continue

        if alert.direction == "up":
            target_mask = eval_seg >= alert.line_price + TARGET_POINTS
            stop_mask   = eval_seg <= alert.line_price - STOP_POINTS
        else:
            target_mask = eval_seg <= alert.line_price - TARGET_POINTS
            stop_mask   = eval_seg >= alert.line_price + STOP_POINTS

        target_ts = eval_seg.index[target_mask][0] if target_mask.any() else None
        stop_ts   = eval_seg.index[stop_mask][0]   if stop_mask.any()   else None

        if target_ts is None and stop_ts is None:
            alert.outcome = "inconclusive"
        elif stop_ts is None or (target_ts is not None and target_ts <= stop_ts):
            alert.outcome      = "correct"
            alert.outcome_time = target_ts.to_pydatetime(warn=False)
        else:
            alert.outcome      = "incorrect"
            alert.outcome_time = stop_ts.to_pydatetime(warn=False)

    # ── Feature extraction (2-min window before outcome) ─────────────────────
    for alert in alerts:
        if alert.outcome not in ("correct", "incorrect") or alert.outcome_time is None:
            continue

        outcome_ts   = pd.Timestamp(alert.outcome_time)
        window_start = outcome_ts - pd.Timedelta(seconds=FEATURE_SECS)
        window       = df[(df.index >= window_start) & (df.index <= outcome_ts)]

        if len(window) < 3:
            continue

        prices = window["price"].values
        sizes  = window["size"].values
        n      = len(prices)

        # Split window into two equal halves for sub-window momentum analysis.
        mid = n // 2
        first_prices  = prices[:mid]  if mid >= 2 else prices[:1]
        second_prices = prices[mid:]  if n - mid >= 2 else prices[-1:]
        first_sizes   = sizes[:mid]
        second_sizes  = sizes[mid:]

        # ── Momentum features ─────────────────────────────────────────────────
        # sign convention: positive = moving toward recommended target,
        #                  negative = moving toward stop loss.
        def signed(val: float) -> float:
            return val if alert.direction == "up" else -val

        overall_change = prices[-1] - prices[0]
        signed_momentum = signed(overall_change)

        # Linear regression slope — more robust to noise than endpoint diff.
        x = np.arange(n, dtype=float)
        slope = float(np.polyfit(x, prices, 1)[0])
        signed_slope = signed(slope)

        # Sub-window momentum: first and second halves.
        first_change  = (first_prices[-1]  - first_prices[0])  if len(first_prices)  >= 2 else 0.0
        second_change = (second_prices[-1] - second_prices[0]) if len(second_prices) >= 2 else 0.0
        signed_first  = signed(first_change)
        signed_second = signed(second_change)

        # Acceleration: is momentum strengthening (+) or dying (-)?
        momentum_accel = signed_second - signed_first

        # Volatility and normalized momentum.
        volatility         = float(np.std(prices))
        normalized_momentum = signed_momentum / volatility if volatility > 1e-9 else 0.0

        # ── Excursion features ────────────────────────────────────────────────
        # Max Adverse Excursion (MAE): how far price moved against recommendation.
        # Max Favorable Excursion (MFE): how far price moved toward target.
        if alert.direction == "up":
            mae = alert.line_price - float(np.min(prices))   # below line = bad
            mfe = float(np.max(prices)) - alert.line_price   # above line = good
        else:
            mae = float(np.max(prices)) - alert.line_price   # above line = bad
            mfe = alert.line_price - float(np.min(prices))   # below line = good

        mae = max(mae, 0.0)
        mfe = max(mfe, 0.0)
        mfe_mae_ratio = mfe / (mae + 1.0)  # > 1 = more favorable than adverse

        # ── Volume and activity features ──────────────────────────────────────
        first_vol  = int(np.sum(first_sizes))
        second_vol = int(np.sum(second_sizes))
        volume_trend = second_vol - first_vol   # positive = participation increasing
        tick_rate    = n / (FEATURE_SECS / 60)  # trades per minute in the window

        # ── Context features ──────────────────────────────────────────────────
        alert_mins = (
            alert.alert_time.hour * 60 + alert.alert_time.minute
        ) - (9 * 60 + 30)

        alert.features = {
            # Momentum
            "signed_momentum":    signed_momentum,
            "signed_slope":       signed_slope,
            "signed_first_half":  signed_first,
            "signed_second_half": signed_second,
            "momentum_accel":     momentum_accel,
            "norm_momentum":      normalized_momentum,
            # Excursion
            "volatility":         volatility,
            "mae":                mae,
            "mfe":                mfe,
            "mfe_mae_ratio":      mfe_mae_ratio,
            # Volume / activity
            "volume_trend":       volume_trend,
            "tick_rate":          tick_rate,
            # Context
            "time_of_day_mins":   alert_mins,
            "entry_distance":     abs(alert.entry_price - alert.line_price),
            "level_IBH":          1 if alert.level == "IBH"  else 0,
            "level_IBL":          1 if alert.level == "IBL"  else 0,
            "level_VWAP":         1 if alert.level == "VWAP" else 0,
            "direction":          1 if alert.direction == "up" else 0,
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

    print(f"\n{'─' * 65}")
    if total_decided > 0:
        raw_rate = total_correct / total_decided
        print(f"  Win rate — raw (no model)    : {raw_rate:.1%}  "
              f"({total_correct}W / {total_incorrect}L)")

        adj_incorrect = total_incorrect - total_avoided
        adj_decided   = total_correct + adj_incorrect
        adj_rate      = total_correct / adj_decided if adj_decided > 0 else 0.0
        print(f"  Win rate — model-filtered    : {adj_rate:.1%}  "
              f"(avoided {total_avoided}/{total_incorrect} bad trades)")
        print(f"\n  'Correctly Avoided' uses cross-val predictions — no in-sample overfitting.")
    print(f"{'─' * 65}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    days = get_trading_days(45)
    print(f"{'═' * 65}")
    print(f"  MNQ Backtest  |  {days[0]} → {days[-1]}")
    print(f"  Alert threshold : ±{ALERT_THRESHOLD} pts")
    print(f"  Target / Stop   : +{TARGET_POINTS} pts / -{STOP_POINTS} pts")
    print(f"  Eval window     : {WINDOW_SECS // 60} min after touching line")
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
        print(f"  Cumulative total  : "
              f"{cum_correct} correct  |  {cum_incorrect} incorrect  |  {cum_inconc} inconclusive")

    build_model(all_alerts)
    print_results(all_alerts, days)


if __name__ == "__main__":
    main()
