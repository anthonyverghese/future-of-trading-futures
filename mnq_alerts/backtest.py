"""
backtest.py — Backtests the MNQ alert system over the last 20 trading days.

Fetches historical trade data from Databento, simulates the exact alert
logic (VWAP, IBH/IBL, zone states), evaluates outcomes, trains a
RandomForest classifier to identify patterns that predict incorrect outcomes,
and prints a summary table with a "Correctly Avoided" column.

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
import numpy as np
import pandas as pd
import pytz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.dirname(__file__))
from config import DATABENTO_API_KEY

ET = pytz.timezone("America/New_York")

DATASET       = "GLBX.MDP3"
SYMBOL        = "MNQ.c.0"
MARKET_OPEN   = datetime.time(9,  30)
IB_END        = datetime.time(10, 30)
MARKET_CLOSE  = datetime.time(16,  0)

ALERT_THRESHOLD = 15.0   # points — zone entry (must match live config)
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
    cv_pred:      int | None = None   # set after cross-val: 1=correct, 0=incorrect


class ZoneState:
    """Alert zone state — mirrors the live LevelState logic exactly."""

    def __init__(self, name: str, price: float) -> None:
        self.name      = name
        self.price     = price
        self.in_zone   = False
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
            self.ref = self.price  # lock reference at moment of entry
            return True

        return False


# ── Data fetching ─────────────────────────────────────────────────────────────

def get_trading_days(n: int = 20) -> list[datetime.date]:
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

    for ts, row in post_ib.iterrows():
        price = row["price"]
        vwap  = row["vwap"]
        level_prices = {"IBH": ibh, "IBL": ibl, "VWAP": vwap}

        for name, zone in zones.items():
            if zone.update(price, new_level_price=level_prices[name]):
                direction = "up" if price > zone.ref else "down"
                alerts.append(Alert(
                    date=date,
                    alert_time=ts.to_pydatetime(),
                    level=name,
                    line_price=zone.ref,
                    entry_price=price,
                    direction=direction,
                ))

    # ── Outcome evaluation ────────────────────────────────────────────────────
    for alert in alerts:
        alert_ts   = pd.Timestamp(alert.alert_time)
        window_end = alert_ts + pd.Timedelta(seconds=WINDOW_SECS)
        forward    = df[df.index > alert_ts]

        # Phase 1: wait for price to touch the line within 15 min of alert.
        hit_ts = None
        for ts, row in forward.iterrows():
            if ts > window_end:
                break
            if abs(row["price"] - alert.line_price) <= HIT_THRESHOLD:
                hit_ts = ts
                alert.hit_time = ts.to_pydatetime()
                break

        if hit_ts is None:
            alert.outcome = "inconclusive"
            continue

        # Phase 2: wait for target (+10) or stop (-20) within 15 min of hit.
        eval_end = hit_ts + pd.Timedelta(seconds=WINDOW_SECS)
        post_hit = df[df.index > hit_ts]

        for ts, row in post_hit.iterrows():
            if ts > eval_end:
                break
            p = row["price"]
            if alert.direction == "up":
                if p >= alert.line_price + TARGET_POINTS:
                    alert.outcome      = "correct"
                    alert.outcome_time = ts.to_pydatetime()
                    break
                elif p <= alert.line_price - STOP_POINTS:
                    alert.outcome      = "incorrect"
                    alert.outcome_time = ts.to_pydatetime()
                    break
            else:
                if p <= alert.line_price - TARGET_POINTS:
                    alert.outcome      = "correct"
                    alert.outcome_time = ts.to_pydatetime()
                    break
                elif p >= alert.line_price + STOP_POINTS:
                    alert.outcome      = "incorrect"
                    alert.outcome_time = ts.to_pydatetime()
                    break
        else:
            alert.outcome = "inconclusive"

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

        # signed_momentum: positive = moving toward recommended target,
        # negative = moving against (toward stop loss).
        price_change     = prices[-1] - prices[0]
        signed_momentum  = price_change if alert.direction == "up" else -price_change

        # Max adverse excursion: how far price moved against recommendation in window.
        if alert.direction == "up":
            mae = alert.line_price - float(np.min(prices))
        else:
            mae = float(np.max(prices)) - alert.line_price

        alert_mins = (alert.alert_time.hour * 60 + alert.alert_time.minute) - (9 * 60 + 30)

        alert.features = {
            "signed_momentum":       signed_momentum,
            "volatility":            float(np.std(prices)),
            "max_adverse_excursion": max(mae, 0.0),
            "volume":                int(np.sum(sizes)),
            "trade_count":           len(prices),
            "time_of_day_mins":      alert_mins,
            "level_IBH":             1 if alert.level == "IBH"  else 0,
            "level_IBL":             1 if alert.level == "IBL"  else 0,
            "level_VWAP":            1 if alert.level == "VWAP" else 0,
            "direction":             1 if alert.direction == "up" else 0,
            "entry_distance":        abs(alert.entry_price - alert.line_price),
        }

    return alerts


# ── Model training ────────────────────────────────────────────────────────────

def build_model(all_alerts: list[Alert]) -> None:
    """
    Train a RandomForest on correct vs incorrect alerts.
    Attaches cross-val predictions (cv_pred) to each alert so the
    'Correctly Avoided' column in the table reflects honest out-of-sample
    estimates rather than in-sample overfitting.
    """
    labeled = [a for a in all_alerts
               if a.outcome in ("correct", "incorrect") and a.features]

    print(f"\n{'─' * 60}")
    print("  MODEL TRAINING")
    print(f"{'─' * 60}")

    n_correct   = sum(1 for a in labeled if a.outcome == "correct")
    n_incorrect = sum(1 for a in labeled if a.outcome == "incorrect")
    print(f"  Labeled samples : {len(labeled)} "
          f"({n_correct} correct, {n_incorrect} incorrect)")

    if len(labeled) < 6 or min(n_correct, n_incorrect) < 2:
        print("  Insufficient samples for reliable model — skipping.")
        return

    X = pd.DataFrame([a.features for a in labeled])
    y = np.array([1 if a.outcome == "correct" else 0 for a in labeled])

    n_splits = min(5, min(n_correct, n_incorrect))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    try:
        preds = cross_val_predict(
            RandomForestClassifier(n_estimators=200, random_state=42,
                                   class_weight="balanced"),
            X, y, cv=cv,
        )
    except Exception as e:
        print(f"  Cross-validation failed: {e}")
        return

    acc = (preds == y).mean()
    print(f"  Cross-val accuracy ({n_splits}-fold): {acc:.1%}")
    print(f"  Note: {len(labeled)} samples is a small dataset — treat results as directional.")

    for alert, pred in zip(labeled, preds):
        alert.cv_pred = int(pred)

    # Train final model on all data for feature importance display.
    clf = RandomForestClassifier(n_estimators=200, random_state=42,
                                 class_weight="balanced")
    clf.fit(X, y)
    imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)

    print("\n  Feature importances (what the model looks at):")
    for feat, val in imp.items():
        bar = "█" * max(1, int(val * 50))
        print(f"    {feat:<30} {val:>5.1%}  {bar}")

    print(f"\n  Key insight: 'signed_momentum' measures how strongly price")
    print(f"  was moving toward the target in the 2 min before outcome.")
    print(f"  Negative = moving toward stop loss (→ likely incorrect).")


# ── Results table ─────────────────────────────────────────────────────────────

def print_results(all_alerts: list[Alert], days: list[datetime.date]) -> None:
    rows = []
    for date in days:
        day = [a for a in all_alerts if a.date == date]
        correct   = sum(1 for a in day if a.outcome == "correct")
        incorrect = sum(1 for a in day if a.outcome == "incorrect")
        inconc    = sum(1 for a in day if a.outcome == "inconclusive")
        avoided   = sum(1 for a in day if a.outcome == "incorrect" and a.cv_pred == 0)
        rows.append({
            "Date":                str(date),
            "Alerts":              len(day),
            "Correct":             correct,
            "Incorrect":           incorrect,
            "Inconclusive":        inconc,
            "Correctly Avoided":   avoided,
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

    print(f"\n{'─' * 60}")
    if total_decided > 0:
        raw_rate = total_correct / total_decided
        print(f"  Win rate (raw, excludes inconclusive) : "
              f"{raw_rate:.1%}  ({total_correct}W / {total_incorrect}L)")

        if total_avoided > 0:
            adj_incorrect = total_incorrect - total_avoided
            adj_decided   = total_correct + adj_incorrect
            adj_rate      = total_correct / adj_decided if adj_decided > 0 else 0.0
            print(f"  Win rate (model-filtered)             : "
                  f"{adj_rate:.1%}  "
                  f"(avoided {total_avoided}/{total_incorrect} bad trades)")
            print(f"\n  'Correctly Avoided' = model predicted incorrect AND it was.")
            print(f"  Uses cross-val predictions to avoid in-sample overfitting.")
    print(f"{'─' * 60}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    days = get_trading_days(45)
    print(f"{'═' * 60}")
    print(f"  MNQ Backtest  |  {days[0]} → {days[-1]}")
    print(f"  Alert threshold : ±{ALERT_THRESHOLD} pts")
    print(f"  Target / Stop   : +{TARGET_POINTS} pts / -{STOP_POINTS} pts")
    print(f"  Eval window     : {WINDOW_SECS // 60} min")
    print(f"{'═' * 60}\n")

    client: db.Historical = db.Historical(key=DATABENTO_API_KEY)
    all_alerts: list[Alert] = []

    for date in days:
        print(f"  {date} ...", end="  ", flush=True)
        try:
            df = fetch_trades(client, date)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        if df.empty:
            print("no data")
            continue

        alerts = simulate_and_evaluate(df, date)
        all_alerts.extend(alerts)

        c = sum(1 for a in alerts if a.outcome == "correct")
        i = sum(1 for a in alerts if a.outcome == "incorrect")
        n = sum(1 for a in alerts if a.outcome == "inconclusive")
        print(f"{len(df):>7,} trades  →  {len(alerts):>2} alerts  "
              f"({c} correct  {i} incorrect  {n} inconclusive)")

    build_model(all_alerts)
    print_results(all_alerts, days)


if __name__ == "__main__":
    main()
