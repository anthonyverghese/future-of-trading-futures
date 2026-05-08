"""Walk-forward folds, gates, and architecture selection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FINAL_TEST_TRADING_DAYS = 30
EMBARGO_DAYS = 1


@dataclass
class FoldDef:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_walk_forward_folds(trading_days: pd.DatetimeIndex) -> list[FoldDef]:
    """Quarterly walk-forward folds over the dev set.

    Final FINAL_TEST_TRADING_DAYS days are reserved for out-of-time test (excluded).
    EMBARGO_DAYS days dropped from the start of each test fold.
    """
    if len(trading_days) <= FINAL_TEST_TRADING_DAYS:
        return []
    dev_days = trading_days[:-FINAL_TEST_TRADING_DAYS]
    quarters = sorted({(d.year, (d.month - 1) // 3 + 1) for d in dev_days})
    folds: list[FoldDef] = []
    for i in range(1, len(quarters)):
        train_quarters = quarters[:i]
        train_days = [d for d in dev_days if (d.year, (d.month - 1) // 3 + 1) in train_quarters]
        test_days = [d for d in dev_days if (d.year, (d.month - 1) // 3 + 1) == quarters[i]]
        if len(train_days) < 20 or len(test_days) < 5:
            continue
        test_days_emb = test_days[EMBARGO_DAYS:]
        if len(test_days_emb) < 5:
            continue
        folds.append(FoldDef(
            train_start=train_days[0],
            train_end=train_days[-1],
            test_start=test_days_emb[0],
            test_end=test_days_emb[-1],
        ))
    return folds


@dataclass
class GateResults:
    top_decile_lift: float
    top_decile_expected_pnl_per_trade: float
    simulated_mean_daily_pnl_dollars: float
    base_rate: float


def score_fold(events: list[dict], preds: dict, threshold: float) -> GateResults:
    """Compute gate metrics for a single fold's test results."""
    from _level_simulation import simulate_strategy

    rows = []
    for ev in events:
        ev_preds = preds.get(ev["event_ts"], {})
        key = (ev["direction"], ev["tp"], ev["sl"])
        if key not in ev_preds:
            continue
        rows.append({
            "p_win": ev_preds[key], "label": ev["label"], "tp": ev["tp"], "sl": ev["sl"],
        })
    if not rows:
        return GateResults(0.0, 0.0, 0.0, 0.0)
    df = pd.DataFrame(rows)
    df = df.sort_values("p_win", ascending=False).reset_index(drop=True)
    base_rate = float(df["label"].mean())
    cutoff = max(1, len(df) // 10)
    top = df.iloc[:cutoff]
    top_wr = float(top["label"].mean())
    top_decile_lift = top_wr - base_rate
    expected_pnl = float((top["tp"] * top["label"] - top["sl"] * (1 - top["label"])).mean())

    sim = simulate_strategy(events, preds, threshold=threshold)
    daily = sim["daily_pnl_dollars"]
    mean_daily = float(np.mean(list(daily.values()))) if daily else 0.0

    return GateResults(top_decile_lift, expected_pnl, mean_daily, base_rate)


def run_architecture_selection(
    dataset: pd.DataFrame, folds: list[FoldDef], v6_per_quarter_pnl: dict,
) -> dict:
    """Train A and B per fold, evaluate gates, return per-architecture per-fold results."""
    from _level_model import train_architecture_a, predict_architecture_a
    from _level_model import train_architecture_b, predict_architecture_b

    results: dict = {"A": {"folds": []}, "B": {"folds": []}}
    for fold in folds:
        train = dataset[dataset["event_ts"] <= fold.train_end].copy()
        test = dataset[(dataset["event_ts"] >= fold.test_start) & (dataset["event_ts"] <= fold.test_end)].copy()
        train_days = sorted(train["event_ts"].dt.date.unique())
        val_cutoff = train_days[-max(1, len(train_days) // 20)]
        val = train[train["event_ts"].dt.date >= val_cutoff].copy()
        train_for_fit = train[train["event_ts"].dt.date < val_cutoff].copy()

        models_a = train_architecture_a(train_for_fit, val)
        preds_a: dict = {}
        for _, ev in test.drop_duplicates("event_ts").iterrows():
            preds_a[ev["event_ts"]] = predict_architecture_a(models_a, ev.to_dict())
        gate_a = score_fold(test.to_dict("records"), preds_a, threshold=0.0)
        results["A"]["folds"].append({"fold": fold, "gates": gate_a})

        model_b = train_architecture_b(train_for_fit, val)
        preds_b: dict = {}
        for _, ev in test.drop_duplicates("event_ts").iterrows():
            preds_b[ev["event_ts"]] = predict_architecture_b(model_b, ev.to_dict())
        gate_b = score_fold(test.to_dict("records"), preds_b, threshold=0.0)
        results["B"]["folds"].append({"fold": fold, "gates": gate_b})

    for arch in ("A", "B"):
        folds_data = results[arch]["folds"]
        lift_pass = sum(1 for f in folds_data if f["gates"].top_decile_lift > 0)
        epnl_pass = sum(1 for f in folds_data if f["gates"].top_decile_expected_pnl_per_trade > 0)
        beats_v6 = 0
        for f in folds_data:
            q = (f["fold"].test_start.year, (f["fold"].test_start.month - 1) // 3 + 1)
            v6 = v6_per_quarter_pnl.get(q, 0.0)
            if f["gates"].simulated_mean_daily_pnl_dollars > v6:
                beats_v6 += 1
        results[arch]["passes_lift_gate"] = lift_pass >= 4
        results[arch]["passes_expected_pnl_gate"] = epnl_pass == len(folds_data)
        results[arch]["passes_v6_gate"] = beats_v6 >= 4

    return results
