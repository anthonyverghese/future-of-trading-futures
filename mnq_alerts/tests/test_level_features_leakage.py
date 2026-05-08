"""Critical leakage protection tests. The 3 tests below are non-negotiable.

If any of these fails, the model is unsafe to deploy.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _level_features import compute_all_features


def _ticks_long():
    base = pd.Timestamp("2025-06-02 14:31:00", tz="UTC")
    n = 1000
    rng = np.random.default_rng(42)
    prices = 18000 + np.cumsum(rng.normal(0, 0.5, n))
    sizes = rng.integers(1, 20, n)
    idx = pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s in range(n)])
    return pd.DataFrame({"price": prices, "size": sizes}, index=idx)


def test_no_future_ticks_in_features():
    """For 100 random events, features computed with full ticks must equal
    features computed with future ticks (ts > event_ts) masked out."""
    rng = np.random.default_rng(7)
    ticks = _ticks_long()
    all_levels = {"FIB_0.618": 18000.0, "VWAP": 18001.0}
    for _ in range(100):
        i = int(rng.integers(50, len(ticks) - 50))
        event_ts = ticks.index[i]
        full = compute_all_features(
            ticks=ticks, event_ts=event_ts, event_price=float(ticks.iloc[i]["price"]),
            level_name="FIB_0.618", level_price=18000.0, approach_direction=1,
            prior_touches=[], all_levels=all_levels,
        )
        masked = compute_all_features(
            ticks=ticks.loc[ticks.index <= event_ts], event_ts=event_ts,
            event_price=float(ticks.iloc[i]["price"]), level_name="FIB_0.618",
            level_price=18000.0, approach_direction=1,
            prior_touches=[], all_levels=all_levels,
        )
        for k in full:
            assert full[k] == masked[k], f"Feature {k} differs at event {event_ts}: {full[k]} vs {masked[k]}"


def test_prior_touch_outcome_resolution_order():
    """A prior touch whose resolution_ts > event_ts must NOT influence
    prior_touch_outcome for the event at event_ts."""
    base = pd.Timestamp("2025-06-02 14:31:00", tz="UTC")
    prior_touches = [
        # Future-resolved touch.
        {"event_ts": base + pd.Timedelta(minutes=10), "resolution_ts": base + pd.Timedelta(minutes=25), "outcome": "breakthrough_held"},
    ]
    event_ts = base + pd.Timedelta(minutes=15)  # current; prior touch UNresolved.
    ticks = pd.DataFrame(
        {"price": [18000.0] * 5, "size": [1] * 5},
        index=pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s in (0, 60, 120, 600, 900)]),
    )
    feats = compute_all_features(
        ticks=ticks, event_ts=event_ts, event_price=18000.0,
        level_name="FIB_0.618", level_price=18000.0, approach_direction=1,
        prior_touches=prior_touches, all_levels={"FIB_0.618": 18000.0, "VWAP": 18001.0},
    )
    assert feats["prior_touch_outcome"] == "none"


def test_label_leakage_permutation():
    """Permutation test: random feature values produce random labels.
    Run a tiny LightGBM on randomized features+labels and confirm AUC ~0.5.
    """
    try:
        import lightgbm as lgb
    except (ImportError, OSError) as exc:
        pytest.skip(f"lightgbm not loadable: {exc}")
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(1)
    n = 2000
    X = rng.normal(0, 1, (n, 24))
    y = rng.integers(0, 2, n)
    X_train, X_test = X[:1500], X[1500:]
    y_train, y_test = y[:1500], y[1500:]
    model = lgb.LGBMClassifier(num_leaves=31, max_depth=6, n_estimators=100, learning_rate=0.05, verbosity=-1)
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    assert 0.40 <= auc <= 0.60, f"AUC={auc} suggests pipeline bias on random data"
