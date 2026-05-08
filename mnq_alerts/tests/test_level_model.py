"""Tests for _level_model."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("lightgbm")
from _level_model import train_architecture_a, predict_architecture_a, FEATURE_COLUMNS


def _synthetic_dataset(n=6000, seed=0):
    """Synthetic dataset with enough rows per (level, direction, tp, sl) cell
    to clear the 100-row training minimum.

    With n=6000: 1500 train rows spread over 3 levels × 2 directions × 4 TP/SL
    = 24 cells → ~250 rows/cell in train, well above the 100-row floor.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        feats = {col: rng.normal() for col in FEATURE_COLUMNS}
        rows.append({
            "level_name": rng.choice(["FIB_0.618", "FIB_0.236", "IBH"]),
            "direction": rng.choice(["bounce", "breakthrough"]),
            "tp": int(rng.choice([8, 10])),
            "sl": int(rng.choice([20, 25])),
            "label": int(rng.integers(0, 2)),
            "event_ts": pd.Timestamp("2025-06-01", tz="UTC") + pd.Timedelta(days=i // 50),
            # prior_touch_outcome is required by CATEGORICAL_FEATURES
            "prior_touch_outcome": rng.choice(["win", "loss", "none"]),
            **feats,
        })
    return pd.DataFrame(rows)


def test_train_architecture_a_returns_dict_of_models():
    df = _synthetic_dataset()
    models = train_architecture_a(df.iloc[:4500], val=df.iloc[4500:5400])
    # Up to 3 levels × 2 directions × 4 TP/SL = 24 in synthetic data.
    assert len(models) > 0
    for key, m in models.items():
        level, direction, tp, sl = key
        assert level in {"FIB_0.618", "FIB_0.236", "IBH"}
        assert direction in {"bounce", "breakthrough"}


def test_predict_architecture_a_returns_probability_per_variant():
    df = _synthetic_dataset()
    models = train_architecture_a(df.iloc[:4500], val=df.iloc[4500:5400])
    test_event = df.iloc[5400].to_dict()
    preds = predict_architecture_a(models, test_event)
    # 8 variants per event: 2 directions × 2 tps × 2 sls.
    assert len(preds) == 8
    for p in preds.values():
        assert 0 <= p <= 1
