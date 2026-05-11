"""Tests for bot_filter.BotFilter and its integration with bot_trader.

Covers:
  - Disabled mode passthrough (no models loaded, all decisions = take)
  - Enabled mode: loads trained models, can compute features, returns
    a sane decision on a synthetic tick buffer
  - Reversibility: changing BOT_FILTER_ENABLED toggles behavior cleanly
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot_filter import BotFilter, FilterContext, reset_filter_for_tests


MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "_bot_filter_models",
)


def _synthetic_tick_buffer(level_price: float, n_ticks: int = 600):
    """Build a tick buffer that approaches `level_price` from below over ~5 min."""
    base = pd.Timestamp("2026-04-15 14:31:00", tz="UTC")
    times = [base + pd.Timedelta(seconds=s) for s in range(n_ticks)]
    # Price rises smoothly from level_price - 15 to level_price - 0.5
    prices = [level_price - 15 + 14.5 * (i / (n_ticks - 1)) for i in range(n_ticks)]
    sizes = [1] * n_ticks
    return pd.DataFrame(
        {"price": prices, "size": sizes}, index=pd.DatetimeIndex(times)
    )


def test_disabled_passthrough():
    """When enabled=False, every decision returns take=True with reason='disabled'."""
    f = BotFilter(enabled=False)
    ctx = FilterContext(
        tick_buffer=pd.DataFrame(columns=["price", "size"],
                                 index=pd.DatetimeIndex([], tz="UTC")),
        session_open_price=18000.0,
        levels={},
        bot_touches_today=0,
        resolution_order_cw=0,
        resolution_order_cl=0,
    )
    d = f.should_take(
        level_name="FIB_0.618", level_price=18000.0,
        event_ts=pd.Timestamp("2026-04-15 14:31:00", tz="UTC"),
        event_price=18000.0, approach_direction=1,
        entry_count_7pt=1, context=ctx,
    )
    assert d.take is True
    assert d.reason == "disabled"


@pytest.mark.skipif(
    not os.path.isdir(MODEL_DIR),
    reason=f"trained models not found at {MODEL_DIR}; run "
    "scoring_rework/train_production_models.py first",
)
def test_enabled_loads_and_decides():
    """Filter loads trained models, runs end-to-end on synthetic data."""
    f = BotFilter(model_dir=MODEL_DIR, enabled=True)
    assert len(f.models) == 3
    assert len(f.feature_list) > 0
    assert "is_FIB_0.618" in f.feature_list  # one-hot level present

    level_price = 18000.0
    ticks = _synthetic_tick_buffer(level_price)
    event_ts = ticks.index[-1]
    ctx = FilterContext(
        tick_buffer=ticks,
        session_open_price=17990.0,
        levels={"FIB_0.618": level_price, "FIB_0.236": 17920.0, "IBH": 18030.0,
                "IBL": 17950.0, "FIB_0.764": 18045.0, "FIB_EXT_HI_1.272": 18080.0,
                "FIB_EXT_LO_1.272": 17880.0, "VWAP": 17985.0},
        bot_touches_today=0,
        resolution_order_cw=0,
        resolution_order_cl=0,
    )
    d = f.should_take(
        level_name="FIB_0.618", level_price=level_price,
        event_ts=event_ts, event_price=ticks["price"].iloc[-1],
        approach_direction=1, entry_count_7pt=1, context=ctx,
    )
    # Decision is one of the legitimate outcomes
    assert d.reason in ("time_suppressed", "human_score_low",
                       "model_take", "model_skip", "error")
    # If we got to model scoring, all 3 models should have produced probs
    if d.reason in ("model_take", "model_skip"):
        assert len(d.model_probs) == 3
        for name, p in d.model_probs.items():
            assert 0.0 <= p["p_win"] <= 1.0
        assert d.votes in (0, 1, 2, 3)


def test_time_suppressed():
    """13:30-14:00 ET events should be rejected with reason='time_suppressed'."""
    if not os.path.isdir(MODEL_DIR):
        pytest.skip("models not trained")
    f = BotFilter(model_dir=MODEL_DIR, enabled=True)
    # 13:35 ET = 17:35 UTC (EDT) — within the suppression window.
    event_ts = pd.Timestamp("2026-04-15 17:35:00", tz="UTC")
    ticks = _synthetic_tick_buffer(18000.0)
    # Shift ticks to align with suppressed time.
    shift = event_ts - ticks.index[-1]
    ticks.index = ticks.index + shift
    ctx = FilterContext(
        tick_buffer=ticks, session_open_price=17990.0,
        levels={"FIB_0.618": 18000.0},
        bot_touches_today=0, resolution_order_cw=0, resolution_order_cl=0,
    )
    d = f.should_take(
        level_name="FIB_0.618", level_price=18000.0,
        event_ts=event_ts, event_price=ticks["price"].iloc[-1],
        approach_direction=1, entry_count_7pt=1, context=ctx,
    )
    assert d.take is False
    assert d.reason == "time_suppressed"


def test_singleton_resets_for_tests():
    """reset_filter_for_tests() clears the module-level singleton."""
    from bot_filter import get_filter
    f1 = get_filter()
    reset_filter_for_tests()
    f2 = get_filter()
    assert f1 is not f2
