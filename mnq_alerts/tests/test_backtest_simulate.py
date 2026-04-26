"""Tests for backtest simulation — verify core logic matches live bot rules.

These tests catch silent bugs that would corrupt backtest results:
- Trade evaluation (target/stop/timeout)
- Zone behavior
- Filters (time window, vol, score, daily loss, 1-position)
- IB level calculation
"""

import sys
import os
import datetime

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

ET = pytz.timezone("America/New_York")


# ---------------------------------------------------------------------------
# Helpers — build minimal DayCache and DayArrays for testing
# ---------------------------------------------------------------------------

def _make_tick_data(
    date: datetime.date,
    prices: list[float],
    start_time: datetime.time = datetime.time(10, 31, 0),
    tick_interval_ms: int = 100,
) -> tuple:
    """Build minimal DayCache and DayArrays from a price sequence.

    All ticks are placed after IB close (10:31) unless overridden.
    IB levels are set from the first 2 prices (min/max).
    """
    from targeted_backtest import DayCache
    from backtest.data import DayArrays

    n = len(prices)
    base_dt = ET.localize(datetime.datetime.combine(date, start_time))
    base_ns = int(base_dt.timestamp() * 1e9)

    ts_ns = np.array([base_ns + i * tick_interval_ms * 1_000_000 for i in range(n)], dtype=np.int64)
    prices_arr = np.array(prices, dtype=np.float64)

    # IB levels: use a fixed range.
    ibh = max(prices) + 50
    ibl = min(prices) - 50
    ib_range = ibh - ibl

    # Dummy VWAP (flat at midpoint).
    vwap_val = (ibh + ibl) / 2
    vwaps = np.full(n, vwap_val, dtype=np.float64)

    dc = DayCache(
        date=date,
        ibh=ibh,
        ibl=ibl,
        fib_lo=ibl - 0.272 * ib_range,
        fib_hi=ibh + 0.272 * ib_range,
        post_ib_prices=prices_arr,
        post_ib_vwaps=vwaps,
        post_ib_timestamps=pd.DatetimeIndex(
            pd.to_datetime(ts_ns, unit="ns", utc=True).tz_convert(ET)
        ),
        post_ib_start_idx=0,
        full_prices=prices_arr,
        full_ts_ns=ts_ns,
        full_df=pd.DataFrame(),
    )

    # Build arrays with realistic values.
    et_dt = base_dt
    utc_off = int(et_dt.utcoffset().total_seconds() * 1e9)
    et_mins = ((ts_ns + utc_off) // 60_000_000_000 % 1440).astype(np.int32)

    arrays = DayArrays(
        tick_rates=np.full(n, 1000.0),    # normal tick rate
        range_30m_pts=np.full(n, 100.0),  # normal volatility
        approach_speed=np.full(n, 2.0),
        tick_density=np.full(n, 10.0),
        et_mins=et_mins,
        session_move=prices_arr - prices_arr[0],
    )

    return dc, arrays


# ---------------------------------------------------------------------------
# 1. Trade evaluation
# ---------------------------------------------------------------------------


class TestEvaluateBotTrade:
    def test_win_up(self):
        """BUY at line, price reaches target → win."""
        from bot_risk_backtest import evaluate_bot_trade

        prices = np.array([100.0, 101.0, 105.0, 108.5], dtype=np.float64)
        ts = np.array([0, 1_000_000_000, 2_000_000_000, 3_000_000_000], dtype=np.int64)

        out, exit_idx, pnl = evaluate_bot_trade(
            0, 100.0, "up", ts, prices, 8.0, 20.0, 900, None,
        )
        assert out == "win"
        assert exit_idx == 3  # price >= 108.0 at index 3
        assert pnl > 0

    def test_loss_up(self):
        """BUY at line, price drops to stop → loss."""
        from bot_risk_backtest import evaluate_bot_trade

        prices = np.array([100.0, 95.0, 80.0, 79.5], dtype=np.float64)
        ts = np.array([0, 1_000_000_000, 2_000_000_000, 3_000_000_000], dtype=np.int64)

        out, exit_idx, pnl = evaluate_bot_trade(
            0, 100.0, "up", ts, prices, 8.0, 20.0, 900, None,
        )
        assert out == "loss"
        assert pnl < 0

    def test_win_down(self):
        """SELL at line, price drops to target → win."""
        from bot_risk_backtest import evaluate_bot_trade

        prices = np.array([100.0, 95.0, 92.0, 91.5], dtype=np.float64)
        ts = np.array([0, 1_000_000_000, 2_000_000_000, 3_000_000_000], dtype=np.int64)

        out, exit_idx, pnl = evaluate_bot_trade(
            0, 100.0, "down", ts, prices, 8.0, 20.0, 900, None,
        )
        assert out == "win"
        assert pnl > 0

    def test_timeout(self):
        """Neither target nor stop hit within window → timeout."""
        from bot_risk_backtest import evaluate_bot_trade

        prices = np.array([100.0, 100.5, 99.5, 100.0], dtype=np.float64)
        ts = np.array([0, 1_000_000_000, 2_000_000_000, 3_000_000_000], dtype=np.int64)

        out, exit_idx, pnl = evaluate_bot_trade(
            0, 100.0, "up", ts, prices, 8.0, 20.0, 2, None,  # 2s window
        )
        assert out == "timeout"

    def test_eod_cutoff(self):
        """EOD cutoff stops evaluation early."""
        from bot_risk_backtest import evaluate_bot_trade

        prices = np.array([100.0, 100.5, 108.5], dtype=np.float64)
        ts = np.array([0, 1_000_000_000, 2_000_000_000], dtype=np.int64)

        # Cutoff at 1.5s — target hit is at 2s, after cutoff.
        out, _, _ = evaluate_bot_trade(
            0, 100.0, "up", ts, prices, 8.0, 20.0, 900, 1_500_000_000,
        )
        assert out == "timeout"  # target not reached before cutoff


# ---------------------------------------------------------------------------
# 2. Simulate day — filter checks
# ---------------------------------------------------------------------------


class TestSimulateDayFilters:
    def _run(self, prices, **kwargs):
        """Run simulate_day with given prices and return trades."""
        from backtest.simulate import simulate_day
        from backtest.zones import BotZoneTradeReset

        date = datetime.date(2026, 4, 1)
        dc, arrays = _make_tick_data(date, prices)

        def zone_factory(name, price, drifts):
            return BotZoneTradeReset(price, drifts)

        defaults = dict(
            zone_factory=zone_factory,
            target_fn=lambda lv: 8.0,
            stop_pts=20.0,
            max_per_level=12,
            weights=None,
            min_score=-99,
            streak_state=(0, 0),
            daily_loss=100.0,
            max_consec=999,
        )
        defaults.update(kwargs)

        trades, streak = simulate_day(dc, arrays, **defaults)
        return trades

    def test_1330_suppression(self):
        """Trades during 13:30-14:00 ET should be suppressed."""
        from backtest.simulate import simulate_day
        from backtest.zones import BotZoneTradeReset
        from targeted_backtest import DayCache
        from backtest.data import DayArrays

        date = datetime.date(2026, 4, 1)
        # Place ticks at 13:30 ET (inside suppressed window)
        dc, arrays = _make_tick_data(
            date, [27000.0] * 100,
            start_time=datetime.time(13, 30, 0),
        )
        # Set IBH to 27000 so price is within 1pt
        dc.ibh = 27000.0

        def zone_factory(name, price, drifts):
            return BotZoneTradeReset(price, drifts)

        trades, _ = simulate_day(
            dc, arrays, zone_factory,
            lambda lv: 8.0, 20.0, 12,
            weights=None, min_score=-99,
        )
        # All entries should be suppressed — 0 trades
        ibh_trades = [t for t in trades if t.level == "IBH"]
        assert len(ibh_trades) == 0

    def test_vol_filter(self):
        """Trades during low volatility should be skipped."""
        from backtest.simulate import simulate_day
        from backtest.zones import BotZoneTradeReset
        from backtest.data import DayArrays

        date = datetime.date(2026, 4, 1)
        dc, arrays = _make_tick_data(date, [27000.0] * 100)
        dc.ibh = 27000.0

        # Set range_30m very low (< 0.15% of price)
        # 0.15% of 27000 = 40.5. Set to 10 (well below threshold).
        arrays.range_30m_pts = np.full(len(arrays.range_30m_pts), 10.0)

        def zone_factory(name, price, drifts):
            return BotZoneTradeReset(price, drifts)

        trades, _ = simulate_day(
            dc, arrays, zone_factory,
            lambda lv: 8.0, 20.0, 12,
            weights=None, min_score=-99,
        )
        ibh_trades = [t for t in trades if t.level == "IBH"]
        assert len(ibh_trades) == 0

    def test_daily_loss_limit_stops_trading(self):
        """After hitting daily loss limit, no more trades."""
        # Create prices that will produce losses
        # IBH at 27050, prices oscillate just above/below 27050
        date = datetime.date(2026, 4, 1)
        # Price approaches IBH, then drops 25 pts (loss), repeats
        prices = []
        for _ in range(20):
            prices.extend([27050.0, 27049.0, 27025.0])  # approach, enter, stop
        dc, arrays = _make_tick_data(date, prices, tick_interval_ms=1000)
        dc.ibh = 27050.0

        trades = self._run(prices, daily_loss=50.0)
        # With $50 daily loss limit, should stop after 1-2 losses
        # Each loss is ~(20+1.24)*2 = ~$42.48
        total_loss = sum(t.pnl_usd for t in trades if t.pnl_usd < 0)
        assert total_loss >= -100  # shouldn't lose more than ~$100 (2 losses)

    def test_one_position_at_a_time(self):
        """Only one trade can be open at any time."""
        trades = self._run([27000.0] * 1000)
        # Verify no overlapping trades
        for i in range(1, len(trades)):
            assert trades[i].entry_idx > trades[i - 1].exit_idx, (
                f"Trade {i} entry ({trades[i].entry_idx}) overlaps "
                f"trade {i-1} exit ({trades[i-1].exit_idx})"
            )


# ---------------------------------------------------------------------------
# 3. IB level calculation
# ---------------------------------------------------------------------------


class TestIBLevels:
    def test_ib_includes_1030_ticks(self):
        """IB levels should include 10:30:xx ticks (before 10:31)."""
        from targeted_backtest import preprocess_day

        date = datetime.date(2026, 4, 1)
        # Create ticks: one at 10:29 (low), one at 10:30:30 (high), one at 10:31 (post-IB)
        times = [
            ET.localize(datetime.datetime(2026, 4, 1, 9, 30, 0)),
            ET.localize(datetime.datetime(2026, 4, 1, 10, 29, 0)),
            ET.localize(datetime.datetime(2026, 4, 1, 10, 30, 30)),
            ET.localize(datetime.datetime(2026, 4, 1, 10, 31, 0)),
        ]
        df = pd.DataFrame(
            {"price": [27000.0, 26900.0, 27100.0, 27050.0], "size": [1, 1, 1, 1]},
            index=pd.DatetimeIndex(times),
        )

        dc = preprocess_day(df, date)
        assert dc is not None
        # IBH should be 27100 (the 10:30:30 tick), not 27000 (the 9:30 tick)
        assert dc.ibh == 27100.0
        # IBL should be 26900 (the 10:29 tick)
        assert dc.ibl == 26900.0

    def test_ib_excludes_1031_ticks(self):
        """IB levels should NOT include 10:31:00 ticks."""
        from targeted_backtest import preprocess_day

        date = datetime.date(2026, 4, 1)
        times = [
            ET.localize(datetime.datetime(2026, 4, 1, 9, 30, 0)),
            ET.localize(datetime.datetime(2026, 4, 1, 10, 30, 59)),
            ET.localize(datetime.datetime(2026, 4, 1, 10, 31, 0)),   # post-IB
            ET.localize(datetime.datetime(2026, 4, 1, 10, 31, 1)),
        ]
        df = pd.DataFrame(
            {"price": [27000.0, 27050.0, 27200.0, 27200.0], "size": [1, 1, 1, 1]},
            index=pd.DatetimeIndex(times),
        )

        dc = preprocess_day(df, date)
        assert dc is not None
        # IBH should be 27050 (10:30:59), NOT 27200 (10:31:00)
        assert dc.ibh == 27050.0


# ---------------------------------------------------------------------------
# 4. Score entry
# ---------------------------------------------------------------------------


class TestScoreEntry:
    def test_zero_weights_give_zero_score(self):
        """Empty weights → score 0 for any entry."""
        from backtest.scoring import score_entry, EntryFactors

        fac = EntryFactors(
            level="IBL", direction="up", entry_count=1,
            et_mins=660, tick_rate=1500, session_move=10.0,
            range_30m=100.0, approach_speed=2.0, tick_density=10.0,
        )
        assert score_entry(fac, {}, 0, 0) == 0

    def test_level_weight_applied(self):
        """Level weight is added to score."""
        from backtest.scoring import score_entry, EntryFactors

        fac = EntryFactors(
            level="IBL", direction="up", entry_count=4,
            et_mins=660, tick_rate=1500, session_move=30.0,
            range_30m=50.0, approach_speed=1.0, tick_density=10.0,
        )
        w = {"lv_ibl": 3}
        assert score_entry(fac, w, 0, 0) == 3

    def test_streak_bonus(self):
        """Consecutive wins add streak bonus."""
        from backtest.scoring import score_entry, EntryFactors

        fac = EntryFactors(
            level="IBL", direction="up", entry_count=4,
            et_mins=660, tick_rate=1500, session_move=30.0,
            range_30m=50.0, approach_speed=1.0, tick_density=10.0,
        )
        w = {"sw": 3}
        assert score_entry(fac, w, cw=2, cl=0) == 3
        assert score_entry(fac, w, cw=1, cl=0) == 0  # need 2+ wins

    def test_multiple_factors_sum(self):
        """Score is sum of all matching factor weights."""
        from backtest.scoring import score_entry, EntryFactors

        fac = EntryFactors(
            level="IBL", direction="down", entry_count=2,
            et_mins=910,        # power hour
            tick_rate=1800,      # sweet spot
            session_move=-15.0,  # mildly red
            range_30m=50.0,
            approach_speed=1.0, tick_density=10.0,
        )
        w = {
            "lv_ibl": 1,
            "co_ibl_down": 1,
            "tp": 2,
            "ts": 2,
            "e2": 1,
            "mr": 2,
        }
        assert score_entry(fac, w, 0, 0) == 1 + 1 + 2 + 2 + 1 + 2  # 9

    def test_direction_combo_independent_of_level(self):
        """Wrong level+direction combo shouldn't get the weight."""
        from backtest.scoring import score_entry, EntryFactors

        fac = EntryFactors(
            level="IBL", direction="up", entry_count=4,
            et_mins=660, tick_rate=1500, session_move=30.0,
            range_30m=50.0, approach_speed=1.0, tick_density=10.0,
        )
        # IBL down combo weight exists, but entry is IBL up — shouldn't match
        w = {"co_ibl_down": 5}
        assert score_entry(fac, w, 0, 0) == 0


# ---------------------------------------------------------------------------
# 5. Fib level calculation
# ---------------------------------------------------------------------------


class TestFibLevels:
    def test_fib_extensions_correct(self):
        """FIB_EXT levels should be 0.272 * IB range beyond IBH/IBL."""
        from targeted_backtest import preprocess_day

        date = datetime.date(2026, 4, 1)
        times = [
            ET.localize(datetime.datetime(2026, 4, 1, 9, 30, 0)),
            ET.localize(datetime.datetime(2026, 4, 1, 10, 30, 0)),
            ET.localize(datetime.datetime(2026, 4, 1, 10, 31, 0)),
        ]
        # IBH=27100, IBL=26900, range=200
        df = pd.DataFrame(
            {"price": [26900.0, 27100.0, 27000.0], "size": [1, 1, 1]},
            index=pd.DatetimeIndex(times),
        )
        dc = preprocess_day(df, date)
        assert dc is not None
        assert dc.ibh == 27100.0
        assert dc.ibl == 26900.0
        # fib_hi = 27100 + 0.272 * 200 = 27154.4
        assert abs(dc.fib_hi - 27154.4) < 0.01
        # fib_lo = 26900 - 0.272 * 200 = 26845.6
        assert abs(dc.fib_lo - 26845.6) < 0.01


# ---------------------------------------------------------------------------
# 6. P&L math
# ---------------------------------------------------------------------------


class TestPnLMath:
    def test_win_pnl_exact(self):
        """Win P&L = (target_pts - fee) * multiplier."""
        from bot_risk_backtest import evaluate_bot_trade, FEE_PTS, MULTIPLIER

        # Price goes straight to target
        prices = np.array([100.0, 108.0], dtype=np.float64)
        ts = np.array([0, 1_000_000_000], dtype=np.int64)
        _, _, pnl_pts = evaluate_bot_trade(0, 100.0, "up", ts, prices, 8.0, 20.0, 900, None)
        pnl_usd = pnl_pts * MULTIPLIER
        expected = (8.0 - FEE_PTS) * MULTIPLIER
        assert abs(pnl_usd - expected) < 0.01

    def test_loss_pnl_exact(self):
        """Loss P&L = -(stop_pts + fee) * multiplier."""
        from bot_risk_backtest import evaluate_bot_trade, FEE_PTS, MULTIPLIER

        prices = np.array([100.0, 80.0], dtype=np.float64)
        ts = np.array([0, 1_000_000_000], dtype=np.int64)
        _, _, pnl_pts = evaluate_bot_trade(0, 100.0, "up", ts, prices, 8.0, 20.0, 900, None)
        pnl_usd = pnl_pts * MULTIPLIER
        expected = -(20.0 + FEE_PTS) * MULTIPLIER
        assert abs(pnl_usd - expected) < 0.01

    def test_target_hit_before_stop_is_win(self):
        """If both target and stop hit on same tick, earliest wins."""
        from bot_risk_backtest import evaluate_bot_trade

        # Target at 108, stop at 80. Price goes to 108 first.
        prices = np.array([100.0, 105.0, 108.0, 80.0], dtype=np.float64)
        ts = np.array([0, 1e9, 2e9, 3e9], dtype=np.int64)
        out, _, _ = evaluate_bot_trade(0, 100.0, "up", ts, prices, 8.0, 20.0, 900, None)
        assert out == "win"

    def test_stop_hit_before_target_is_loss(self):
        """If stop hits first, it's a loss even if target hits later."""
        from bot_risk_backtest import evaluate_bot_trade

        prices = np.array([100.0, 95.0, 80.0, 108.0], dtype=np.float64)
        ts = np.array([0, 1e9, 2e9, 3e9], dtype=np.int64)
        out, _, _ = evaluate_bot_trade(0, 100.0, "up", ts, prices, 8.0, 20.0, 900, None)
        assert out == "loss"


# ---------------------------------------------------------------------------
# 7. Max per level
# ---------------------------------------------------------------------------


class TestMaxPerLevel:
    def test_max_per_level_caps_entries(self):
        """After max_per_level entries on a level, no more trades on it."""
        from backtest.simulate import simulate_day
        from backtest.zones import BotZoneTradeReset

        date = datetime.date(2026, 4, 1)
        # 500 ticks at IBH price — many zone entries possible
        prices = []
        for _ in range(100):
            prices.extend([27050.0, 27060.0])  # toggle near/away
        dc, arrays = _make_tick_data(date, prices, tick_interval_ms=5000)
        dc.ibh = 27050.0

        def zone_factory(name, price, drifts):
            return BotZoneTradeReset(price, drifts)

        trades, _ = simulate_day(
            dc, arrays, zone_factory,
            lambda lv: 8.0, 20.0, max_per_level=3,
            weights=None, min_score=-99,
        )
        ibh_trades = [t for t in trades if t.level == "IBH"]
        assert len(ibh_trades) <= 3


# ---------------------------------------------------------------------------
# 8. Score filtering
# ---------------------------------------------------------------------------


class TestScoreFiltering:
    def test_min_score_filters_low_entries(self):
        """Entries below min_score are rejected."""
        from backtest.simulate import simulate_day
        from backtest.zones import BotZoneTradeReset

        date = datetime.date(2026, 4, 1)
        dc, arrays = _make_tick_data(date, [27050.0] * 100)
        dc.ibh = 27050.0

        def zone_factory(name, price, drifts):
            return BotZoneTradeReset(price, drifts)

        # All weights zero → score = 0. min_score=1 should reject all.
        trades_filtered, _ = simulate_day(
            dc, arrays, zone_factory,
            lambda lv: 8.0, 20.0, 12,
            weights={}, min_score=1,
        )
        ibh_filtered = [t for t in trades_filtered if t.level == "IBH"]

        # Same but min_score=-99 should accept all.
        trades_all, _ = simulate_day(
            dc, arrays, zone_factory,
            lambda lv: 8.0, 20.0, 12,
            weights={}, min_score=-99,
        )
        ibh_all = [t for t in trades_all if t.level == "IBH"]

        assert len(ibh_filtered) == 0
        assert len(ibh_all) > 0


# ---------------------------------------------------------------------------
# 9. Streak tracking across trades
# ---------------------------------------------------------------------------


class TestStreakTracking:
    def test_streak_carries_across_trades(self):
        """Consecutive wins/losses should update streak state."""
        from backtest.simulate import simulate_day
        from backtest.zones import BotZoneTradeReset

        date = datetime.date(2026, 4, 1)
        # Price at IBH, then moves to target (win)
        prices = [27050.0, 27042.0]  # sell at IBH, target at 27042
        dc, arrays = _make_tick_data(date, prices * 50, tick_interval_ms=2000)
        dc.ibh = 27050.0

        def zone_factory(name, price, drifts):
            return BotZoneTradeReset(price, drifts)

        _, (cw, cl) = simulate_day(
            dc, arrays, zone_factory,
            lambda lv: 8.0, 20.0, 12,
            weights=None, min_score=-99,
            streak_state=(0, 0),
        )
        # Should have some wins, so cw > 0
        assert cw >= 0 and cl >= 0
        assert cw + cl > 0  # at least some trades happened

    def test_streak_state_input_is_used(self):
        """Starting with cw=5 should give streak bonus on first trade."""
        from backtest.scoring import score_entry, EntryFactors

        fac = EntryFactors(
            level="IBL", direction="up", entry_count=4,
            et_mins=660, tick_rate=1500, session_move=30.0,
            range_30m=50.0, approach_speed=1.0, tick_density=10.0,
        )
        w = {"sw": 3, "sl": -2}
        # Starting with 5 consecutive wins
        assert score_entry(fac, w, cw=5, cl=0) == 3
        # Starting with 3 consecutive losses
        assert score_entry(fac, w, cw=0, cl=3) == -2


# ---------------------------------------------------------------------------
# 10. Train weights sanity
# ---------------------------------------------------------------------------


class TestTrainWeights:
    def test_returns_dict(self):
        """train_weights returns a dict of integer weights."""
        from backtest.scoring import train_weights, EntryFactors

        entries = []
        for i in range(100):
            fac = EntryFactors(
                level="IBL", direction="up", entry_count=2,
                et_mins=660, tick_rate=1500, session_move=10.0,
                range_30m=100.0, approach_speed=2.0, tick_density=10.0,
            )
            outcome = "win" if i % 2 == 0 else "loss"
            entries.append((fac, outcome, 0, 0))

        w = train_weights(entries)
        assert isinstance(w, dict)
        assert all(isinstance(v, int) for v in w.values())

    def test_empty_input_returns_empty(self):
        from backtest.scoring import train_weights
        assert train_weights([]) == {}

    def test_weights_bounded(self):
        """All weights should be between -4 and +4."""
        from backtest.scoring import train_weights, EntryFactors

        entries = []
        for i in range(200):
            fac = EntryFactors(
                level=["IBL", "IBH", "VWAP"][i % 3],
                direction=["up", "down"][i % 2],
                entry_count=(i % 5) + 1,
                et_mins=660 + (i % 300),
                tick_rate=500 + i * 10,
                session_move=-50 + i * 0.5,
                range_30m=50 + i * 0.5,
                approach_speed=i * 0.05,
                tick_density=i * 0.1,
            )
            outcome = "win" if i % 3 != 0 else "loss"
            entries.append((fac, outcome, 0, 0))

        w = train_weights(entries)
        for k, v in w.items():
            assert -4 <= v <= 4, f"Weight {k}={v} out of bounds"


# ---------------------------------------------------------------------------
# 11. Post-IB data integrity
# ---------------------------------------------------------------------------


class TestPostIBData:
    def test_post_ib_start_idx_correct(self):
        """post_ib_start_idx should point to the first post-IB tick."""
        from targeted_backtest import preprocess_day

        date = datetime.date(2026, 4, 1)
        times = [
            ET.localize(datetime.datetime(2026, 4, 1, 9, 30, 0)),
            ET.localize(datetime.datetime(2026, 4, 1, 10, 0, 0)),
            ET.localize(datetime.datetime(2026, 4, 1, 10, 30, 0)),
            ET.localize(datetime.datetime(2026, 4, 1, 10, 31, 0)),  # first post-IB
            ET.localize(datetime.datetime(2026, 4, 1, 11, 0, 0)),
        ]
        df = pd.DataFrame(
            {"price": [27000.0, 27010.0, 27020.0, 27030.0, 27040.0],
             "size": [1, 1, 1, 1, 1]},
            index=pd.DatetimeIndex(times),
        )
        dc = preprocess_day(df, date)
        assert dc is not None
        assert dc.post_ib_start_idx == 3  # index of 10:31 tick
        assert len(dc.post_ib_prices) == 2  # 10:31 and 11:00

    def test_post_ib_prices_match_full_prices(self):
        """post_ib_prices should be a slice of full_prices."""
        from targeted_backtest import preprocess_day

        date = datetime.date(2026, 4, 1)
        times = [
            ET.localize(datetime.datetime(2026, 4, 1, 9, 30, 0)),
            ET.localize(datetime.datetime(2026, 4, 1, 10, 31, 0)),
            ET.localize(datetime.datetime(2026, 4, 1, 12, 0, 0)),
        ]
        df = pd.DataFrame(
            {"price": [26900.0, 27000.0, 27100.0], "size": [1, 1, 1]},
            index=pd.DatetimeIndex(times),
        )
        dc = preprocess_day(df, date)
        assert dc is not None
        s = dc.post_ib_start_idx
        np.testing.assert_array_equal(
            dc.post_ib_prices,
            dc.full_prices[s : s + len(dc.post_ib_prices)],
        )

    def test_vwap_is_cumulative(self):
        """VWAP should be cumulative price*size / cumulative size."""
        from targeted_backtest import preprocess_day

        date = datetime.date(2026, 4, 1)
        times = [
            ET.localize(datetime.datetime(2026, 4, 1, 9, 30, 0)),
            ET.localize(datetime.datetime(2026, 4, 1, 10, 31, 0)),
        ]
        # Price 100 with size 3, then price 200 with size 1
        # VWAP at tick 2 = (100*3 + 200*1) / (3+1) = 500/4 = 125
        df = pd.DataFrame(
            {"price": [100.0, 200.0], "size": [3, 1]},
            index=pd.DatetimeIndex(times),
        )
        dc = preprocess_day(df, date)
        assert dc is not None
        assert abs(dc.post_ib_vwaps[0] - 125.0) < 0.01
