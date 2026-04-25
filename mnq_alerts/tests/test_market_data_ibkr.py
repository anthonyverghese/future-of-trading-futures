"""Tests for market_data_ibkr module — session accumulator and interface parity."""

import sys
import os
import datetime

import pandas as pd
import pytz

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import market_data_ibkr as md
from config import IBKR_CLIENT_ID

ET = pytz.timezone("America/New_York")


# ---------------------------------------------------------------------------
# 1. Reset session
# ---------------------------------------------------------------------------


class TestResetSession:
    def test_clears_all_state(self):
        md._prices = [100.0, 200.0]
        md._sizes = [1, 2]
        md._timestamps = [pd.Timestamp("2026-04-24 10:00:00", tz=ET)]
        md._trades_cache = pd.DataFrame({"Price": [100.0]})
        md._current_price = 200.0
        md._reconnect_count = 5

        md.reset_session()

        assert md._prices == []
        assert md._sizes == []
        assert md._timestamps == []
        assert md._trades_cache is None
        assert md._current_price is None
        assert md._reconnect_count == 0

    def test_get_session_trades_empty_after_reset(self):
        md.reset_session()
        df = md.get_session_trades()
        assert df.empty
        assert list(df.columns) == ["Price", "Size"]

    def test_get_current_price_none_after_reset(self):
        md.reset_session()
        assert md.get_current_price() is None


# ---------------------------------------------------------------------------
# 2. Load session cache
# ---------------------------------------------------------------------------


class TestLoadSessionCache:
    def setup_method(self):
        md.reset_session()

    def test_loads_trades_from_dataframe(self):
        ts = pd.DatetimeIndex([
            pd.Timestamp("2026-04-24 10:00:00", tz=ET),
            pd.Timestamp("2026-04-24 10:00:01", tz=ET),
        ])
        trades = pd.DataFrame({"Price": [100.0, 101.0], "Size": [5, 10]}, index=ts)

        md.load_session_cache(trades)

        assert len(md._prices) == 2
        assert md._prices == [100.0, 101.0]
        assert md._sizes == [5, 10]
        assert md._current_price == 101.0

    def test_empty_dataframe_is_noop(self):
        md.load_session_cache(pd.DataFrame())
        assert md._prices == []
        assert md._current_price is None

    def test_invalidates_cache(self):
        md._trades_cache = pd.DataFrame({"Price": [999.0]})
        ts = pd.DatetimeIndex([pd.Timestamp("2026-04-24 10:00:00", tz=ET)])
        md.load_session_cache(pd.DataFrame({"Price": [100.0], "Size": [1]}, index=ts))
        assert md._trades_cache is None

    def test_wrong_columns_skipped(self, capsys):
        """DataFrame with wrong column names is rejected with a warning."""
        ts = pd.DatetimeIndex([pd.Timestamp("2026-04-24 10:00:00", tz=ET)])
        bad_df = pd.DataFrame({"price": [100.0], "size": [1]}, index=ts)
        md.load_session_cache(bad_df)
        assert md._prices == []  # not loaded
        captured = capsys.readouterr()
        assert "unexpected columns" in captured.out


# ---------------------------------------------------------------------------
# 3. Get session trades
# ---------------------------------------------------------------------------


class TestGetSessionTrades:
    def setup_method(self):
        md.reset_session()

    def test_returns_accumulated_trades(self):
        ts1 = pd.Timestamp("2026-04-24 10:00:00", tz=ET)
        ts2 = pd.Timestamp("2026-04-24 10:00:01", tz=ET)
        md._prices = [100.0, 101.0]
        md._sizes = [5, 10]
        md._timestamps = [ts1, ts2]
        md._trades_cache = None

        df = md.get_session_trades()
        assert len(df) == 2
        assert list(df.columns) == ["Price", "Size"]
        assert df["Price"].iloc[0] == 100.0
        assert df["Size"].iloc[1] == 10

    def test_caches_result(self):
        md._prices = [100.0]
        md._sizes = [1]
        md._timestamps = [pd.Timestamp("2026-04-24 10:00:00", tz=ET)]
        md._trades_cache = None

        df1 = md.get_session_trades()
        df2 = md.get_session_trades()
        assert df1 is df2  # same object, cached

    def test_cache_invalidated_on_new_trade(self):
        md._prices = [100.0]
        md._sizes = [1]
        md._timestamps = [pd.Timestamp("2026-04-24 10:00:00", tz=ET)]
        md._trades_cache = None

        df1 = md.get_session_trades()
        assert md._trades_cache is not None

        md._prices.append(101.0)
        md._sizes.append(2)
        md._timestamps.append(pd.Timestamp("2026-04-24 10:00:01", tz=ET))
        md._trades_cache = None  # trade_stream does this

        df2 = md.get_session_trades()
        assert len(df2) == 2
        assert df1 is not df2


# ---------------------------------------------------------------------------
# 4. Interface parity with Databento module
# ---------------------------------------------------------------------------


class TestInterfaceParity:
    def test_exports_same_functions(self):
        import market_data as md_databento
        import market_data_ibkr as md_ibkr

        expected = [
            "trade_stream",
            "reset_session",
            "load_session_cache",
            "get_session_trades",
            "get_current_price",
        ]
        for fn_name in expected:
            assert hasattr(md_databento, fn_name), f"Databento missing {fn_name}"
            assert hasattr(md_ibkr, fn_name), f"IBKR missing {fn_name}"
            assert callable(getattr(md_ibkr, fn_name))

    def test_empty_trades_same_schema(self):
        import market_data as md_databento

        md.reset_session()
        md_databento.reset_session()

        df_ibkr = md.get_session_trades()
        df_db = md_databento.get_session_trades()

        assert list(df_ibkr.columns) == list(df_db.columns)
        assert df_ibkr.index.dtype == df_db.index.dtype


# ---------------------------------------------------------------------------
# 5. trade_stream signature
# ---------------------------------------------------------------------------


class TestTradeStreamSignature:
    def test_accepts_session_start_none(self):
        gen = md.trade_stream(session_start=None)
        assert gen is not None

    def test_accepts_session_start_datetime(self):
        start = datetime.datetime(2026, 4, 24, 9, 30, tzinfo=ET)
        gen = md.trade_stream(session_start=start)
        assert gen is not None


# ---------------------------------------------------------------------------
# 6. Tick validation
# ---------------------------------------------------------------------------


class TestTickValidation:
    """Verify that invalid ticks are handled properly by the accumulator.

    We can't test trade_stream directly (needs IBKR connection), but we
    can verify that the accumulator logic handles edge cases.
    """

    def setup_method(self):
        md.reset_session()

    def test_accumulator_handles_normal_data(self):
        """Simulate what trade_stream does for each valid tick."""
        ts = pd.Timestamp("2026-04-24 10:00:00", tz=ET)
        md._prices.append(27000.0)
        md._sizes.append(5)
        md._timestamps.append(ts)
        md._trades_cache = None
        md._current_price = 27000.0

        assert md.get_current_price() == 27000.0
        df = md.get_session_trades()
        assert len(df) == 1
        assert df["Price"].iloc[0] == 27000.0

    def test_multiple_ticks_accumulated(self):
        base = pd.Timestamp("2026-04-24 10:00:00", tz=ET)
        for i in range(100):
            md._prices.append(27000.0 + i * 0.25)
            md._sizes.append(1)
            md._timestamps.append(base + pd.Timedelta(seconds=i))
        md._trades_cache = None
        md._current_price = md._prices[-1]

        df = md.get_session_trades()
        assert len(df) == 100
        assert md.get_current_price() == 27000.0 + 99 * 0.25


# ---------------------------------------------------------------------------
# 7. Config
# ---------------------------------------------------------------------------


class TestCacheRoundTrip:
    """Verify timestamps survive the save_trades → load_trades round-trip.

    cache.save_trades uses df.index.asi8 (nanoseconds) and
    cache.load_trades uses pd.to_datetime(..., unit='ns'). The IBKR
    timestamps must be stored in nanosecond resolution for this to work.
    """

    def setup_method(self):
        md.reset_session()

    def test_timestamp_nanosecond_resolution(self):
        """Accumulated timestamps must have ns resolution for cache compatibility."""
        tick_time = datetime.datetime(2026, 4, 24, 14, 0, 0, 123456, tzinfo=pytz.utc)
        ts_et = tick_time.astimezone(ET)
        ts_pd = pd.Timestamp(ts_et).as_unit("ns")

        md._prices.append(27000.0)
        md._sizes.append(5)
        md._timestamps.append(ts_pd)
        md._trades_cache = None

        df = md.get_session_trades()
        # DatetimeIndex must be ns resolution
        assert df.index.dtype == pd.DatetimeIndex([], dtype="datetime64[ns, America/New_York]").dtype

    def test_asi8_returns_nanoseconds(self):
        """asi8 must return nanoseconds, not microseconds."""
        tick_time = datetime.datetime(2026, 4, 24, 14, 0, 0, 123456, tzinfo=pytz.utc)
        ts_et = tick_time.astimezone(ET)
        ts_pd = pd.Timestamp(ts_et).as_unit("ns")

        md._prices.append(27000.0)
        md._sizes.append(5)
        md._timestamps.append(ts_pd)
        md._trades_cache = None

        df = md.get_session_trades()
        asi8_val = df.index.asi8[0]
        expected_ns = ts_pd.value
        assert asi8_val == expected_ns, f"asi8={asi8_val} != value={expected_ns}"

    def test_full_round_trip(self):
        """Simulate save_trades → load_trades and verify timestamps match."""
        tick_time = datetime.datetime(2026, 4, 24, 14, 0, 0, 123456, tzinfo=pytz.utc)
        ts_et = tick_time.astimezone(ET)
        ts_pd = pd.Timestamp(ts_et).as_unit("ns")

        md._prices.append(27000.0)
        md._sizes.append(5)
        md._timestamps.append(ts_pd)
        md._trades_cache = None

        df = md.get_session_trades()

        # Simulate save_trades: index → asi8
        ns_val = df.index.asi8[0]

        # Simulate load_trades: ns → pd.to_datetime
        loaded = pd.to_datetime(ns_val, unit="ns", utc=True).tz_convert(ET)

        assert loaded == ts_pd, f"Round-trip failed: {loaded} != {ts_pd}"


class TestConfig:
    def test_feed_client_id_differs_from_broker(self):
        """Feed clientId must be different from broker clientId."""
        assert md._FEED_CLIENT_ID != IBKR_CLIENT_ID
        assert md._FEED_CLIENT_ID == IBKR_CLIENT_ID + 1
