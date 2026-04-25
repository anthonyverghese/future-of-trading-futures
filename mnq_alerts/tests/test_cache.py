"""Tests for cache.py — SQLite persistence for trades and alerts."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime

import pandas as pd
import pytest
import pytz

import cache

ET = pytz.timezone("America/New_York")


@pytest.fixture(autouse=True)
def _isolate_db_paths(tmp_path, monkeypatch):
    """Point both DB paths to the tmp directory so tests never touch production."""
    monkeypatch.setattr(cache, "CACHE_PATH", str(tmp_path / ".session_cache.db"))
    monkeypatch.setattr(cache, "ALERTS_LOG_PATH", str(tmp_path / "alerts_log.db"))


# ── Trade cache round-trip ───────────────────────────────────────────────────


def _make_trades(n: int = 5) -> pd.DataFrame:
    """Build a small trades DataFrame with a tz-aware DatetimeIndex in ET."""
    base = datetime.datetime.now(ET).replace(hour=10, minute=0, second=0, microsecond=0)
    timestamps = [base + datetime.timedelta(seconds=i) for i in range(n)]
    return pd.DataFrame(
        {"Price": [20000.0 + i for i in range(n)], "Size": [1] * n},
        index=pd.DatetimeIndex(timestamps),
    )


class TestSaveLoadTrades:
    def test_round_trip(self):
        trades = _make_trades()
        cache.save_trades(trades)
        loaded = cache.load_trades()
        assert len(loaded) == len(trades)
        assert list(loaded["Price"]) == list(trades["Price"])
        assert list(loaded["Size"]) == list(trades["Size"])

    def test_load_empty_when_no_cache(self):
        loaded = cache.load_trades()
        assert loaded.empty

    def test_save_empty_is_noop(self):
        cache.save_trades(
            pd.DataFrame({"Price": [], "Size": []}, index=pd.DatetimeIndex([]))
        )
        loaded = cache.load_trades()
        assert loaded.empty


# ── Daily parquet export ────────────────────────────────────────────────────


class TestExportDailyParquet:
    """Tests use monkeypatch on os.path.dirname inside cache module to
    redirect parquet output to tmp_path instead of the real data_cache/."""

    @pytest.fixture(autouse=True)
    def _redirect_data_cache(self, tmp_path, monkeypatch):
        """Redirect export_daily_parquet output to tmp_path/data_cache/."""
        self._data_cache = str(tmp_path / "data_cache")
        # cache.py uses os.path.dirname(__file__) to find data_cache.
        # Monkeypatch __file__ so dirname points to tmp_path.
        monkeypatch.setattr(cache, "__file__", str(tmp_path / "cache.py"))

    def _exported_path(self):
        today = datetime.datetime.now(ET).date().isoformat()
        return os.path.join(self._data_cache, f"MNQ_{today}.parquet")

    def test_exports_correct_format(self):
        """Exported parquet matches Databento format."""
        trades = _make_trades(10)
        path = cache.export_daily_parquet(trades)

        assert path is not None
        assert os.path.exists(path)

        df = pd.read_parquet(path)
        assert list(df.columns) == ["price", "size"]
        assert df.index.name == "ts"
        assert "America/New_York" in str(df.index.tz)
        assert len(df) == 10
        assert df["price"].dtype == float
        assert df["size"].dtype == int

    def test_empty_trades_returns_none(self):
        empty = pd.DataFrame({"Price": [], "Size": []}, index=pd.DatetimeIndex([]))
        result = cache.export_daily_parquet(empty)
        assert result is None

    def test_wrong_columns_returns_none(self, capsys):
        bad = pd.DataFrame({"price": [100.0], "size": [1]},
                           index=pd.DatetimeIndex([datetime.datetime.now(ET)]))
        result = cache.export_daily_parquet(bad)
        assert result is None
        captured = capsys.readouterr()
        assert "unexpected columns" in captured.out

    def test_skips_overwrite_if_existing_larger(self, capsys):
        """Don't overwrite existing parquet if it has more trades."""
        # First export: 20 trades
        large = _make_trades(20)
        path1 = cache.export_daily_parquet(large)
        assert path1 is not None
        df1 = pd.read_parquet(path1)
        assert len(df1) == 20

        # Second export: 10 trades — should skip
        small = _make_trades(10)
        path2 = cache.export_daily_parquet(small)

        # File should still have 20 trades
        df2 = pd.read_parquet(self._exported_path())
        assert len(df2) == 20
        captured = capsys.readouterr()
        assert "skipping overwrite" in captured.out

    def test_overwrites_if_new_has_more_trades(self):
        """Overwrite existing parquet if new export has more trades."""
        # First export: 10 trades
        small = _make_trades(10)
        cache.export_daily_parquet(small)

        # Second export: 20 trades — should overwrite
        large = _make_trades(20)
        cache.export_daily_parquet(large)

        df = pd.read_parquet(self._exported_path())
        assert len(df) == 20

    def test_overwrites_corrupted_file(self):
        """Overwrite if existing parquet is corrupted."""
        # Write garbage to the path
        path = self._exported_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("not a parquet file")

        trades = _make_trades(5)
        result = cache.export_daily_parquet(trades)
        assert result is not None

        df = pd.read_parquet(result)
        assert len(df) == 5

    def test_backtest_load_day_compatible(self):
        """Exported parquet can be loaded by targeted_backtest.load_day."""
        trades = _make_trades(10)
        path = cache.export_daily_parquet(trades)
        assert path is not None

        df = pd.read_parquet(path)
        # Verify it has the same structure load_day expects
        assert "price" in df.columns
        assert "size" in df.columns
        assert df.index.tz is not None


# ── Alert log ────────────────────────────────────────────────────────────────


class TestLogAlert:
    def test_returns_id(self):
        alert_id = cache.log_alert("MNQ", "IBH", 20000.0, 19995.0, "up")
        assert isinstance(alert_id, int)
        assert alert_id >= 1

    def test_creates_record(self):
        alert_id = cache.log_alert("MNQ", "IBL", 19800.0, 19805.0, "down")
        # Verify the record exists in the DB
        import sqlite3

        with sqlite3.connect(cache.ALERTS_LOG_PATH) as conn:
            row = conn.execute(
                "SELECT * FROM alerts WHERE id = ?", (alert_id,)
            ).fetchone()
        assert row is not None

    def test_increments_daily_stats(self):
        cache.log_alert("MNQ", "IBH", 20000.0, 19995.0, "up")
        cache.log_alert("MNQ", "IBL", 19800.0, 19805.0, "down")
        import sqlite3

        today = datetime.datetime.now().strftime("%Y-%m-%d")
        with sqlite3.connect(cache.ALERTS_LOG_PATH) as conn:
            row = conn.execute(
                "SELECT notifications_sent FROM daily_stats WHERE date = ?", (today,)
            ).fetchone()
        assert row is not None
        assert row[0] == 2

    def test_with_trade_ts(self):
        ts = datetime.datetime(2026, 3, 25, 10, 15, 0, tzinfo=ET)
        alert_id = cache.log_alert("MNQ", "IBH", 20000.0, 19995.0, "up", trade_ts=ts)
        import sqlite3

        with sqlite3.connect(cache.ALERTS_LOG_PATH) as conn:
            row = conn.execute(
                "SELECT date, time FROM alerts WHERE id = ?", (alert_id,)
            ).fetchone()
        assert row[0] == "2026-03-25"


class TestUpdateAlertHit:
    def test_updates_hit_time(self):
        alert_id = cache.log_alert("MNQ", "IBH", 20000.0, 19995.0, "up")
        hit_time = "2026-03-25T10:20:00"
        cache.update_alert_hit(alert_id, hit_time)

        import sqlite3

        with sqlite3.connect(cache.ALERTS_LOG_PATH) as conn:
            row = conn.execute(
                "SELECT hit_time FROM alerts WHERE id = ?", (alert_id,)
            ).fetchone()
        assert row[0] == hit_time


class TestUpdateAlertOutcome:
    def test_correct_outcome(self):
        alert_id = cache.log_alert("MNQ", "IBH", 20000.0, 19995.0, "up")
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        cache.update_alert_outcome(alert_id, "correct", date_str)

        import sqlite3

        with sqlite3.connect(cache.ALERTS_LOG_PATH) as conn:
            row = conn.execute(
                "SELECT outcome FROM alerts WHERE id = ?", (alert_id,)
            ).fetchone()
        assert row[0] == "correct"

    def test_incorrect_increments_daily_stats(self):
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        alert_id = cache.log_alert("MNQ", "IBL", 19800.0, 19805.0, "down")
        cache.update_alert_outcome(alert_id, "incorrect", date_str)

        import sqlite3

        with sqlite3.connect(cache.ALERTS_LOG_PATH) as conn:
            row = conn.execute(
                "SELECT incorrect_recs FROM daily_stats WHERE date = ?", (date_str,)
            ).fetchone()
        assert row[0] == 1

    def test_unresolved_no_stat_change(self):
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        alert_id = cache.log_alert("MNQ", "IBH", 20000.0, 19995.0, "up")
        cache.update_alert_outcome(alert_id, "unresolved", date_str)

        import sqlite3

        with sqlite3.connect(cache.ALERTS_LOG_PATH) as conn:
            row = conn.execute(
                "SELECT correct_recs, incorrect_recs FROM daily_stats WHERE date = ?",
                (date_str,),
            ).fetchone()
        # 'unresolved' should not increment correct or incorrect
        assert row[0] == 0
        assert row[1] == 0


class TestGetDailySummary:
    def test_empty_db(self):
        summary = cache.get_daily_summary("2026-03-25")
        assert summary == {"correct": 0, "incorrect": 0, "inconclusive": 0}

    def test_counts(self):
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        a1 = cache.log_alert("MNQ", "IBH", 20000.0, 19995.0, "up")
        a2 = cache.log_alert("MNQ", "IBL", 19800.0, 19805.0, "down")
        a3 = cache.log_alert("MNQ", "VWAP", 19900.0, 19895.0, "up")
        cache.update_alert_outcome(a1, "correct", date_str)
        cache.update_alert_outcome(a2, "incorrect", date_str)
        cache.update_alert_outcome(a3, "correct", date_str)

        summary = cache.get_daily_summary(date_str)
        assert summary["correct"] == 2
        assert summary["incorrect"] == 1


class TestLoadRecentOutcomes:
    def test_empty_db(self):
        assert cache.load_recent_outcomes() == []

    def test_chronological_order(self):
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        a1 = cache.log_alert("MNQ", "IBH", 20000.0, 19995.0, "up")
        a2 = cache.log_alert("MNQ", "IBL", 19800.0, 19805.0, "down")
        a3 = cache.log_alert("MNQ", "VWAP", 19900.0, 19895.0, "up")
        cache.update_alert_outcome(a1, "correct", date_str)
        cache.update_alert_outcome(a2, "incorrect", date_str)
        cache.update_alert_outcome(a3, "correct", date_str)

        outcomes = cache.load_recent_outcomes(limit=10)
        assert outcomes == ["correct", "incorrect", "correct"]

    def test_limit(self):
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        for i in range(5):
            aid = cache.log_alert("MNQ", "IBH", 20000.0, 19995.0, "up")
            cache.update_alert_outcome(aid, "correct", date_str)
        assert len(cache.load_recent_outcomes(limit=3)) == 3


class TestLoadPendingAlerts:
    def test_empty_db(self):
        assert cache.load_pending_alerts("2026-03-25") == []

    def test_returns_unresolved(self):
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        a1 = cache.log_alert("MNQ", "IBH", 20000.0, 19995.0, "up")
        a2 = cache.log_alert("MNQ", "IBL", 19800.0, 19805.0, "down")
        cache.update_alert_outcome(a1, "correct", date_str)
        # a2 has no outcome — it should be pending

        pending = cache.load_pending_alerts(date_str)
        assert len(pending) == 1
        assert pending[0]["alert_id"] == a2
        assert pending[0]["direction"] == "down"
        assert pending[0]["line_price"] == 19800.0

    def test_includes_hit_time(self):
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        aid = cache.log_alert("MNQ", "IBH", 20000.0, 19995.0, "up")
        cache.update_alert_hit(aid, "2026-03-25T10:20:00")

        pending = cache.load_pending_alerts(date_str)
        assert len(pending) == 1
        assert pending[0]["hit_time"] is not None


# ── Bot trade helpers ───────────────────────────────────────────────────────

DATE = "2026-04-10"


def _insert_closed_trade(level="IBH", direction="long", pnl=24.0, outcome="win"):
    """Insert a closed bot trade and return its row id."""
    tid = cache.log_bot_trade_entry(
        DATE, "10:00:00", level, direction, 20000.0,
        20001.0, 20013.0, 19976.0,
    )
    cache.update_bot_trade_exit(tid, "10:05:00", 20013.0, pnl, outcome, "target")
    return tid


def _insert_open_trade(level="IBH", direction="long", parent_order_id=None):
    """Insert an open (no exit) bot trade and return its row id."""
    return cache.log_bot_trade_entry(
        DATE, "10:10:00", level, direction, 20000.0,
        20001.0, 20013.0, 19976.0,
        parent_order_id=parent_order_id,
    )


# ── TestBotDailyRiskState ──────────────────────────────────────────────────


class TestBotDailyRiskState:
    def test_empty_when_no_trades(self):
        state = cache.load_bot_daily_risk_state(DATE)
        assert state["pnl_usd"] == 0.0
        assert state["trades"] == 0
        assert state["wins"] == 0
        assert state["losses"] == 0
        assert state["consecutive_losses"] == 0

    def test_counts_closed_trades_only(self):
        _insert_closed_trade(outcome="win", pnl=24.0)
        _insert_closed_trade(outcome="loss", pnl=-50.0)
        _insert_open_trade()

        state = cache.load_bot_daily_risk_state(DATE)
        assert state["trades"] == 2
        assert state["wins"] == 1
        assert state["losses"] == 1

    def test_consecutive_losses_from_tail(self):
        _insert_closed_trade(outcome="win", pnl=24.0)
        _insert_closed_trade(outcome="loss", pnl=-50.0)
        _insert_closed_trade(outcome="loss", pnl=-50.0)

        state = cache.load_bot_daily_risk_state(DATE)
        assert state["consecutive_losses"] == 2

    def test_consecutive_losses_reset_by_win(self):
        _insert_closed_trade(outcome="loss", pnl=-50.0)
        _insert_closed_trade(outcome="loss", pnl=-50.0)
        _insert_closed_trade(outcome="win", pnl=24.0)

        state = cache.load_bot_daily_risk_state(DATE)
        assert state["consecutive_losses"] == 0

    def test_pnl_sum(self):
        _insert_closed_trade(outcome="win", pnl=24.0)
        _insert_closed_trade(outcome="loss", pnl=-50.0)
        _insert_closed_trade(outcome="win", pnl=24.0)

        state = cache.load_bot_daily_risk_state(DATE)
        assert state["pnl_usd"] == pytest.approx(-2.0)


# ── TestBotDailyLevelCounts ────────────────────────────────────────────────


class TestBotDailyLevelCounts:
    def test_empty_when_no_trades(self):
        counts = cache.load_bot_daily_level_counts(DATE)
        assert counts == {}

    def test_counts_by_level(self):
        for _ in range(3):
            _insert_closed_trade(level="IBH")
        for _ in range(2):
            _insert_closed_trade(level="FIB")

        counts = cache.load_bot_daily_level_counts(DATE)
        assert counts["IBH"] == 3
        assert counts["FIB"] == 2

    def test_excludes_open_trades(self):
        _insert_closed_trade(level="IBH")
        _insert_open_trade(level="IBH")

        counts = cache.load_bot_daily_level_counts(DATE)
        assert counts.get("IBH") == 1


# ── TestMarkOrphaned ───────────────────────────────────────────────────────


class TestMarkOrphaned:
    def test_marks_open_as_orphaned(self):
        open_id = _insert_open_trade()
        closed_id = _insert_closed_trade()

        cache.mark_open_bot_trades_orphaned(DATE)

        import sqlite3

        with sqlite3.connect(cache.ALERTS_LOG_PATH) as conn:
            open_row = conn.execute(
                "SELECT outcome FROM bot_trades WHERE id = ?", (open_id,)
            ).fetchone()
            closed_row = conn.execute(
                "SELECT outcome FROM bot_trades WHERE id = ?", (closed_id,)
            ).fetchone()

        assert open_row[0] == "orphaned"
        assert closed_row[0] == "win"

    def test_returns_count(self):
        _insert_open_trade()
        _insert_open_trade()

        count = cache.mark_open_bot_trades_orphaned(DATE)
        assert count == 2

    def test_noop_when_none_open(self):
        _insert_closed_trade()

        count = cache.mark_open_bot_trades_orphaned(DATE)
        assert count == 0


# ── TestBotOpenTradeLookup ─────────────────────────────────────────────────


class TestBotOpenTradeLookup:
    def test_finds_matching_row(self):
        _insert_open_trade(parent_order_id=42)

        row = cache.load_bot_open_trade_by_parent_order_id(42, DATE)
        assert row is not None
        assert row["level_name"] == "IBH"
        assert row["direction"] == "long"

    def test_returns_none_for_wrong_id(self):
        _insert_open_trade(parent_order_id=42)

        row = cache.load_bot_open_trade_by_parent_order_id(999, DATE)
        assert row is None

    def test_returns_none_for_closed(self):
        tid = cache.log_bot_trade_entry(
            DATE, "10:00:00", "IBH", "long", 20000.0,
            20001.0, 20013.0, 19976.0,
            parent_order_id=42,
        )
        cache.update_bot_trade_exit(tid, "10:05:00", 20013.0, 24.0, "win", "target")

        row = cache.load_bot_open_trade_by_parent_order_id(42, DATE)
        assert row is None

    def test_returns_none_for_wrong_date(self):
        _insert_open_trade(parent_order_id=42)

        row = cache.load_bot_open_trade_by_parent_order_id(42, "2026-01-01")
        assert row is None


# ── TestParentOrderIdColumn ────────────────────────────────────────────────


class TestParentOrderIdColumn:
    def test_stores_parent_order_id(self):
        tid = cache.log_bot_trade_entry(
            DATE, "10:00:00", "IBH", "long", 20000.0,
            20001.0, 20013.0, 19976.0,
            parent_order_id=99,
        )
        import sqlite3

        with sqlite3.connect(cache.ALERTS_LOG_PATH) as conn:
            row = conn.execute(
                "SELECT parent_order_id FROM bot_trades WHERE id = ?", (tid,)
            ).fetchone()
        assert row[0] == 99

    def test_parent_order_id_default_none(self):
        tid = cache.log_bot_trade_entry(
            DATE, "10:00:00", "IBH", "long", 20000.0,
            20001.0, 20013.0, 19976.0,
        )
        import sqlite3

        with sqlite3.connect(cache.ALERTS_LOG_PATH) as conn:
            row = conn.execute(
                "SELECT parent_order_id FROM bot_trades WHERE id = ?", (tid,)
            ).fetchone()
        assert row[0] is None
