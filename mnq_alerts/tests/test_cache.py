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
