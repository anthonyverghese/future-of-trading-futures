"""
cache.py — Persists session trades and notification history to local SQLite databases.

Session cache (.session_cache.db): checkpoints trades every CACHE_INTERVAL_SECONDS
so the app can resume from the last checkpoint on restart. Cleared on new session.

Alert log (alerts_log.db): permanent record of every push notification sent,
for data analysis. Never deleted automatically.
"""

from __future__ import annotations

import datetime
import os
import sqlite3

import pandas as pd
import pytz

ET = pytz.timezone("America/New_York")

CACHE_PATH = os.path.join(os.path.dirname(__file__), ".session_cache.db")
ALERTS_LOG_PATH = os.path.join(os.path.dirname(__file__), "alerts_log.db")
CACHE_INTERVAL_SECONDS = 300  # save every 5 minutes


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {"Price": pd.Series(dtype=float), "Size": pd.Series(dtype=int)},
        index=pd.DatetimeIndex([]),
    )


def clear_if_stale() -> None:
    """Delete the cache file if it contains no data for today (leftover from a prior session)."""
    if not os.path.exists(CACHE_PATH):
        return
    today = datetime.datetime.now(ET).date().isoformat()
    try:
        with sqlite3.connect(CACHE_PATH) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE date = ?", (today,)
            ).fetchone()[0]
        if count == 0:
            os.remove(CACHE_PATH)
            print("[cache] Stale cache from previous session cleared.")
    except Exception:
        pass


def save_trades(trades: pd.DataFrame) -> None:
    """Persist today's trades to the local cache database."""
    if trades.empty:
        return
    today = datetime.datetime.now(ET).date().isoformat()

    df = trades.copy()
    df.index = df.index.asi8  # nanoseconds since epoch (UTC)
    df = df.reset_index().rename(
        columns={"index": "ts_ns", "Price": "price", "Size": "size"}
    )
    df["date"] = today

    with sqlite3.connect(CACHE_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades
            (date TEXT, ts_ns INTEGER, price REAL, size INTEGER)
        """)
        conn.execute("DELETE FROM trades WHERE date = ?", (today,))
        df[["date", "ts_ns", "price", "size"]].to_sql(
            "trades", conn, if_exists="append", index=False
        )
    print(f"[cache] Saved {len(trades)} trades.")


def load_trades() -> pd.DataFrame:
    """Load today's cached trades. Returns empty DataFrame if none exist."""
    if not os.path.exists(CACHE_PATH):
        return _empty_trades()
    today = datetime.datetime.now(ET).date().isoformat()
    try:
        with sqlite3.connect(CACHE_PATH) as conn:
            df = pd.read_sql(
                "SELECT ts_ns, price, size FROM trades WHERE date = ? ORDER BY ts_ns",
                conn,
                params=(today,),
            )
    except Exception:
        return _empty_trades()

    if df.empty:
        return _empty_trades()

    ts_index = pd.to_datetime(df["ts_ns"], unit="ns", utc=True).dt.tz_convert(ET)
    return pd.DataFrame(
        {"Price": df["price"].values, "Size": df["size"].astype(int).values},
        index=ts_index,
    )


def get_replay_start(cached_trades: pd.DataFrame) -> datetime.datetime:
    """
    Return the timestamp to start replaying from.
    If cached trades exist AND cover from near session open, replay from 1
    second past the last checkpoint.  Otherwise replay from 9:30 AM ET so
    VWAP and IB are calculated from the full session.
    """
    now_et = datetime.datetime.now(ET)
    session_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)

    if cached_trades.empty:
        return session_open

    first_ts = cached_trades.index[0].to_pydatetime(warn=False)
    # If the cache doesn't start within 10 minutes of session open,
    # it's from a mid-session restart — replay the full session.
    if (first_ts - session_open).total_seconds() > 600:
        return session_open

    last_ts = cached_trades.index[-1].to_pydatetime(warn=False)
    return last_ts + datetime.timedelta(seconds=1)


def _ensure_alerts_schema(conn: sqlite3.Connection) -> None:
    """Create alerts and daily_stats tables if they don't exist, and migrate old schemas."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            date          TEXT,
            time          TEXT,
            ticker        TEXT,
            line          TEXT,
            line_price    REAL,
            current_price REAL,
            direction     TEXT,
            hit_time      TEXT,
            outcome       TEXT
        )
    """)
    # Migrate older schemas that may be missing the new columns.
    for col in [
        "current_price REAL",
        "direction TEXT",
        "hit_time TEXT",
        "outcome TEXT",
    ]:
        try:
            conn.execute(f"ALTER TABLE alerts ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass  # column already exists

    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_stats (
            date               TEXT PRIMARY KEY,
            ibh                REAL,
            ibl                REAL,
            notifications_sent INTEGER DEFAULT 0,
            correct_recs       INTEGER DEFAULT 0,
            incorrect_recs     INTEGER DEFAULT 0
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS bot_trades (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            date           TEXT,
            entry_time     TEXT,
            exit_time      TEXT,
            level_name     TEXT,
            direction      TEXT,
            line_price     REAL,
            entry_price    REAL,
            exit_price     REAL,
            target_price   REAL,
            stop_price     REAL,
            pnl_usd        REAL,
            outcome        TEXT,
            exit_reason    TEXT
        )
    """)


def log_alert(
    ticker: str,
    line: str,
    line_price: float,
    current_price: float,
    direction: str,
    trade_ts: datetime.datetime | None = None,
) -> int:
    """
    Persist a sent push notification to the permanent alerts log.
    Returns the alert id for outcome tracking.
    direction: 'up' if price was above the line (buy bias), 'down' if below (sell bias).
    trade_ts: the Databento trade timestamp that triggered the alert (preferred
              over wall clock so alert time and hit_time use the same source).
    """
    if trade_ts is not None:
        date_str = trade_ts.strftime("%Y-%m-%d")
        time_str = trade_ts.strftime("%H:%M:%S %Z")
    else:
        now_local = datetime.datetime.now().astimezone()
        date_str = now_local.strftime("%Y-%m-%d")
        time_str = now_local.strftime("%H:%M:%S %Z")

    with sqlite3.connect(ALERTS_LOG_PATH) as conn:
        _ensure_alerts_schema(conn)
        cur = conn.execute(
            """INSERT INTO alerts (date, time, ticker, line, line_price, current_price, direction)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (date_str, time_str, ticker, line, line_price, current_price, direction),
        )
        # Inline the daily_stats upsert — calling upsert_daily_stats() here
        # triggers _ensure_alerts_schema DDL which implicitly commits the
        # pending INSERT transaction and can cause "database is locked".
        conn.execute(
            """INSERT INTO daily_stats (date, notifications_sent, correct_recs, incorrect_recs)
               VALUES (?, 1, 0, 0)
               ON CONFLICT(date) DO UPDATE SET
                   notifications_sent = notifications_sent + 1""",
            (date_str,),
        )
        return cur.lastrowid


def update_alert_hit(alert_id: int, hit_time: str) -> None:
    """Record when price first touched the line for a pending alert."""
    with sqlite3.connect(ALERTS_LOG_PATH) as conn:
        conn.execute(
            "UPDATE alerts SET hit_time = ? WHERE id = ?",
            (hit_time, alert_id),
        )


def update_alert_outcome(alert_id: int, outcome: str, date_str: str) -> None:
    """
    Set the final outcome for an alert ('correct', 'incorrect', 'unresolved').
    Also increments the relevant counter in daily_stats.
    """
    with sqlite3.connect(ALERTS_LOG_PATH) as conn:
        conn.execute(
            "UPDATE alerts SET outcome = ? WHERE id = ?",
            (outcome, alert_id),
        )
        if outcome == "correct":
            conn.execute(
                """INSERT INTO daily_stats (date, notifications_sent, correct_recs, incorrect_recs)
                   VALUES (?, 0, 1, 0)
                   ON CONFLICT(date) DO UPDATE SET
                       correct_recs = correct_recs + 1""",
                (date_str,),
            )
        elif outcome == "incorrect":
            conn.execute(
                """INSERT INTO daily_stats (date, notifications_sent, correct_recs, incorrect_recs)
                   VALUES (?, 0, 0, 1)
                   ON CONFLICT(date) DO UPDATE SET
                       incorrect_recs = incorrect_recs + 1""",
                (date_str,),
            )


def get_daily_summary(date_str: str) -> dict[str, int]:
    """Return outcome counts for a given date: {correct, incorrect, inconclusive}."""
    result = {"correct": 0, "incorrect": 0, "inconclusive": 0}
    if not os.path.exists(ALERTS_LOG_PATH):
        return result
    try:
        with sqlite3.connect(ALERTS_LOG_PATH) as conn:
            _ensure_alerts_schema(conn)
            rows = conn.execute(
                """SELECT outcome, COUNT(*) FROM alerts
                   WHERE date = ? AND outcome IS NOT NULL
                   GROUP BY outcome""",
                (date_str,),
            ).fetchall()
        for outcome, count in rows:
            if outcome in result:
                result[outcome] = count
    except Exception:
        pass
    return result


def load_pending_alerts(date_str: str) -> list[dict]:
    """Load alerts with no outcome yet for the given date.

    Returns a list of dicts with keys: alert_id, line_price, direction,
    alert_time (datetime), date_str, hit_time (datetime or None).
    Used to resume outcome tracking after a restart.
    """
    if not os.path.exists(ALERTS_LOG_PATH):
        return []
    try:
        with sqlite3.connect(ALERTS_LOG_PATH) as conn:
            _ensure_alerts_schema(conn)
            rows = conn.execute(
                """SELECT id, line_price, direction, date, time, hit_time
                   FROM alerts
                   WHERE date = ? AND outcome IS NULL AND direction IS NOT NULL
                   ORDER BY id""",
                (date_str,),
            ).fetchall()
    except Exception:
        return []

    results = []
    for row in rows:
        alert_id, line_price, direction, date, time_str, hit_time_str = row
        # Parse alert_time from date + time columns (e.g. "2026-03-23" + "12:09:03 PDT")
        try:
            # Strip timezone abbreviation — we know it's local time on the server.
            time_clean = time_str.rsplit(" ", 1)[0] if " " in time_str else time_str
            alert_dt = datetime.datetime.strptime(
                f"{date} {time_clean}", "%Y-%m-%d %H:%M:%S"
            )
        except (ValueError, TypeError):
            continue
        hit_dt = None
        if hit_time_str:
            try:
                hit_dt = datetime.datetime.fromisoformat(hit_time_str)
            except (ValueError, TypeError):
                pass
        results.append(
            {
                "alert_id": alert_id,
                "line_price": line_price,
                "direction": direction,
                "alert_time": alert_dt,
                "date_str": date,
                "hit_time": hit_dt,
            }
        )
    return results


def load_recent_outcomes(limit: int = 10) -> list[str]:
    """Load the most recent decided outcomes from alerts_log.db.

    Returns a chronological list of 'correct'/'incorrect' strings,
    oldest first, so streak tracking can persist across sessions.
    """
    if not os.path.exists(ALERTS_LOG_PATH):
        return []
    try:
        with sqlite3.connect(ALERTS_LOG_PATH) as conn:
            rows = conn.execute(
                """SELECT outcome FROM alerts
                   WHERE outcome IN ('correct', 'incorrect')
                   ORDER BY id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        # Rows come newest-first; reverse to chronological order.
        return [r[0] for r in reversed(rows)]
    except Exception:
        return []


def log_bot_trade_entry(
    date_str: str,
    entry_time: str,
    level_name: str,
    direction: str,
    line_price: float,
    entry_price: float,
    target_price: float,
    stop_price: float,
) -> int:
    """Log a bot trade entry. Returns the row id for later update on exit."""
    with sqlite3.connect(ALERTS_LOG_PATH) as conn:
        _ensure_alerts_schema(conn)
        cur = conn.execute(
            """INSERT INTO bot_trades
               (date, entry_time, level_name, direction, line_price,
                entry_price, target_price, stop_price, outcome)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')""",
            (
                date_str,
                entry_time,
                level_name,
                direction,
                line_price,
                entry_price,
                target_price,
                stop_price,
            ),
        )
        return cur.lastrowid


def update_bot_trade_exit(
    trade_id: int,
    exit_time: str,
    exit_price: float,
    pnl_usd: float,
    outcome: str,
    exit_reason: str,
) -> None:
    """Update a bot trade with exit details."""
    with sqlite3.connect(ALERTS_LOG_PATH) as conn:
        conn.execute(
            """UPDATE bot_trades
               SET exit_time = ?, exit_price = ?, pnl_usd = ?,
                   outcome = ?, exit_reason = ?
               WHERE id = ?""",
            (exit_time, exit_price, pnl_usd, outcome, exit_reason, trade_id),
        )


def get_bot_daily_summary(date_str: str) -> dict:
    """Return bot trading stats for a given date."""
    result = {"trades": 0, "wins": 0, "losses": 0, "pnl_usd": 0.0}
    if not os.path.exists(ALERTS_LOG_PATH):
        return result
    try:
        with sqlite3.connect(ALERTS_LOG_PATH) as conn:
            _ensure_alerts_schema(conn)
            row = conn.execute(
                """SELECT COUNT(*),
                          SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END),
                          SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END),
                          COALESCE(SUM(pnl_usd), 0)
                   FROM bot_trades
                   WHERE date = ? AND outcome != 'open'""",
                (date_str,),
            ).fetchone()
            if row:
                result["trades"] = row[0]
                result["wins"] = row[1] or 0
                result["losses"] = row[2] or 0
                result["pnl_usd"] = row[3] or 0.0
    except Exception:
        pass
    return result


def upsert_daily_stats(
    date: str,
    ibh: float | None = None,
    ibl: float | None = None,
) -> None:
    """Insert or update the daily_stats row for the given date (IBH/IBL only)."""
    with sqlite3.connect(ALERTS_LOG_PATH) as conn:
        _ensure_alerts_schema(conn)
        conn.execute(
            """INSERT INTO daily_stats (date, ibh, ibl, notifications_sent, correct_recs, incorrect_recs)
               VALUES (?, ?, ?, 0, 0, 0)
               ON CONFLICT(date) DO UPDATE SET
                   ibh = COALESCE(excluded.ibh, ibh),
                   ibl = COALESCE(excluded.ibl, ibl)""",
            (date, ibh, ibl),
        )
