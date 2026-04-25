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


def export_daily_parquet(trades: pd.DataFrame) -> str | None:
    """Export today's session trades to a parquet file for backtesting.

    Saves to data_cache/MNQ_<date>.parquet in the same format as the
    historical Databento downloads (columns: price, size; index: ET
    DatetimeIndex). This lets the backtest infrastructure use live-
    collected data without needing Databento.

    Returns the file path on success, None on failure.
    """
    if trades.empty:
        print("[cache] No trades to export to parquet.")
        return None
    if "Price" not in trades.columns or "Size" not in trades.columns:
        print(
            f"[cache] Cannot export parquet: unexpected columns "
            f"{list(trades.columns)}, expected ['Price', 'Size']"
        )
        return None

    today = datetime.datetime.now(ET).date().isoformat()
    cache_dir = os.path.join(os.path.dirname(__file__), "data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"MNQ_{today}.parquet")

    # Don't overwrite a larger existing file (e.g. from a mid-day restart
    # where the second session has fewer trades than the first).
    if os.path.exists(path):
        try:
            existing = pd.read_parquet(path)
            if len(existing) >= len(trades):
                print(
                    f"[cache] Parquet already exists with {len(existing)} trades "
                    f"(>= {len(trades)} current) — skipping overwrite"
                )
                return path
        except Exception:
            pass  # corrupted file, overwrite it

    try:
        # Convert from session format (Price/Size) to backtest format
        # (price/size) to match historical Databento parquet files.
        df = pd.DataFrame(
            {"price": trades["Price"].values, "size": trades["Size"].values},
            index=trades.index,
        )
        df.index.name = "ts"  # match Databento parquet format
        df.to_parquet(path)
        print(f"[cache] Exported {len(df)} trades to {path}")
        return path
    except Exception as e:
        print(f"[cache] Failed to export parquet: {e}")
        return None


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
            outcome       TEXT,
            score         INTEGER,
            tier          TEXT,
            range_30m     REAL,
            entry_count   INTEGER
        )
    """)
    # Migrate older schemas that may be missing the new columns.
    for col in [
        "current_price REAL",
        "direction TEXT",
        "hit_time TEXT",
        "outcome TEXT",
        "score INTEGER",
        "tier TEXT",
        "range_30m REAL",
        "entry_count INTEGER",
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
            exit_reason    TEXT,
            score          INTEGER,
            trend_60m      REAL,
            entry_count    INTEGER,
            parent_order_id INTEGER
        )
    """)
    # Migrate older bot_trades schemas that may be missing the new columns.
    for col in [
        "score INTEGER",
        "trend_60m REAL",
        "entry_count INTEGER",
        "parent_order_id INTEGER",
    ]:
        try:
            conn.execute(f"ALTER TABLE bot_trades ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass  # column already exists


def log_alert(
    ticker: str,
    line: str,
    line_price: float,
    current_price: float,
    direction: str,
    trade_ts: datetime.datetime | None = None,
    score: int | None = None,
    tier: str | None = None,
    range_30m: float | None = None,
    entry_count: int | None = None,
) -> int:
    """
    Persist a sent push notification to the permanent alerts log.
    Returns the alert id for outcome tracking.
    direction: 'up' if price was above the line (buy bias), 'down' if below (sell bias).
    trade_ts: the Databento trade timestamp that triggered the alert (preferred
              over wall clock so alert time and hit_time use the same source).
    score: composite score that gated the alert.
    tier: signal tier ("Good"/"Strong"/"Elite") shown in the notification.
    range_30m: 30-minute price range at alert time (volatility metric).
    entry_count: which retest of this level (1=first, 2=second, etc.).
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
            """INSERT INTO alerts (date, time, ticker, line, line_price, current_price, direction, score, tier, range_30m, entry_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                date_str,
                time_str,
                ticker,
                line,
                line_price,
                current_price,
                direction,
                score,
                tier,
                range_30m,
                entry_count,
            ),
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
    score: int | None = None,
    trend_60m: float | None = None,
    entry_count: int | None = None,
    parent_order_id: int | None = None,
) -> int:
    """Log a bot trade entry. Returns the row id for later update on exit.

    score: bot entry score that passed the BOT_MIN_SCORE filter.
    trend_60m: 60-minute price trend at entry (positive = up).
    entry_count: which retest of this level (1=first, 2=second, etc.).
    parent_order_id: IBKR client-side orderId of the parent (market) order.
        Stored so a subsequent restart can look this row up when adopting
        an open position via orderRef on the live bracket children.
    """
    with sqlite3.connect(ALERTS_LOG_PATH) as conn:
        _ensure_alerts_schema(conn)
        cur = conn.execute(
            """INSERT INTO bot_trades
               (date, entry_time, level_name, direction, line_price,
                entry_price, target_price, stop_price, outcome,
                score, trend_60m, entry_count, parent_order_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?)""",
            (
                date_str,
                entry_time,
                level_name,
                direction,
                line_price,
                entry_price,
                target_price,
                stop_price,
                score,
                trend_60m,
                entry_count,
                parent_order_id,
            ),
        )
        return cur.lastrowid


def load_bot_open_trade_by_parent_order_id(
    parent_order_id: int, date_str: str
) -> dict | None:
    """Look up an open bot_trades row by parent_order_id (the IBKR parent
    orderId tagged at submission time).

    Used by IBKRBroker._reconcile_open_position() after a restart: we
    parse the orderRef off the live bracket children, pull parent_order_id
    out, and call this to hydrate _pending_* state.

    Returns None if no matching row is found.
    """
    if not os.path.exists(ALERTS_LOG_PATH):
        return None
    try:
        with sqlite3.connect(ALERTS_LOG_PATH) as conn:
            _ensure_alerts_schema(conn)
            row = conn.execute(
                """SELECT id, level_name, direction, line_price, entry_price,
                          target_price, stop_price, score, trend_60m, entry_count
                   FROM bot_trades
                   WHERE parent_order_id = ? AND date = ? AND outcome = 'open'""",
                (parent_order_id, date_str),
            ).fetchone()
    except Exception:
        return None
    if row is None:
        return None
    return {
        "id": row[0],
        "level_name": row[1],
        "direction": row[2],
        "line_price": row[3],
        "entry_price": row[4],
        "target_price": row[5],
        "stop_price": row[6],
        "score": row[7],
        "trend_60m": row[8],
        "entry_count": row[9],
    }


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


def mark_open_bot_trades_orphaned(date_str: str) -> int:
    """Mark any `outcome='open'` bot_trades rows for the given date as
    `orphaned`. Returns the number of rows updated.

    Called from IBKRBroker._defensive_flatten after a failed reconcile,
    so stale open rows don't silently under-count future daily restores
    (which filter `outcome != 'open'`).
    """
    if not os.path.exists(ALERTS_LOG_PATH):
        return 0
    try:
        with sqlite3.connect(ALERTS_LOG_PATH) as conn:
            _ensure_alerts_schema(conn)
            cur = conn.execute(
                """UPDATE bot_trades
                   SET outcome = 'orphaned', exit_reason = 'defensive_flatten'
                   WHERE date = ? AND outcome = 'open'""",
                (date_str,),
            )
            return cur.rowcount
    except Exception:
        return 0


def load_bot_daily_risk_state(date_str: str) -> dict:
    """Restore broker risk counters from today's closed bot_trades.

    Scans rows where outcome != 'open' in entry order. Returns:
      pnl_usd, trades, wins, losses, consecutive_losses
    consecutive_losses walks back from the tail until the first non-loss,
    matching how broker._on_order_status maintains the streak.

    Called by IBKRBroker.connect() so a mid-day restart doesn't hand the
    bot a fresh daily-loss budget or clear a consecutive-loss stop.
    """
    state = {
        "pnl_usd": 0.0,
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "consecutive_losses": 0,
    }
    if not os.path.exists(ALERTS_LOG_PATH):
        return state
    try:
        with sqlite3.connect(ALERTS_LOG_PATH) as conn:
            _ensure_alerts_schema(conn)
            rows = conn.execute(
                """SELECT outcome, COALESCE(pnl_usd, 0) FROM bot_trades
                   WHERE date = ? AND outcome != 'open'
                   ORDER BY id""",
                (date_str,),
            ).fetchall()
    except Exception:
        return state

    for outcome, pnl in rows:
        state["trades"] += 1
        state["pnl_usd"] += pnl
        if outcome == "win":
            state["wins"] += 1
        elif outcome == "loss":
            state["losses"] += 1

    for outcome, _ in reversed(rows):
        if outcome == "loss":
            state["consecutive_losses"] += 1
        else:
            break
    return state


def load_bot_daily_level_counts(date_str: str) -> dict[str, int]:
    """Return {level_name: trade_count} for today's closed bot_trades.

    Used by BotTrader.connect() to restore the per-level daily cap
    (BOT_MAX_ENTRIES_PER_LEVEL) so a restart can't hand each level a
    fresh allotment.
    """
    counts: dict[str, int] = {}
    if not os.path.exists(ALERTS_LOG_PATH):
        return counts
    try:
        with sqlite3.connect(ALERTS_LOG_PATH) as conn:
            _ensure_alerts_schema(conn)
            rows = conn.execute(
                """SELECT level_name, COUNT(*) FROM bot_trades
                   WHERE date = ? AND outcome != 'open'
                   GROUP BY level_name""",
                (date_str,),
            ).fetchall()
    except Exception:
        return counts
    for level_name, count in rows:
        if level_name:
            counts[level_name] = count
    return counts


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
