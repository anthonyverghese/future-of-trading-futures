"""
recache_with_side.py — Re-download cached trade data with aggressor side field.

The existing parquet files only have (price, size). This script re-downloads
each day from Databento's Historical API and adds the 'side' column:
  'B' = buyer aggressor (hit the ask)
  'A' = seller aggressor (hit the bid)
  'N' = no aggressor info

Backs up old files to data_cache/backup/ before overwriting.

Usage:
    python recache_with_side.py          # re-download all days
    python recache_with_side.py --test   # re-download just 1 day to verify
"""

from __future__ import annotations

import datetime
import os
import shutil
import sys

import databento as db
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(__file__))
from config import DATABENTO_API_KEY
from targeted_backtest import load_cached_days

ET = pytz.timezone("America/New_York")

DATASET = "GLBX.MDP3"
SYMBOL = "MNQ.c.0"
MARKET_OPEN = datetime.time(9, 30)
MARKET_CLOSE = datetime.time(16, 0)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")
BACKUP_DIR = os.path.join(CACHE_DIR, "backup")


def side_char(side_val) -> str:
    """Convert Databento Side enum to single char."""
    name = side_val.name if hasattr(side_val, "name") else str(side_val)
    if name == "BID":
        return "B"  # buyer aggressor
    elif name == "ASK":
        return "A"  # seller aggressor
    return "N"  # none/unknown


def fetch_with_side(client: db.Historical, date: datetime.date) -> pd.DataFrame:
    """Fetch RTH trades for one day including aggressor side."""
    start = ET.localize(datetime.datetime.combine(date, MARKET_OPEN)).isoformat()
    end = ET.localize(datetime.datetime.combine(date, MARKET_CLOSE)).isoformat()

    store = client.timeseries.get_range(
        dataset=DATASET,
        schema="trades",
        stype_in="continuous",
        symbols=[SYMBOL],
        start=start,
        end=end,
    )

    rows = []
    for rec in store:
        if not isinstance(rec, db.TradeMsg):
            continue
        ts = pd.Timestamp(rec.ts_event, unit="ns", tz="UTC").tz_convert(ET)
        price = rec.price / 1_000_000_000
        size = int(rec.size)
        side = side_char(rec.side)
        rows.append((ts, price, size, side))

    if not rows:
        return pd.DataFrame(columns=["price", "size", "side"])

    df = (
        pd.DataFrame(rows, columns=["ts", "price", "size", "side"])
        .set_index("ts")
        .sort_index()
    )
    return df


def main() -> None:
    test_mode = "--test" in sys.argv
    days = load_cached_days()

    if test_mode:
        days = days[:1]
        print(f"TEST MODE: re-downloading only {days[0]}")
    else:
        print(f"Re-downloading {len(days)} days with side field...")

    # Check which days already have side column
    need_download = []
    for date in days:
        path = os.path.join(CACHE_DIR, f"MNQ_{date}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if "side" in df.columns:
                continue
        need_download.append(date)

    if not need_download:
        print("All cached files already have 'side' column. Nothing to do.")
        return

    print(f"  {len(need_download)} days need re-downloading.")

    # Back up existing files
    os.makedirs(BACKUP_DIR, exist_ok=True)

    client = db.Historical(DATABENTO_API_KEY)

    for i, date in enumerate(need_download):
        path = os.path.join(CACHE_DIR, f"MNQ_{date}.parquet")

        # Backup old file
        if os.path.exists(path):
            backup_path = os.path.join(BACKUP_DIR, f"MNQ_{date}.parquet")
            if not os.path.exists(backup_path):
                shutil.copy2(path, backup_path)

        try:
            df = fetch_with_side(client, date)
            df.to_parquet(path)
            side_counts = df["side"].value_counts().to_dict()
            print(
                f"  [{i+1}/{len(need_download)}] {date}: {len(df):,} trades "
                f"(B={side_counts.get('B', 0)}, A={side_counts.get('A', 0)}, "
                f"N={side_counts.get('N', 0)})"
            )
        except Exception as e:
            print(f"  [{i+1}/{len(need_download)}] {date}: ERROR — {e}")

        if test_mode:
            print("\nTest complete. Check the output above.")
            print(f"Sample:\n{df.head(10).to_string()}")
            return

    print(f"\nDone. {len(need_download)} days re-downloaded with side field.")


if __name__ == "__main__":
    main()
