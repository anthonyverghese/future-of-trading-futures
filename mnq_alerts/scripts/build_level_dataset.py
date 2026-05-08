"""CLI: build the full-history labeled+featured dataset.

Usage: python -m mnq_alerts.scripts.build_level_dataset
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _level_dataset import build_full_history


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parquet_dir = os.path.join(here, "data_cache")
    out_path = os.path.join(here, "_level_events_labeled.parquet")
    n = build_full_history(parquet_dir, out_path)
    print(f"Wrote {n} rows to {out_path}")


if __name__ == "__main__":
    main()
