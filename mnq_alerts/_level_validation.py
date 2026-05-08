"""Walk-forward folds, gates, and architecture selection."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FINAL_TEST_TRADING_DAYS = 30
EMBARGO_DAYS = 1


@dataclass
class FoldDef:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_walk_forward_folds(trading_days: pd.DatetimeIndex) -> list[FoldDef]:
    """Quarterly walk-forward folds over the dev set.

    Final FINAL_TEST_TRADING_DAYS days are reserved for out-of-time test (excluded).
    EMBARGO_DAYS days dropped from the start of each test fold.
    """
    if len(trading_days) <= FINAL_TEST_TRADING_DAYS:
        return []
    dev_days = trading_days[:-FINAL_TEST_TRADING_DAYS]
    quarters = sorted({(d.year, (d.month - 1) // 3 + 1) for d in dev_days})
    folds: list[FoldDef] = []
    for i in range(1, len(quarters)):
        train_quarters = quarters[:i]
        train_days = [d for d in dev_days if (d.year, (d.month - 1) // 3 + 1) in train_quarters]
        test_days = [d for d in dev_days if (d.year, (d.month - 1) // 3 + 1) == quarters[i]]
        if len(train_days) < 20 or len(test_days) < 5:
            continue
        test_days_emb = test_days[EMBARGO_DAYS:]
        if len(test_days_emb) < 5:
            continue
        folds.append(FoldDef(
            train_start=train_days[0],
            train_end=train_days[-1],
            test_start=test_days_emb[0],
            test_end=test_days_emb[-1],
        ))
    return folds
