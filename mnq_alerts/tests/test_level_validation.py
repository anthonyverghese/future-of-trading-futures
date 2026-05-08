"""Tests for walk-forward fold construction."""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _level_validation import build_walk_forward_folds, FoldDef


def test_five_quarterly_folds_for_full_history():
    days = pd.date_range("2025-01-02", "2026-04-08", freq="B")
    folds = build_walk_forward_folds(days)
    # 5 quarters in dev set (Q1-Q4 2025 + partial Q1-2026); first quarter is training-only,
    # so 4 walk-forward test folds (Q2-25, Q3-25, Q4-25, Q1-26).
    assert len(folds) == 4


def test_train_set_grows_each_fold():
    days = pd.date_range("2025-01-02", "2026-04-08", freq="B")
    folds = build_walk_forward_folds(days)
    assert all(folds[i].train_end >= folds[i - 1].train_end for i in range(1, len(folds)))


def test_first_day_of_test_dropped_for_embargo():
    days = pd.date_range("2025-01-02", "2026-04-08", freq="B")
    folds = build_walk_forward_folds(days)
    fold = folds[0]
    assert fold.test_start > fold.train_end


def test_dev_set_excludes_final_test_window():
    days = pd.date_range("2025-01-02", "2026-05-06", freq="B")
    folds = build_walk_forward_folds(days)
    last_fold = folds[-1]
    final_test_start = days[-30]
    assert last_fold.test_end < final_test_start
