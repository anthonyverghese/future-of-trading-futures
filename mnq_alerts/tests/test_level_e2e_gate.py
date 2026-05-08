"""Tests for E2E gate verification — the v3 catch."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _level_e2e_gate import compare_offline_vs_replay, E2EGateResult


def test_identical_predictions_pass_gate():
    base = pd.Timestamp("2025-06-02 14:31:00", tz="UTC")
    offline = {
        base + pd.Timedelta(minutes=5): {("bounce", 8, 25): 0.6},
        base + pd.Timedelta(minutes=10): {("bounce", 8, 25): 0.7},
    }
    replay = dict(offline)
    result = compare_offline_vs_replay(offline, replay)
    assert result.passes is True
    assert result.max_diff < 1e-9


def test_large_divergence_fails_gate():
    base = pd.Timestamp("2025-06-02 14:31:00", tz="UTC")
    offline = {base + pd.Timedelta(minutes=5): {("bounce", 8, 25): 0.6}}
    replay = {base + pd.Timedelta(minutes=5): {("bounce", 8, 25): 0.9}}
    result = compare_offline_vs_replay(offline, replay)
    assert result.passes is False
    assert result.max_diff > 0.1


def test_threshold_pass_at_002():
    base = pd.Timestamp("2025-06-02 14:31:00", tz="UTC")
    offline = {base + pd.Timedelta(minutes=5): {("bounce", 8, 25): 0.600}}
    replay = {base + pd.Timedelta(minutes=5): {("bounce", 8, 25): 0.615}}
    result = compare_offline_vs_replay(offline, replay)
    assert result.passes is True
