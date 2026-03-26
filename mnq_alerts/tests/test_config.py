"""Tests for config.py — verify all expected constants exist with correct types and values."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config


class TestMarketTiming:
    def test_market_open_hour(self):
        assert config.MARKET_OPEN_HOUR == 9

    def test_market_open_min(self):
        assert config.MARKET_OPEN_MIN == 30

    def test_market_close_hour(self):
        assert config.MARKET_CLOSE_HOUR == 16

    def test_market_close_min(self):
        assert config.MARKET_CLOSE_MIN == 0

    def test_ib_end_hour(self):
        assert config.IB_END_HOUR == 10

    def test_ib_end_min(self):
        assert config.IB_END_MIN == 30


class TestAlertConstants:
    def test_alert_threshold_points(self):
        assert config.ALERT_THRESHOLD_POINTS == 7

    def test_alert_exit_points(self):
        assert config.ALERT_EXIT_POINTS == 20

    def test_check_interval_seconds(self):
        assert config.CHECK_INTERVAL_SECONDS == 30


class TestOutcomeConstants:
    def test_hit_threshold(self):
        assert config.HIT_THRESHOLD == 1.0

    def test_target_points(self):
        assert config.TARGET_POINTS == 8.0

    def test_stop_points(self):
        assert config.STOP_POINTS == 20.0

    def test_eval_window_mins(self):
        assert config.EVAL_WINDOW_MINS == 15


class TestDatabentoConstants:
    def test_dataset(self):
        assert config.DATABENTO_DATASET == "GLBX.MDP3"

    def test_symbol(self):
        assert config.DATABENTO_SYMBOL == "MNQ.c.0"


class TestConstantTypes:
    def test_timing_are_ints(self):
        for attr in [
            "MARKET_OPEN_HOUR",
            "MARKET_OPEN_MIN",
            "MARKET_CLOSE_HOUR",
            "MARKET_CLOSE_MIN",
            "IB_END_HOUR",
            "IB_END_MIN",
        ]:
            assert isinstance(getattr(config, attr), int), f"{attr} should be int"

    def test_alert_threshold_is_int(self):
        assert isinstance(config.ALERT_THRESHOLD_POINTS, int)

    def test_outcome_are_floats(self):
        for attr in ["HIT_THRESHOLD", "TARGET_POINTS", "STOP_POINTS"]:
            assert isinstance(getattr(config, attr), float), f"{attr} should be float"

    def test_eval_window_is_int(self):
        assert isinstance(config.EVAL_WINDOW_MINS, int)

    def test_credential_strings_exist(self):
        """Credential attrs exist and are strings (may be empty in test env)."""
        for attr in ["DATABENTO_API_KEY", "PUSHOVER_TOKEN", "PUSHOVER_USER_KEY"]:
            assert isinstance(getattr(config, attr), str), f"{attr} should be str"
