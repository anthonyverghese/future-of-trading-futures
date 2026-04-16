"""Tests for ib-gateway-watchdog.py regex patterns."""

import re

import pytest

# Copy the exact patterns from ib-gateway-watchdog.py to avoid import
# complexity (the watchdog lives outside mnq_alerts/).
LOGIN_FAILED_RE = re.compile(
    r"IBC:.*(Too many failed login attempts|Login.*failed|password.*incorrect|Invalid login|Unrecognized Username or Password)",
    re.IGNORECASE,
)
LOGIN_SUCCESS_RE = re.compile(
    r"IBC:.*(Login has completed|Logged in|Connected to)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 1. LOGIN_FAILED_RE
# ---------------------------------------------------------------------------


class TestLoginFailedRegex:
    def test_matches_too_many_failed(self):
        line = "IBC: Too many failed login attempts"
        assert LOGIN_FAILED_RE.search(line) is not None

    def test_matches_unrecognized_username(self):
        line = (
            "IBC: detected dialog entitled: Unrecognized Username or Password; "
            "event=Opened"
        )
        assert LOGIN_FAILED_RE.search(line) is not None

    def test_matches_password_incorrect(self):
        line = "IBC: password incorrect"
        assert LOGIN_FAILED_RE.search(line) is not None

    def test_matches_invalid_login(self):
        line = "IBC: Invalid login"
        assert LOGIN_FAILED_RE.search(line) is not None

    def test_does_not_match_login_completed(self):
        line = "IBC: Login has completed"
        assert LOGIN_FAILED_RE.search(line) is None

    def test_does_not_match_relogin_prompt(self):
        line = "IBC: Re-login to session"
        assert LOGIN_FAILED_RE.search(line) is None

    def test_does_not_match_random_line(self):
        line = "2026-04-15 socat listening..."
        assert LOGIN_FAILED_RE.search(line) is None

    def test_matches_with_timestamp_prefix(self):
        line = (
            "2026-04-15 00:50:06:203 IBC: detected dialog entitled: "
            "Unrecognized Username or Password; event=Activated"
        )
        assert LOGIN_FAILED_RE.search(line) is not None


# ---------------------------------------------------------------------------
# 2. LOGIN_SUCCESS_RE
# ---------------------------------------------------------------------------


class TestLoginSuccessRegex:
    def test_matches_login_completed(self):
        line = "IBC: Login has completed"
        assert LOGIN_SUCCESS_RE.search(line) is not None

    def test_matches_logged_in(self):
        line = "IBC: Logged in"
        assert LOGIN_SUCCESS_RE.search(line) is not None

    def test_does_not_match_failure(self):
        line = "IBC: Too many failed login attempts"
        assert LOGIN_SUCCESS_RE.search(line) is None
