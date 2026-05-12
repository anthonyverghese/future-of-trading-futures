"""Test-suite safety net: never let tests touch the production SQLite files.

Without this, any test that runs through a DB-writing code path (broker
entry/exit, log_alert, save trades, mark_open_bot_trades_orphaned) would
mutate the live alerts_log.db / .session_cache.db on the host running
pytest. On EC2 that pollutes the live bot's state — which actually
happened once (2026-05-12: TestEntryCancelRace fixtures left two
'orphaned' rows in production).

Pytest loads conftest.py files before any test module, so the module-
level monkeypatching here completes before broker.py / cache.py are
imported by any test. Every cache.* writer reads the path constant by
name at call time, so the redirect propagates without explicit fixtures.
"""

import atexit
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cache  # noqa: E402

_TMPDIR = tempfile.TemporaryDirectory(prefix="mnq-tests-")
cache.ALERTS_LOG_PATH = os.path.join(_TMPDIR.name, "alerts_log.db")
cache.CACHE_PATH = os.path.join(_TMPDIR.name, ".session_cache.db")

atexit.register(_TMPDIR.cleanup)
