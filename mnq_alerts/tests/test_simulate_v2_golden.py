"""Golden results test: verify simulate_day_v2 matches simulate_day.

Loads golden results generated from the current simulate_day and
verifies simulate_day_v2 (using real BotTrader + BacktestBroker)
produces identical trade sequences.

Run golden generation first:
    PYTHONPATH=. python -c "..." > tests/golden_sim_results.json

Then run this test:
    PYTHONPATH=. python -m pytest tests/test_simulate_v2_golden.py -v
"""

import json
import os
import sys
import datetime

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load golden results
GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_sim_results.json")


@pytest.fixture(scope="module")
def golden():
    if not os.path.exists(GOLDEN_PATH):
        pytest.skip("Golden results file not found — run generation first")
    with open(GOLDEN_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def sim_data():
    """Load tick data (slow — cached for module)."""
    from backtest.data import load_all_days, precompute_arrays
    dates, caches = load_all_days()
    arrays = {d: precompute_arrays(caches[d]) for d in dates}
    return dates, caches, arrays


def _run_v2_for_day(caches, arrays, day):
    """Run simulate_day_v2 for a specific day."""
    from backtest.simulate_v2 import simulate_day_v2

    dc = caches[day]
    arr = arrays[day]
    caps = {
        "FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
        "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7,
    }
    if day.weekday() == 0:
        caps = {k: v * 2 for k, v in caps.items()}

    trades = simulate_day_v2(
        dc, arr,
        per_level_ts={
            "FIB_EXT_HI_1.272": (6, 20), "FIB_EXT_LO_1.272": (6, 20),
            "FIB_0.236": (8, 25), "FIB_0.618": (12, 20), "FIB_0.764": (10, 25),
            "IBH": (6, 20),
        },
        per_level_caps=caps,
        exclude_levels={"FIB_0.5", "IBL"},
        include_ibl=False,
        include_vwap=False,
        direction_filter={"IBH": "down"},
        daily_loss=200.0,
        momentum_max=5.0,
    )
    return trades


class TestSimulateV2MatchesGolden:
    """Verify simulate_day_v2 produces same results as simulate_day."""

    def test_trade_count_matches(self, golden, sim_data):
        """Same number of trades per day."""
        _, caches, arrays = sim_data
        for day_str, expected in golden.items():
            day = datetime.date.fromisoformat(day_str)
            if day not in caches:
                continue
            trades = _run_v2_for_day(caches, arrays, day)
            assert len(trades) == expected["trades"], \
                f"{day}: expected {expected['trades']} trades, got {len(trades)}"

    def test_pnl_matches(self, golden, sim_data):
        """Same total P&L per day."""
        _, caches, arrays = sim_data
        for day_str, expected in golden.items():
            day = datetime.date.fromisoformat(day_str)
            if day not in caches:
                continue
            trades = _run_v2_for_day(caches, arrays, day)
            pnl = round(sum(t.pnl_usd for t in trades), 2)
            assert pnl == expected["pnl"], \
                f"{day}: expected P&L ${expected['pnl']}, got ${pnl}"

    def test_trade_details_match(self, golden, sim_data):
        """Same level, direction, outcome, and P&L per trade."""
        _, caches, arrays = sim_data
        for day_str, expected in golden.items():
            day = datetime.date.fromisoformat(day_str)
            if day not in caches:
                continue
            trades = _run_v2_for_day(caches, arrays, day)
            for i, (actual, exp) in enumerate(zip(trades, expected["trade_details"])):
                assert actual.level == exp["level"], \
                    f"{day} trade {i}: level {actual.level} != {exp['level']}"
                assert actual.direction == exp["direction"], \
                    f"{day} trade {i}: direction {actual.direction} != {exp['direction']}"
                assert actual.outcome == exp["outcome"], \
                    f"{day} trade {i}: outcome {actual.outcome} != {exp['outcome']}"
                assert round(actual.pnl_usd, 2) == exp["pnl_usd"], \
                    f"{day} trade {i}: pnl {actual.pnl_usd} != {exp['pnl_usd']}"
