# Backtest Results — Simulation Version Notes

## simulate_day (old) — all results before 2026-05-03

All JSON files in this directory dated before 2026-05-03 were generated
using `backtest/simulate.py` (`simulate_day`), which had a known
discrepancy from production:

**Zone reset on trade close**: `simulate_day` reset only the active zone,
while the live bot (`BotTrader.on_tick`) resets ALL zones. This caused
~$4/day P&L difference (~10% higher than reality on the old sim).

Affected files include all `*_DEPLOYED_*.json`, `defensive_v2_*.json`,
`first_hour_v1_*.json`, `momentum_v1_*.json`, `momentum_v2_*.json`,
`direction_caps_v1_*.json`, `loss_limits_v1_*.json`, `loss_limits_v2_*.json`,
`adaptive_caps_v2_*.json`, `filter_removal_v1_*.json`, `profit_stop_v1_*.json`,
and all earlier results.

**Impact on relative comparisons**: Variant-vs-baseline comparisons within
the same file are still valid (both used the same sim), but absolute P&L
numbers are ~$4/day too high.

## simulate_day_v2 (accurate) — results from 2026-05-03 onward

`backtest/simulate_v2.py` uses the real `BotTrader` code with a
`BacktestBroker`, eliminating drift. Accurate baseline:
- **+$45.09/day** (vs +$49.46 from old sim)
- All future backtests should use `simulate_day_v2`.
