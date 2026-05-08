# Tick-data-driven per-level outcome model — design spec

**Date:** 2026-05-08
**Status:** Design approved, plan pending
**Supersedes context:** v3 / v4 ML attempts (rejected 2026-05-07; see `project_v3_v4_models_rejected_2026_05_07.md` memory). Those attempts modeled trigger selection only; this attempt redesigns the bot's level-interaction logic.

## Motivation

The bot's current strategy (V6, deployed 2026-05-06) is a bounce-only trader with hardcoded direction per level and per-level test caps. Quarterly P&L has decayed from +$37.63/day (Q1'25) to +$6.78/day (Q4'25); recent 60-day full-sample P&L is +$1.83/day. The human alert algorithm continues to outperform the bot (~+$19–21/day) despite the bot having strictly more information available.

Hypothesis driving this design:
1. Each price level has a unique outcome distribution given approach context — not just "bounce probability" but a joint distribution over (bounce wins, bounce loses, breakthrough wins, breakthrough loses).
2. Approach kinematics (speed, acceleration, path efficiency) materially shift the outcome distribution. Fast approaches favor breakthrough; slow approaches favor bounce.
3. Tick-rule aggressor balance and trade-size profile reveal institutional footprint and shift outcome probabilities further.
4. With trade-tick data alone (no order book), we cannot capture queue dynamics or iceberg behavior, but we CAN extract aggressor inference, surge detection, and large-print signal.

The design produces a model that, at each level-touch event, outputs `P(win)` for each of 8 trade variants (2 directions × 4 TP/SL combos). The bot picks the variant with highest expected P&L, or skips.

This is a departure from V6's hardcoded direction-per-level structure (e.g., V6's `BOT_DIRECTION_FILTER` blocks IBH BUY). Under this design, every level can fire in either direction if the model finds a positive-expectation trade there. V6's per-level direction caps and test caps are NOT carried forward — the model replaces them entirely.

## Out of scope

- VWAP as a tradeable level (excluded per user direction; backtest history shows it as unreliable). VWAP retained only as `distance_to_vwap` informational feature.
- Order book / Level 2 features (data not available).
- Trade-streak features (consecutive wins/losses). Excluded by construction — yesterday's v3/v4 work confirmed even resolution-order-safe streaks fail to find real signal.
- Hyperparameter search across folds. Single conservative config used identically for both architectures to keep comparison fair and avoid implicit search-leakage.

## Section 1 — Event extraction & labels

### Event pipeline

A new module `mnq_alerts/_level_events.py` that, given a day's tick parquet:

1. Compute IBH/IBL/FIB levels for that session (reuse `levels.py`; IB locked at 10:31 ET per existing config, FIB derived from IB, fixed for the day).
2. Walk ticks forward chronologically. For each level (excluding VWAP), emit an event whenever `|price − level| ≤ 1.0` AND the level is armed.
3. Per-level arming state machine:
   - All levels (IBH, IBL, FIB derivatives) arm at 10:31 ET when IB locks. No events fire on these levels before 10:31.
   - After an event, the level is disarmed.
   - Re-arms when `|price − level| > 3.0`.
   - Independent of trade outcome — re-arming is purely about price distance from the level. (TP and SL are both > 3.0, so trade closures naturally place price outside the re-arm zone; timeouts may leave price inside the 3pt zone, in which case re-arming waits for price to exit.)
4. Each event records: `event_ts`, `level_name`, `level_price`, `event_price`, `approach_direction` (sign of `event_price − price_60s_before`).

Position constraint (1 open position max) is NOT applied at labeling time — every event is labeled regardless of bot position state. The 1-position rule applies at simulation time only. This separation ensures labeled statistics reflect the universe of events the bot could face, not events conditional on the bot's position-state history.

### Labels

For each event, compute 8 binary labels:

For each `(direction ∈ {bounce, breakthrough}) × ((TP, SL) ∈ {(8,25), (8,20), (10,25), (10,20)})`:

- **Bounce trade:** entry at `event_price`, target = price moving AWAY from approach direction by TP, stop = price moving WITH approach direction by SL.
- **Breakthrough trade:** entry at `event_price`, target = price moving WITH approach direction by TP, stop = price moving AWAY from approach direction by SL.
- Resolution window: `min(event_ts + 15min, 4:00 PM ET)`.
- **Win** = TP touched first within resolution window.
- **Loss** = SL touched first OR neither TP nor SL hit by window end (15-min cap or market close, whichever comes first).
- Track `time_to_resolution` for analysis.

Entry slippage modeled by using the next tick price after `event_ts`. Exit slippage modeled by using the tick that crosses TP/SL, not interpolated — matches existing slippage-aware backtest infrastructure.

Events firing AFTER 4:00 PM ET are dropped.

### Output

`mnq_alerts/_level_events_labeled.parquet`. One row per `(event, direction, TP, SL)`: event metadata + binary label + `time_to_resolution`.

## Section 2 — Features

Computed strictly from data ending at `event_ts`. Resolution-order safe. Multi-window features take snapshots over `{5s, 30s, 5min, 15min}` lookbacks where applicable.

### Family 1 — Approach kinematics

- `velocity_5s`, `velocity_30s`, `velocity_5min` — signed points per second
- `acceleration_30s` — Δvelocity over 30s
- `path_efficiency_5min` — `|displacement| / Σ|tick-to-tick moves|`. 1.0 = straight-line approach, low = choppy.

### Family 2 — Tick-rule aggressor balance

- `aggressor_balance_5s`, `aggressor_balance_30s`, `aggressor_balance_5min` — `(buy_volume − sell_volume) / total_volume`, using uptick/downtick aggressor inference (zero-tick inherits prior side)
- `net_dollar_flow_5min` — `Σ(price × size × side)` over 5min

### Family 3 — Volume / size profile

- `volume_5s`, `volume_30s`, `volume_5min` — total contracts traded
- `trade_rate_30s` — trades per second
- `max_print_size_30s` — largest single trade in last 30s
- `volume_concentration_30s` — `Σ(size_i²) / (Σsize_i)²`, Herfindahl-style. High = a few big trades, low = many small.

### Family 4 — Level-specific context

- `touches_today` — count of prior events at this level today
- `prior_touch_outcome` — categorical: `none / bounce_held / bounce_failed / breakthrough_held / breakthrough_failed`. Computed in **resolution-order**: a prior touch's outcome only becomes visible after its resolution timestamp. Highest leakage risk in the feature set; covered by a unit test.
- `seconds_since_last_touch` — at this level
- `distance_to_vwap`, `distance_to_nearest_other_level` — geometric context
- `is_post_IB` — binary, true after 10:30 ET

### Family 5 — Volatility & time-of-day

- `realized_vol_5min`, `realized_vol_30min` — std of tick returns
- `range_30min` — high-low range
- `seconds_to_market_close` — for late-day events
- `seconds_into_session`
- `day_of_week` — one-hot Mon...Fri

### Conditioning features (Architecture B only)

- `level_id` (categorical, 7 values), `direction` (bounce/breakthrough), `tp`, `sl`

## Section 3 — Model architecture

Both A and B are built on identical features, evaluated on identical folds. The honest validation framework picks the winner.

**Model class:** LightGBM gradient-boosted classifier.

### Architecture A — per-(level, direction, TP, SL) classifiers

- 56 separate models (7 levels × 2 directions × 4 TP/SL).
- Each trained on its own slice (~1.5K–3K rows).
- Output: `P(win)` for that exact trade variant.
- Features for A do NOT include `level_id`, `direction`, `tp`, `sl` (baked into the model selection).

### Architecture B — single pooled model

- One model.
- Each event expands to 8 rows (one per `(direction, TP, SL)` combo) with `level_id`, `direction`, `tp`, `sl` as features.
- Output: `P(win)`, queried 8 times per event with different conditioning.

### Trade decision (identical for A and B at inference time)

```
At event:
  for each of 8 (direction, TP, SL) variants:
    p = model.predict(features, variant)
    expected_pnl = TP * p - SL * (1 - p)         # in points
  pick variant with max expected_pnl
  if max(expected_pnl) < deploy_threshold:
    skip
  else:
    enter trade with chosen direction & TP/SL
```

`deploy_threshold` is a single scalar tuned during validation; see Section 4.

### Hyperparameters

Held conservative to avoid overfitting on small per-level samples; identical for A and B.

- `num_leaves`: 31, `max_depth`: 6, `min_data_in_leaf`: 50
- `learning_rate`: 0.05, `n_estimators`: 500 with early stopping (50 rounds patience)
- `feature_fraction`: 0.8, `bagging_fraction`: 0.8

No hyperparameter search across folds. Single config used identically.

### Persisted outputs per training run

- A: 56 `.joblib` files in `mnq_alerts/_level_models/`
- B: 1 `.joblib` file in `mnq_alerts/_level_models/`
- Both: `feature_importance.csv`, `oof_predictions.parquet`, `per_quarter_metrics.json`

### Resource estimate

~10K total events × ~25 features × 56 models for A. LightGBM with these settings: ~1–2 min total train time, ~10 ms inference per event for both variants combined.

## Section 4 — Validation methodology

### Data split

- **Dev set:** days 1 → 309 (Jan 2025 → ~Apr 8 2026). Used for walk-forward CV and architecture selection.
- **Final out-of-time test:** days 310 → 339 (last 30 trading days). Touched ONCE, by the architecture chosen on dev.

The final test is the deploy gate. Touching it during architecture selection invalidates it.

### Walk-forward folds (dev set)

Quarterly walk-forward, expanding window:

- Train Q1'25 → Test Q2'25
- Train Q1–Q2'25 → Test Q3'25
- Train Q1–Q3'25 → Test Q4'25
- Train Q1–Q4'25 → Test Q1'26
- Train Q1'25–Q1'26 → Test partial Q2'26 (up to day 309)

5 test folds.

### Embargo

- Drop the first calendar day of each test fold.
- Inside each train fold, the **last 5%** of training days (chronological) are used for LightGBM early stopping. Same drop-first-day embargo rule applied at this internal validation boundary.

### Architecture selection (A vs B)

Score each architecture across all 5 folds:

- **Top-decile lift:** mean win rate of top-10% predicted trades vs base rate. Must be positive in ≥ 4/5 folds.
- **Top-decile expected P&L per trade:** must be positive in 5/5 folds.
- **Simulated mean daily P&L** (1-position constraint + slippage modeling): for each fold, the test-quarter's mean daily P&L from the model must exceed V6's slippage-aware backtest mean daily P&L over the same calendar quarter. Must hold in ≥ 4/5 folds.

If A passes all three: A is candidate. If B passes all three: B is candidate. If both pass: pick higher mean simulated daily P&L. **If neither passes: stop. No iterating to rescue.**

### E2E gate verification (mandatory)

The test that caught v3.

- Pick 1 day not in any dev fold (e.g., the day before final test starts).
- Replay through live bot path: tick → feature computation → live `predict()` call.
- Compare to offline-pipeline predictions for the same triggers.
- **PASS:** max `|p_live − p_offline|` < 0.02, mean `|p_live − p_offline|` < 0.005.
- **FAIL:** leakage somewhere; do not deploy.

### Final out-of-time test (deploy decision)

Once architecture is selected and E2E gate passes:

- Run candidate on days 310–339 (untouched until this moment).
- **Required:** simulated mean daily P&L > V6's slippage-aware backtest mean daily P&L over the same 30-day window. (V6 was deployed 2026-05-06 — for most of this window we compare against V6's backtest, not its live record.)
- **Required:** top-decile lift > +5pp over base rate.
- **Required:** per-week P&L positive in ≥ 3 of last 4 calendar weeks (regime sanity).
- All three must pass to deploy. Any failure rejects the candidate; report findings and stop.

### Leakage protections (codified as unit tests)

- `test_prior_touch_outcome_resolution_order` — `prior_touch_outcome` for an event at time T is computed only from prior touches whose outcomes resolved before T.
- `test_no_future_ticks_in_features` — for 100 random events, masking ticks with `ts > event_ts` produces identical features.
- `test_label_leakage` — permutation test: shuffle labels in train, retrain, AUC on test should be ~0.5.

### Reporting

All metrics reported per-fold AND aggregated. If aggregate looks good but one fold is terrible, that's the headline.

## Section 5 — Deploy bar & rollout

### If final test FAILS any gate

Stop. Write findings as a project memory. Do not iterate the architecture to "rescue" a failure (the explicit anti-pattern from v4).

### If final test PASSES all gates — staged rollout

**Stage 1: Shadow mode (10 trading days minimum).**
- V6 stays live; model runs in parallel on the same tick stream.
- Logs predictions and would-have-traded P&L.
- **Shadow gate:** model's daily P&L tracks within ±20% of its backtest expectation, AND outperforms V6 cumulatively over 10 days. Drift from backtest expectation = unmodeled production effect; investigate.

**Stage 2: Canary (next 10 trading days).**
- Bot uses model for trade decisions.
- Position size at **half** of normal.
- V6 retained as monitoring shadow.
- **Canary gate:** simulated full-size P&L > V6's actual full-size P&L. Loss to V6 over 10 days at half size = roll back.

**Stage 3: Full deploy.**
- Position size returned to normal.
- V6 retained as monitoring shadow for at least the next quarter.

### Rollback triggers (any one fires → revert to V6 immediately)

- 3 consecutive losing days for the model
- Cumulative drawdown exceeds backtest's MaxDD
- E2E gate divergence > 0.05 on any production day (run nightly on the day's events)
- Any unit-test failure on the leakage protection suite

### Daily monitoring (post-deploy)

Add to existing daily report:
- Model P&L, V6 shadow P&L, top-decile realized lift, E2E divergence stat
- Alert if any rollback trigger fires

### Logging additions to bot

Per trade decision:
- Features used
- All 8 variant probabilities
- Expected P&L per variant
- Chosen variant

This enables nightly E2E re-check and post-hoc "why did the model trade this?" analysis.

## Open questions deferred to plan

- Exact `deploy_threshold` calibration (will be derived from dev-set top-decile expected P&L).
- Decimal precision for slippage modeling at entry — currently "next tick price"; may want to enforce a 0.25pt minimum offset.
- Whether to expose the model's expected P&L distribution to the daily monitoring report (full distribution vs picked variant only).
