# V_MULTI is DEPRECATED as of 2026-05-14

**Do not trust the backtests in this directory.** Read `meta.json` -> `untrustworthy_backtest_note` for the full explanation.

## TL;DR

The V_MULTI filter (and every backtest experiment that uses `_events_augmented_v4.parquet` -> `human_score`) reconstructs the human algorithm's composite_score by walking back through stored ticks. We confirmed live on 2026-05-14 that this reconstruction diverges from the actual live human algorithm by 3-9 points on every observed event (live=5 -> reconstructed=-1; live=8 -> reconstructed=4; etc.).

That means:

- The `+$26.73/day` out-of-time improvement is the gain over the reconstruction-gated baseline, not over what a live-alert-gated bot would do.
- The 3 LightGBM models were trained on a different event universe (reconstruction-gated) than the live alerter selects.
- All four follow-up experiments (vote aggregation, speed filter, speed x regime, bidirectional bot) ran on the same broken universe and may produce different results when re-evaluated against live alerts.

## What we're doing instead

The bot is being switched to a live-alert gate: when `alert_manager.check_and_notify` fires an alert on a bot-whitelisted level, the bot submits a bracket trade. The 1pt-zone trigger and the V_MULTI filter are turned off in this mode.

## What to do if revisiting V_MULTI

1. Pull historical alerts from the `alerts` table in `alerts_log.db`.
2. Match each alert to the bot's 1pt-zone-entry event (or the alert moment directly, depending on the design).
3. Compute features at the matched moment.
4. Retrain LGBMs on the live-alert-matched universe with the same (8/20, 8/25, 10/20) labels.
5. Walk-forward + out-of-time evaluate against unfiltered V6.

Skipping the retrain and just swapping the gate is NOT a valid shortcut.
