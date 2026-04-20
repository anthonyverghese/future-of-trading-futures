# Backtest Archive

All previous backtest files, preserved for reference. The modular `backtest/` library supersedes these.

| File | Date | What it tested | Key result | Why superseded |
|------|------|---------------|------------|----------------|
| bot_backtest.py | pre-2026-04 | Baseline T/S sweep, entry/exit threshold sweep | Best: T12/S25 EV/day | Superseded by walk-forward approach |
| bot_risk_backtest.py | pre-2026-04 | Risk-aware sim (daily loss, consec loss, 1-pos) | $150/3 risk gates validated | Core evaluate_bot_trade still used |
| bot_improvement_backtest.py | pre-2026-04 | Walk-forward T/S + risk selection | T12/S25 most selected | IB_END was wrong (10:30 not 10:31) |
| bot_trend_backtest.py | pre-2026-04 | Trend filter + time-of-day scoring | Score >= 1 filter helped | Weights were too simple |
| bot_score_optimizer.py | pre-2026-04 | Factor analysis for human scoring weights | Derived Weights() class | Still used as reference for human weights |
| bot_optimal_wf.py | pre-2026-04 | Walk-forward optimization | Various T/S configs | Superseded by bot_full_backtest |
| bot_full_backtest.py | 2026-04-17 | Full human scoring for bot entries | +$7.3/day OOS | Used human weights on bot entries (wrong approach) |
| bot_pct_backtest.py | 2026-04-17 | %-based T/S, bot-specific weight training | +$16.7/day in-sample, weaker OOS | %-based thresholds didn't help enough |
| bot_bounce_backtest.py | 2026-04-17 | Bounce analysis: MFE/MAE, tight stops, trailing | Tight stops don't work (MAE too high) | Key finding: median winner 27s, loser 78s |
| bot_fast_backtest.py | 2026-04-18 | Time-based exit (scratch losers early) | Time cuts didn't help OOS | Scratches ate gains from higher WR |
| bot_frequency_backtest.py | 2026-04-18 | Entry/exit threshold sweep for frequency | Killed (too slow) | Approach was wrong |
| bot_edge_backtest.py | 2026-04-18 | Line-specific factors (approach speed, tick density) | Approach fast +4.5pp, fresh test +3.5pp | Factors helped but scoring too weak overall |
| bot_combined_backtest.py | 2026-04-18 | Human score as factor + bot extras + per-level T/S | +$10.9/day OOS per-level/S20 | Bug: parallel zone matching lost entries |
| bot_trader_backtest.py | 2026-04-18 | Human pre-filter → bot at line | Not completed | Filtering human entries can only match, not beat |
| bot_v2_backtest.py | 2026-04-19 | Correct entry matching (human alert → line touch) | Not completed | Still just filtering human entries |
| bot_v3_backtest.py | 2026-04-19 | Bot-trained scoring, all entries | +$8.8/day OOS | Missing streak + wrong factor buckets |
| bot_v4_backtest.py | 2026-04-19 | All human factors + streak + bot extras | +$12.1/day OOS | Fixed exit threshold limited volume |
| bot_v5_backtest.py | 2026-04-19 | Zone resets on trade close | **+$21.4/day OOS** | **Current best — deployed to live** |

## Key lessons learned

1. Zone exit threshold was the biggest lever (fixed 20pt → trade-close reset doubled P&L)
2. Missing scoring factors (especially streak +3/-2) severely weakened bot scoring
3. Factor buckets must match the human scoring exactly (tick sweet spot 1750-2000, session move point-based)
4. The bot should always outperform the human — if it doesn't, the backtest has a bug
5. Per-level targets matter: VWAP=T6, FIB_HI=T5, IBL=T10, IBH=T14, FIB_LO=T8
6. The human $26/day benchmark has no risk limits — drops to ~$14-21/day with 1-position constraint
