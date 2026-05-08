"""Strategy simulator: pick best variant per event, enforce 1-position constraint."""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

POINTS_PER_DOLLAR = 2.0


def simulate_strategy(
    events: list[dict],
    preds: dict,
    threshold: float,
) -> dict:
    """Run the full strategy simulation.

    `events`: list of event dicts. Each event_ts may have up to 8 rows
    (one per (direction, tp, sl) variant) all sharing the same event_ts.
    `preds`: event_ts -> dict of (direction, tp, sl) -> P(win).
    `threshold`: min expected_pnl (points) to take a trade.
    """
    groups: dict = defaultdict(list)
    for ev in events:
        groups[ev["event_ts"]].append(ev)

    in_position_until = pd.Timestamp("1900-01-01", tz="UTC")
    trades = 0
    wins = 0
    losses = 0
    total_points = 0.0
    chosen = []
    daily_points: dict = {}

    for event_ts in sorted(groups.keys()):
        if event_ts < in_position_until:
            continue
        ev_preds = preds.get(event_ts, {})
        if not ev_preds:
            continue
        group_rows = groups[event_ts]
        best = None
        best_ev = -1e9
        for (direction, tp, sl), p in ev_preds.items():
            expected = tp * p - sl * (1 - p)
            if expected > best_ev:
                best_ev = expected
                best = (direction, tp, sl)
        if best is None or best_ev < threshold:
            continue
        chosen_row = next(
            (r for r in group_rows
             if r["direction"] == best[0] and r["tp"] == best[1] and r["sl"] == best[2]),
            None,
        )
        if chosen_row is None:
            continue
        label = chosen_row["label"]
        ttr = chosen_row.get("time_to_resolution_sec")
        if ttr is None or pd.isna(ttr):
            ttr = 15 * 60
        in_position_until = event_ts + pd.Timedelta(seconds=ttr)
        trades += 1
        if label == 1:
            wins += 1
            total_points += best[1]
        else:
            losses += 1
            total_points -= best[2]
        chosen.append(best)
        day = event_ts.date()
        daily_points[day] = daily_points.get(day, 0.0) + (best[1] if label == 1 else -best[2])

    return {
        "trades": trades, "wins": wins, "losses": losses,
        "total_points": total_points,
        "total_dollars": total_points * POINTS_PER_DOLLAR,
        "chosen_variants": chosen,
        "daily_pnl_dollars": {d: v * POINTS_PER_DOLLAR for d, v in daily_points.items()},
    }
