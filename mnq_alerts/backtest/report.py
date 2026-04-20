"""Formatting and summary statistics."""

from __future__ import annotations

from .simulate import TradeRecord

STARTING_BALANCE = 10_000.0


def fmt(trades: list[TradeRecord], n_days: int, label: str = "") -> str:
    """One-line summary of trade results."""
    if not trades:
        return f"  {label:>50s}  no trades"
    w = sum(1 for t in trades if t.outcome == "win")
    l = sum(1 for t in trades if t.outcome == "loss")
    o = len(trades) - w - l
    d = w + l
    wr = w / d * 100 if d else 0
    pnl = sum(t.pnl_usd for t in trades)
    ppd = pnl / n_days
    eq = STARTING_BALANCE
    peak = eq
    dd = 0.0
    for t in trades:
        eq += t.pnl_usd
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    aw = sum(t.pnl_usd for t in trades if t.outcome == "win") / w if w else 0
    al = sum(t.pnl_usd for t in trades if t.outcome == "loss") / l if l else 0
    return (
        f"  {label:>50s}  {len(trades):>4} ({len(trades)/n_days:.1f}/d) "
        f"{w}W/{l}L/{o}O {wr:>5.1f}%  "
        f"W:{aw:>+6.1f} L:{al:>+6.1f}  "
        f"PnL {pnl:>+8,.0f} ({ppd:>+5.1f}/d)  DD {dd:>5,.0f}"
    )


def per_level_breakdown(
    trades: list[TradeRecord], n_days: int,
) -> None:
    """Print per-level summary."""
    levels = sorted(set(t.level for t in trades))
    for lv in levels:
        lv_trades = [t for t in trades if t.level == lv]
        print(fmt(lv_trades, n_days, lv))
