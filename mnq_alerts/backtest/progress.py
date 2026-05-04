"""Simple progress tracker for backtest runs.

Writes progress to a file that can be read while the process runs,
bypassing stdout buffering from conda run.

Usage:
    from mnq_alerts.backtest.progress import Progress

    prog = Progress("my_backtest", total_variants=10)
    prog.update("Variant 1", trades=15, pnl=42.3)
    prog.update("Variant 2", trades=12, pnl=-10.5)
    prog.done()

    # From another terminal:
    cat /tmp/backtest_progress_my_backtest.txt
"""

import os
import time

_PROGRESS_DIR = "/tmp"


class Progress:
    def __init__(self, name: str, total_variants: int = 0):
        self._path = os.path.join(_PROGRESS_DIR, f"backtest_progress_{name}.txt")
        self._total = total_variants
        self._completed = 0
        self._start = time.time()
        self._write(f"Started: {total_variants} variants\n")

    def update(self, variant_name: str, **stats):
        self._completed += 1
        elapsed = time.time() - self._start
        stats_str = ", ".join(f"{k}={v}" for k, v in stats.items())
        remaining = (elapsed / self._completed) * (self._total - self._completed) if self._completed > 0 else 0
        line = (
            f"[{self._completed}/{self._total}] {variant_name} "
            f"({stats_str}) "
            f"elapsed={elapsed/60:.1f}min, ~{remaining/60:.0f}min remaining\n"
        )
        self._write(line)

    def done(self):
        elapsed = time.time() - self._start
        self._write(f"Done in {elapsed/60:.1f}min\n")

    def _write(self, line: str):
        with open(self._path, "a") as f:
            f.write(line)
            f.flush()
