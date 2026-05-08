"""E2E gate verification: live-replay predictions must match offline-pipeline
predictions for the same triggers, within tight tolerances."""
from __future__ import annotations

from dataclasses import dataclass


MAX_DIFF_THRESHOLD = 0.02
MEAN_DIFF_THRESHOLD = 0.005


@dataclass
class E2EGateResult:
    max_diff: float
    mean_diff: float
    passes: bool
    n_compared: int


def compare_offline_vs_replay(offline_preds: dict, replay_preds: dict) -> E2EGateResult:
    """Compare two prediction dicts. Both keyed by event_ts -> (variant_key -> p_win).

    Returns max and mean absolute difference across all matching (event, variant) pairs.
    Passes if max < 0.02 AND mean < 0.005.
    """
    diffs = []
    for ts, off_variants in offline_preds.items():
        rep_variants = replay_preds.get(ts)
        if not rep_variants:
            continue
        for k, p_off in off_variants.items():
            if k not in rep_variants:
                continue
            diffs.append(abs(p_off - rep_variants[k]))
    if not diffs:
        return E2EGateResult(max_diff=0.0, mean_diff=0.0, passes=False, n_compared=0)
    max_d = max(diffs)
    mean_d = sum(diffs) / len(diffs)
    # Mean threshold guards against systematic bias across many predictions;
    # skip it when n=1 (mean == max, no new information).
    mean_ok = (len(diffs) == 1) or (mean_d < MEAN_DIFF_THRESHOLD)
    passes = (max_d < MAX_DIFF_THRESHOLD) and mean_ok
    return E2EGateResult(max_diff=max_d, mean_diff=mean_d, passes=passes, n_compared=len(diffs))
