"""Cost-sensitive decision logic for the German Credit dataset.

Official UCI/Statlog cost matrix:
    Reject a good customer : 1 unit (lost margin)
    Accept a bad customer  : 5 units (lost principal)

The key result: accept a loan if 5 * PD < 1 * (1 - PD), i.e. PD < 1/6.
"""

from typing import Sequence, Union

import numpy as np
import pandas as pd

COST_REJECT_GOOD = 1
COST_ACCEPT_BAD = 5


def expected_cost(pd_value: float, approved: bool) -> float:
    """Expected cost of one decision given the probability of default."""
    if approved:
        return COST_ACCEPT_BAD * float(pd_value)
    return COST_REJECT_GOOD * (1.0 - float(pd_value))


def optimal_pd_threshold() -> float:
    """Break-even PD derived from the cost matrix: 1 / (5 + 1) = 1/6."""
    return COST_REJECT_GOOD / (COST_ACCEPT_BAD + COST_REJECT_GOOD)


def total_portfolio_cost(
    y_true: Sequence[int],
    pd_values: Sequence[float],
    approved: Sequence[bool],
) -> float:
    """Total cost over a portfolio. This is THE financial metric."""
    y = np.asarray(y_true)
    pd_arr = np.asarray(pd_values, dtype=float)
    dec = np.asarray(approved, dtype=bool)
    n_bad_accepted = int(np.sum((y == 1) & dec))
    n_good_rejected = int(np.sum((y == 0) & ~dec))
    return float(COST_ACCEPT_BAD * n_bad_accepted + COST_REJECT_GOOD * n_good_rejected)


def cost_by_threshold(
    y_true: Sequence[int],
    pd_values: Sequence[float],
    thresholds: Sequence[float],
) -> pd.DataFrame:
    """Cost, approval rate and confusion counts for each PD cutoff."""
    y = np.asarray(y_true)
    pds = np.asarray(pd_values, dtype=float)
    rows = []
    for t in thresholds:
        approved = pds < t
        rows.append({
            "threshold_pd": float(t),
            "n_approved": int(approved.sum()),
            "approval_rate": float(approved.mean()),
            "bad_accepted": int(np.sum((y == 1) & approved)),
            "good_rejected": int(np.sum((y == 0) & ~approved)),
            "total_cost": total_portfolio_cost(y, pds, approved),
        })
    return pd.DataFrame(rows)


def stress_cost(
    y_true: Sequence[int],
    pd_values: Sequence[float],
    approved: Sequence[bool],
    shock: float = 0.20,
) -> dict:
    """Downturn simulation: a share of accepted good clients turn bad.

    The decision policy stays fixed; we measure the additional cost caused by
    more defaults among the accepted population.
    """
    y = np.asarray(y_true)
    dec = np.asarray(approved, dtype=bool)
    y_stressed = y.copy()
    accepted_good = np.where((y == 0) & dec)[0]
    n_extra = int(round(len(accepted_good) * shock))
    if n_extra > 0:
        rng = np.random.default_rng(42)
        idx = rng.choice(accepted_good, size=n_extra, replace=False)
        y_stressed[idx] = 1
    return {
        "shock": float(shock),
        "n_extra_defaults": n_extra,
        "baseline_cost": total_portfolio_cost(y, pd_values, dec),
        "stressed_cost": total_portfolio_cost(y_stressed, pd_values, dec),
        "baseline_expected_cost": float(np.mean([
            expected_cost(p, a) for p, a in zip(pd_values, dec)
        ])),
        "stressed_expected_cost": float(np.mean([
            expected_cost(p, a) for p, a in zip(pd_values, dec)
        ])),
    }
