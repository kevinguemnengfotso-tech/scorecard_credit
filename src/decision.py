"""Conversion between scorecard points and probability of default.

Project-specific note: scorecard points distribute both the intercept and the
offset across variables, and `calculate_score` adds a base point term as well.
The resulting relationship is:

    score = BASE - Factor * logit,  where BASE = 2 * Offset + Factor * intercept

Higher score means lower risk. This module centralises the conversion so the
cost-matrix decision rule can be applied directly to scorecard scores.
"""

from typing import Union

import numpy as np

from cost_matrix import optimal_pd_threshold


def scorecard_base_score(offset: float, factor: float, intercept: float) -> float:
    """Constant linking score and log-odds in this project."""
    return 2.0 * offset + factor * intercept


def score_to_pd(
    score: Union[float, np.ndarray],
    offset: float,
    factor: float,
    intercept: float,
) -> Union[float, np.ndarray]:
    """Convert scorecard points to probability of default."""
    base = scorecard_base_score(offset, factor, intercept)
    logit = (base - np.asarray(score, dtype=float)) / factor
    return 1.0 / (1.0 + np.exp(-logit))


def pd_to_score(
    pd_value: Union[float, np.ndarray],
    offset: float,
    factor: float,
    intercept: float,
) -> Union[float, np.ndarray]:
    """Convert probability of default to scorecard points."""
    base = scorecard_base_score(offset, factor, intercept)
    p = np.asarray(pd_value, dtype=float)
    logit = np.log(p / (1.0 - p))
    return base - factor * logit


def optimal_score_threshold(offset: float, factor: float, intercept: float) -> float:
    """Score threshold implied by the cost matrix (PD < 1/6)."""
    return float(pd_to_score(optimal_pd_threshold(), offset, factor, intercept))
