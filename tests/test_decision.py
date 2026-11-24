"""Tests for src/decision.py."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from decision import (
    optimal_score_threshold,
    pd_to_score,
    score_to_pd,
    scorecard_base_score,
)

OFFSET, FACTOR, INTERCEPT = 513.5614381022527, 28.85390081777927, -0.8474334901765069


def test_base_score():
    expected = 2 * OFFSET + FACTOR * INTERCEPT
    assert scorecard_base_score(OFFSET, FACTOR, INTERCEPT) == pytest.approx(expected)


def test_round_trip_scalar():
    pd_value = 0.2
    score = pd_to_score(pd_value, OFFSET, FACTOR, INTERCEPT)
    back = score_to_pd(score, OFFSET, FACTOR, INTERCEPT)
    assert float(back) == pytest.approx(pd_value)


def test_round_trip_array():
    pds = np.array([0.05, 0.2, 0.5])
    scores = pd_to_score(pds, OFFSET, FACTOR, INTERCEPT)
    back = score_to_pd(scores, OFFSET, FACTOR, INTERCEPT)
    np.testing.assert_array_almost_equal(back, pds)


def test_score_is_decreasing_in_pd():
    s1 = pd_to_score(0.1, OFFSET, FACTOR, INTERCEPT)
    s2 = pd_to_score(0.3, OFFSET, FACTOR, INTERCEPT)
    assert s1 > s2


def test_optimal_threshold_above_1000():
    threshold = optimal_score_threshold(OFFSET, FACTOR, INTERCEPT)
    assert 1000 < threshold < 1100
