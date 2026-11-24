"""Tests for src/cost_matrix.py."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cost_matrix import (
    COST_ACCEPT_BAD,
    COST_REJECT_GOOD,
    cost_by_threshold,
    expected_cost,
    optimal_pd_threshold,
    stress_cost,
    total_portfolio_cost,
)


def test_cost_constants():
    assert COST_REJECT_GOOD == 1
    assert COST_ACCEPT_BAD == 5


def test_expected_cost_accept():
    assert expected_cost(0.2, True) == pytest.approx(1.0)
    assert expected_cost(0.0, True) == pytest.approx(0.0)


def test_expected_cost_reject():
    assert expected_cost(0.2, False) == pytest.approx(0.8)
    assert expected_cost(1.0, False) == pytest.approx(0.0)


def test_optimal_threshold_is_one_sixth():
    assert optimal_pd_threshold() == pytest.approx(1.0 / 6.0)


def test_optimal_threshold_break_even():
    t = optimal_pd_threshold()
    assert expected_cost(t, True) == pytest.approx(expected_cost(t, False))


def test_total_cost():
    y = [0, 0, 1, 1]
    pds = [0.1, 0.3, 0.7, 0.2]
    approved = [True, False, True, False]
    # bad accepted: 1 -> 5 ; good rejected: 1 -> 1 ; total 6
    assert total_portfolio_cost(y, pds, approved) == 6.0


def test_cost_by_threshold_shape():
    y = [0, 0, 1, 1]
    pds = [0.1, 0.3, 0.7, 0.2]
    df = cost_by_threshold(y, pds, [0.1, 0.2, 0.5])
    assert len(df) == 3
    assert "total_cost" in df.columns


def test_stress_cost_increases():
    y = [0, 0, 1, 1]
    pds = [0.1, 0.3, 0.7, 0.2]
    approved = [True, True, False, False]
    r = stress_cost(y, pds, approved, shock=0.6)
    assert r["stressed_cost"] > r["baseline_cost"]
    assert r["n_extra_defaults"] > 0
