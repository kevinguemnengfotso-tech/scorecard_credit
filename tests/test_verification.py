"""Tests for src/verification.py.

Run with: pytest tests/test_verification.py -v
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from verification import (
    compute_metrics,
    verify_metrics,
    stress_test_noise,
    stress_test_shift,
    stress_test_target_increase,
    score_to_proba,
    check_calibration,
    check_rank_ordering,
    check_leakage,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_scores():
    """Synthetic score data for testing."""
    np.random.seed(42)
    n = 200
    scores = np.random.normal(1000, 50, n)
    defaults = (scores < 980).astype(int)
    # Add some noise
    flip = np.random.choice(n, size=20, replace=False)
    defaults[flip] = 1 - defaults[flip]
    df = pd.DataFrame({"score": scores, "default": defaults})
    return df


@pytest.fixture
def train_test_scores(sample_scores):
    """Split into train and test."""
    train = sample_scores.iloc[:140].copy().reset_index(drop=True)
    test = sample_scores.iloc[140:].copy().reset_index(drop=True)
    return train, test


# ---------------------------------------------------------------------------
# Tests: compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_returns_dict(self, sample_scores):
        m = compute_metrics(sample_scores)
        assert isinstance(m, dict)

    def test_has_required_keys(self, sample_scores):
        m = compute_metrics(sample_scores)
        for key in ["AUC", "Gini", "KS", "n_obs", "default_rate"]:
            assert key in m

    def test_auc_between_0_and_1(self, sample_scores):
        m = compute_metrics(sample_scores)
        assert 0 <= m["AUC"] <= 1

    def test_gini_formula(self, sample_scores):
        m = compute_metrics(sample_scores)
        # Rounded to 4 decimals -> tolerance 1e-3
        assert abs(m["Gini"] - (2 * m["AUC"] - 1)) < 1e-3

    def test_n_obs_matches(self, sample_scores):
        m = compute_metrics(sample_scores)
        assert m["n_obs"] == len(sample_scores)


# ---------------------------------------------------------------------------
# Tests: verify_metrics
# ---------------------------------------------------------------------------

class TestVerifyMetrics:
    def test_returns_dict(self, train_test_scores):
        train, test = train_test_scores
        result = verify_metrics(train, test)
        assert isinstance(result, dict)
        assert "train" in result
        assert "test" in result
        assert "checks" in result

    def test_checks_are_booleans(self, train_test_scores):
        train, test = train_test_scores
        result = verify_metrics(train, test)
        for v in result["checks"].values():
            assert isinstance(v, (bool, np.bool_))


# ---------------------------------------------------------------------------
# Tests: stress_test_noise
# ---------------------------------------------------------------------------

class TestStressTestNoise:
    def test_returns_dict(self, sample_scores):
        result = stress_test_noise(sample_scores, flip_rate=0.10)
        assert isinstance(result, dict)

    def test_has_required_keys(self, sample_scores):
        result = stress_test_noise(sample_scores, flip_rate=0.10)
        for key in ["auc_original", "auc_stress", "flip_rate", "n_flipped"]:
            assert key in result

    def test_flip_rate_respected(self, sample_scores):
        result = stress_test_noise(sample_scores, flip_rate=0.10, seed=42)
        expected = int(len(sample_scores) * 0.10)
        assert abs(result["n_flipped"] - expected) <= 5

    def test_reproducible(self, sample_scores):
        r1 = stress_test_noise(sample_scores, flip_rate=0.10, seed=42)
        r2 = stress_test_noise(sample_scores, flip_rate=0.10, seed=42)
        assert r1["auc_stress"] == r2["auc_stress"]


# ---------------------------------------------------------------------------
# Tests: stress_test_target_increase
# ---------------------------------------------------------------------------

class TestStressTestTargetIncrease:
    def test_returns_dict(self, sample_scores):
        result = stress_test_target_increase(sample_scores, increase_pct=0.50)
        assert isinstance(result, dict)

    def test_increase_pct_approximately(self, sample_scores):
        result = stress_test_target_increase(sample_scores, increase_pct=0.50)
        assert result["increase_pct"] > 0

    def test_original_rate_unchanged(self, sample_scores):
        original = sample_scores["default"].mean()
        result = stress_test_target_increase(sample_scores, increase_pct=0.50)
        assert abs(result["original_rate"] - original) < 1e-10


# ---------------------------------------------------------------------------
# Tests: score_to_proba
# ---------------------------------------------------------------------------

class TestScoreToProba:
    def test_returns_array(self):
        scores = np.array([900, 1000, 1100])
        proba = score_to_proba(scores, Factor=28.85, Offset=513.0, intercept=-0.85)
        assert len(proba) == 3

    def test_proba_between_0_and_1(self):
        scores = np.array([900, 1000, 1100])
        proba = score_to_proba(scores, Factor=28.85, Offset=513.0, intercept=-0.85)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_higher_score_lower_proba(self):
        scores = np.array([900, 1100])
        proba = score_to_proba(scores, Factor=28.85, Offset=513.0, intercept=-0.85)
        assert proba[0] > proba[1]  # lower score = higher default prob


# ---------------------------------------------------------------------------
# Tests: check_calibration
# ---------------------------------------------------------------------------

class TestCheckCalibration:
    def test_returns_dict(self, sample_scores):
        result = check_calibration(sample_scores, Factor=28.85, Offset=513.0, intercept=-0.85)
        assert isinstance(result, dict)

    def test_has_brier_score(self, sample_scores):
        result = check_calibration(sample_scores, Factor=28.85, Offset=513.0, intercept=-0.85)
        assert "brier_score" in result
        assert 0 <= result["brier_score"] <= 1


# ---------------------------------------------------------------------------
# Tests: check_rank_ordering
# ---------------------------------------------------------------------------

class TestCheckRankOrdering:
    def test_returns_dict(self, sample_scores):
        result = check_rank_ordering(sample_scores)
        assert isinstance(result, dict)

    def test_has_required_keys(self, sample_scores):
        result = check_rank_ordering(sample_scores)
        for key in ["is_monotonic", "n_bins", "first_bin_rate", "last_bin_rate"]:
            assert key in result

    def test_first_bin_worse_than_last(self, sample_scores):
        result = check_rank_ordering(sample_scores)
        assert result["first_bin_rate"] >= result["last_bin_rate"]


# ---------------------------------------------------------------------------
# Tests: check_leakage
# ---------------------------------------------------------------------------

class TestCheckLeakage:
    def test_returns_dict(self, train_test_scores):
        train, test = train_test_scores
        result = check_leakage(train, test)
        assert isinstance(result, dict)

    def test_has_auc_gap(self, train_test_scores):
        train, test = train_test_scores
        result = check_leakage(train, test)
        assert "auc_gap" in result
        assert "auc_gap_ok" in result

    def test_with_iv_df(self, train_test_scores):
        train, test = train_test_scores
        iv_df = pd.DataFrame({"IV": [0.1, 0.3, 0.05]})
        result = check_leakage(train, test, iv_df=iv_df)
        assert "max_iv" in result
        assert "iv_ok" in result
