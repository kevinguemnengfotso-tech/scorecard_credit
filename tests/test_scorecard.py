"""Tests for src/scorecard.py — Phase 8 scorecard scaling functions.

Run with: pytest tests/test_scorecard.py -v
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scorecard import (
    compute_scaling_params,
    extract_coefficients,
    build_scorecard,
    calculate_score,
    score_distribution,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scaling_params():
    """Standard scaling parameters for testing."""
    Factor, Offset, params = compute_scaling_params(PDO=20, base_score=600, base_odds=20)
    return Factor, Offset, params


@pytest.fixture
def mock_model():
    """A minimal sklearn logistic model mock."""
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    X = np.array([[0.5, -0.3], [1.2, 0.8], [-0.5, 0.1], [0.0, -0.5]])
    y = np.array([0, 1, 0, 1])
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(X, y)
    return model


@pytest.fixture
def mock_bp():
    """A minimal optbinning BinningProcess mock."""
    from optbinning import BinningProcess
    import numpy as np
    X = pd.DataFrame({"feat_a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      "feat_b": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
    y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    bp = BinningProcess(variable_names=["feat_a", "feat_b"])
    bp.fit(X, y)
    return bp


@pytest.fixture
def mock_scorecard():
    """A minimal scorecard DataFrame."""
    return pd.DataFrame({
        "variable": ["feat_a", "feat_a", "feat_b", "feat_b"],
        "bin": ["bin1", "bin2", "bin1", "bin2"],
        "woe": [0.5, -0.3, 0.8, -0.2],
        "count": [5, 5, 5, 5],
        "event_rate": [0.2, 0.6, 0.1, 0.7],
        "points": [70.0, 40.0, 80.0, 35.0],
    })


# ---------------------------------------------------------------------------
# Tests: compute_scaling_params
# ---------------------------------------------------------------------------

class TestComputeScalingParams:
    def test_factor_positive(self, scaling_params):
        Factor, Offset, params = scaling_params
        assert Factor > 0

    def test_factor_formula(self):
        Factor, Offset, params = compute_scaling_params(PDO=20, base_score=600, base_odds=20)
        expected_factor = 20 / np.log(2)
        assert abs(Factor - expected_factor) < 1e-10

    def test_offset_formula(self):
        Factor, Offset, params = compute_scaling_params(PDO=20, base_score=600, base_odds=20)
        expected_offset = 600 - Factor * np.log(20)
        assert abs(Offset - expected_offset) < 1e-10

    def test_params_dict_keys(self, scaling_params):
        Factor, Offset, params = scaling_params
        assert set(params.keys()) == {"PDO", "BASE_SCORE", "BASE_ODDS", "Factor", "Offset"}

    def test_pdo_50(self):
        Factor, Offset, params = compute_scaling_params(PDO=50, base_score=600, base_odds=20)
        assert Factor > compute_scaling_params(PDO=20, base_score=600, base_odds=20)[0]


# ---------------------------------------------------------------------------
# Tests: extract_coefficients
# ---------------------------------------------------------------------------

class TestExtractCoefficients:
    def test_returns_dict_and_float(self, mock_model):
        features = ["feat1", "feat2"]
        coef_dict, intercept = extract_coefficients(mock_model, features)
        assert isinstance(coef_dict, dict)
        assert isinstance(intercept, float)

    def test_dict_has_all_features(self, mock_model):
        features = ["feat1", "feat2"]
        coef_dict, intercept = extract_coefficients(mock_model, features)
        assert set(coef_dict.keys()) == set(features)

    def test_values_are_floats(self, mock_model):
        features = ["feat1", "feat2"]
        coef_dict, intercept = extract_coefficients(mock_model, features)
        for v in coef_dict.values():
            assert isinstance(v, (float, np.floating))


# ---------------------------------------------------------------------------
# Tests: build_scorecard
# ---------------------------------------------------------------------------

class TestBuildScorecard:
    def test_returns_dataframe(self, mock_bp, mock_model):
        features = ["feat_a", "feat_b"]
        coef_dict, intercept = extract_coefficients(mock_model, features)
        Factor, Offset, _ = compute_scaling_params()
        sc = build_scorecard(mock_bp, features, coef_dict, Factor, Offset, intercept)
        assert isinstance(sc, pd.DataFrame)

    def test_has_required_columns(self, mock_bp, mock_model):
        features = ["feat_a", "feat_b"]
        coef_dict, intercept = extract_coefficients(mock_model, features)
        Factor, Offset, _ = compute_scaling_params()
        sc = build_scorecard(mock_bp, features, coef_dict, Factor, Offset, intercept)
        for col in ["variable", "bin", "woe", "count", "event_rate", "points"]:
            assert col in sc.columns

    def test_no_total_rows(self, mock_bp, mock_model):
        features = ["feat_a", "feat_b"]
        coef_dict, intercept = extract_coefficients(mock_model, features)
        Factor, Offset, _ = compute_scaling_params()
        sc = build_scorecard(mock_bp, features, coef_dict, Factor, Offset, intercept)
        assert not (sc["bin"].str.lower() == "total").any()

    def test_no_zero_count_rows(self, mock_bp, mock_model):
        features = ["feat_a", "feat_b"]
        coef_dict, intercept = extract_coefficients(mock_model, features)
        Factor, Offset, _ = compute_scaling_params()
        sc = build_scorecard(mock_bp, features, coef_dict, Factor, Offset, intercept)
        assert (sc["count"] > 0).all()

    def test_woe_rounded_to_4(self, mock_bp, mock_model):
        features = ["feat_a", "feat_b"]
        coef_dict, intercept = extract_coefficients(mock_model, features)
        Factor, Offset, _ = compute_scaling_params()
        sc = build_scorecard(mock_bp, features, coef_dict, Factor, Offset, intercept)
        for w in sc["woe"]:
            assert round(w, 4) == w

    def test_points_are_finite(self, mock_bp, mock_model):
        features = ["feat_a", "feat_b"]
        coef_dict, intercept = extract_coefficients(mock_model, features)
        Factor, Offset, _ = compute_scaling_params()
        sc = build_scorecard(mock_bp, features, coef_dict, Factor, Offset, intercept)
        assert np.isfinite(sc["points"]).all()


# ---------------------------------------------------------------------------
# Tests: calculate_score
# ---------------------------------------------------------------------------

class TestCalculateScore:
    def test_returns_series(self, mock_scorecard):
        df_woe = pd.DataFrame({
            "feat_a": [0.5, -0.3, 0.5],
            "feat_b": [0.8, -0.2, 0.8],
        }, index=[0, 1, 2])
        scores = calculate_score(df_woe, ["feat_a", "feat_b"], mock_scorecard,
                                 Factor=28.85, Offset=500, intercept=-0.5)
        assert isinstance(scores, pd.Series)

    def test_no_nan(self, mock_scorecard):
        df_woe = pd.DataFrame({
            "feat_a": [0.5, -0.3],
            "feat_b": [0.8, -0.2],
        }, index=[0, 1])
        scores = calculate_score(df_woe, ["feat_a", "feat_b"], mock_scorecard,
                                 Factor=28.85, Offset=500, intercept=-0.5)
        assert not scores.isna().any()

    def test_length_matches_input(self, mock_scorecard):
        df_woe = pd.DataFrame({
            "feat_a": [0.5, -0.3, 0.5, -0.3],
            "feat_b": [0.8, -0.2, 0.8, -0.2],
        }, index=[0, 1, 2, 3])
        scores = calculate_score(df_woe, ["feat_a", "feat_b"], mock_scorecard,
                                 Factor=28.85, Offset=500, intercept=-0.5)
        assert len(scores) == 4

    def test_rounding_handles_float_precision(self, mock_scorecard):
        # WoE with extra decimals should still match after rounding
        df_woe = pd.DataFrame({
            "feat_a": [0.50001, -0.29999],
            "feat_b": [0.80001, -0.19999],
        }, index=[0, 1])
        scores = calculate_score(df_woe, ["feat_a", "feat_b"], mock_scorecard,
                                 Factor=28.85, Offset=500, intercept=-0.5)
        assert not scores.isna().any()


# ---------------------------------------------------------------------------
# Tests: score_distribution
# ---------------------------------------------------------------------------

class TestScoreDistribution:
    def test_runs_without_error(self, capsys):
        scores = pd.Series([100, 200, 300, 400, 500])
        score_distribution(scores, label="test")
        captured = capsys.readouterr()
        assert "Score distribution" in captured.out

"""Tests for src/scorecard.py — Phase 8 scorecard scaling functions.

Run with: pytest tests/test_scorecard.py -v
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scorecard import (
    compute_scaling_params,
    extract_coefficients,
    build_scorecard,
    calculate_score,
    score_distribution,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scaling_params():
    """Standard scaling parameters for testing."""
    Factor, Offset, params = compute_scaling_params(PDO=20, base_score=600, base_odds=20)
    return Factor, Offset, params


@pytest.fixture
def mock_model():
    """A minimal sklearn logistic model mock."""
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    X = np.array([[0.5, -0.3], [1.2, 0.8], [-0.5, 0.1], [0.0, -0.5]])
    y = np.array([0, 1, 0, 1])
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(X, y)
    return model


@pytest.fixture
def mock_bp():
    """A minimal optbinning BinningProcess mock."""
    from optbinning import BinningProcess
    import numpy as np
    X = pd.DataFrame({"feat_a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      "feat_b": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
    y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    bp = BinningProcess(variable_names=["feat_a", "feat_b"])
    bp.fit(X, y)
    return bp


@pytest.fixture
def mock_scorecard():
    """A minimal scorecard DataFrame."""
    return pd.DataFrame({
        "variable": ["feat_a", "feat_a", "feat_b", "feat_b"],
        "bin": ["bin1", "bin2", "bin1", "bin2"],
        "woe": [0.5, -0.3, 0.8, -0.2],
        "count": [5, 5, 5, 5],
        "event_rate": [0.2, 0.6, 0.1, 0.7],
        "points": [70.0, 40.0, 80.0, 35.0],
    })


# ---------------------------------------------------------------------------
# Tests: compute_scaling_params
# ---------------------------------------------------------------------------

class TestComputeScalingParams:
    def test_factor_positive(self, scaling_params):
        Factor, Offset, params = scaling_params
        assert Factor > 0

    def test_factor_formula(self):
        Factor, Offset, params = compute_scaling_params(PDO=20, base_score=600, base_odds=20)
        expected_factor = 20 / np.log(2)
        assert abs(Factor - expected_factor) < 1e-10

    def test_offset_formula(self):
        Factor, Offset, params = compute_scaling_params(PDO=20, base_score=600, base_odds=20)
        expected_offset = 600 - Factor * np.log(20)
        assert abs(Offset - expected_offset) < 1e-10

    def test_params_dict_keys(self, scaling_params):
        Factor, Offset, params = scaling_params
        assert set(params.keys()) == {"PDO", "BASE_SCORE", "BASE_ODDS", "Factor", "Offset"}

    def test_pdo_50(self):
        Factor_50, _, _ = compute_scaling_params(PDO=50, base_score=600, base_odds=20)
        Factor_20, _, _ = compute_scaling_params(PDO=20, base_score=600, base_odds=20)
        assert Factor_50 > Factor_20


# ---------------------------------------------------------------------------
# Tests: extract_coefficients
# ---------------------------------------------------------------------------

class TestExtractCoefficients:
    def test_returns_dict_and_float(self, mock_model):
        features = ["feat1", "feat2"]
        coef_dict, intercept = extract_coefficients(mock_model, features)
        assert isinstance(coef_dict, dict)
        assert isinstance(intercept, float)

    def test_dict_has_all_features(self, mock_model):
        features = ["feat1", "feat2"]
        coef_dict, intercept = extract_coefficients(mock_model, features)
        assert set(coef_dict.keys()) == set(features)

    def test_values_are_floats(self, mock_model):
        features = ["feat1", "feat2"]
        coef_dict, intercept = extract_coefficients(mock_model, features)
        for v in coef_dict.values():
            assert isinstance(v, (float, np.floating))


# ---------------------------------------------------------------------------
# Tests: build_scorecard
# ---------------------------------------------------------------------------

class TestBuildScorecard:
    def test_returns_dataframe(self, mock_bp, mock_model):
        features = ["feat_a", "feat_b"]
        coef_dict, intercept = extract_coefficients(mock_model, features)
        Factor, Offset, _ = compute_scaling_params()
        sc = build_scorecard(mock_bp, features, coef_dict, Factor, Offset, intercept)
        assert isinstance(sc, pd.DataFrame)

    def test_has_required_columns(self, mock_bp, mock_model):
        features = ["feat_a", "feat_b"]
        coef_dict, intercept = extract_coefficients(mock_model, features)
        Factor, Offset, _ = compute_scaling_params()
        sc = build_scorecard(mock_bp, features, coef_dict, Factor, Offset, intercept)
        for col in ["variable", "bin", "woe", "count", "event_rate", "points"]:
            assert col in sc.columns

    def test_no_total_rows(self, mock_bp, mock_model):
        features = ["feat_a", "feat_b"]
        coef_dict, intercept = extract_coefficients(mock_model, features)
        Factor, Offset, _ = compute_scaling_params()
        sc = build_scorecard(mock_bp, features, coef_dict, Factor, Offset, intercept)
        assert not (sc["bin"].str.lower() == "total").any()

    def test_no_zero_count_rows(self, mock_bp, mock_model):
        features = ["feat_a", "feat_b"]
        coef_dict, intercept = extract_coefficients(mock_model, features)
        Factor, Offset, _ = compute_scaling_params()
        sc = build_scorecard(mock_bp, features, coef_dict, Factor, Offset, intercept)
        assert (sc["count"] > 0).all()

    def test_woe_rounded_to_4(self, mock_bp, mock_model):
        features = ["feat_a", "feat_b"]
        coef_dict, intercept = extract_coefficients(mock_model, features)
        Factor, Offset, _ = compute_scaling_params()
        sc = build_scorecard(mock_bp, features, coef_dict, Factor, Offset, intercept)
        for w in sc["woe"]:
            assert round(w, 4) == w

    def test_points_are_finite(self, mock_bp, mock_model):
        features = ["feat_a", "feat_b"]
        coef_dict, intercept = extract_coefficients(mock_model, features)
        Factor, Offset, _ = compute_scaling_params()
        sc = build_scorecard(mock_bp, features, coef_dict, Factor, Offset, intercept)
        assert np.isfinite(sc["points"]).all()


# ---------------------------------------------------------------------------
# Tests: calculate_score
# ---------------------------------------------------------------------------

class TestCalculateScore:
    def test_returns_series(self, mock_scorecard):
        df_woe = pd.DataFrame({
            "feat_a": [0.5, -0.3, 0.5],
            "feat_b": [0.8, -0.2, 0.8],
        }, index=[0, 1, 2])
        scores = calculate_score(df_woe, ["feat_a", "feat_b"], mock_scorecard,
                                 factor=28.85, offset=500, intercept=-0.5)
        assert isinstance(scores, pd.Series)

    def test_no_nan(self, mock_scorecard):
        df_woe = pd.DataFrame({
            "feat_a": [0.5, -0.3],
            "feat_b": [0.8, -0.2],
        }, index=[0, 1])
        scores = calculate_score(df_woe, ["feat_a", "feat_b"], mock_scorecard,
                                 factor=28.85,offset=500, intercept=-0.5)
        assert not scores.isna().any()

    def test_length_matches_input(self, mock_scorecard):
        df_woe = pd.DataFrame({
            "feat_a": [0.5, -0.3, 0.5, -0.3],
            "feat_b": [0.8, -0.2, 0.8, -0.2],
        }, index=[0, 1, 2, 3])
        scores = calculate_score(df_woe, ["feat_a", "feat_b"], mock_scorecard,
                                 factor=28.85, offset=500, intercept=-0.5)
        assert len(scores) == 4

    def test_rounding_handles_float_precision(self, mock_scorecard):
        df_woe = pd.DataFrame({
            "feat_a": [0.50001, -0.29999],
            "feat_b": [0.80001, -0.19999],
        }, index=[0, 1])
        scores = calculate_score(df_woe, ["feat_a", "feat_b"], mock_scorecard,
                                 factor=28.85, offset=500, intercept=-0.5)
        assert not scores.isna().any()


# ---------------------------------------------------------------------------
# Tests: score_distribution
# ---------------------------------------------------------------------------

class TestScoreDistribution:
    def test_runs_without_error(self, capsys):
        scores = pd.Series([100, 200, 300, 400, 500])
        score_distribution(scores, label="test")
        captured = capsys.readouterr()
        assert "Score distribution" in captured.out
