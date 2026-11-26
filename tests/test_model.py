"""Tests for src/model.py functions."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model import (
    check_vif,
    check_collinearity,
    fit_logistic,
    fit_logistic_sklearn,
    backward_elimination,
    diagnose_model,
    save_model,
    load_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """Generate a small synthetic WoE-like dataset for testing."""
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({
        "var_a": rng.normal(0, 1, n),
        "var_b": rng.normal(0, 1, n),
        "var_c": rng.normal(0, 1, n),
    })
    # target correlated with var_a
    prob = 1 / (1 + np.exp(-X["var_a"]))
    y = pd.Series((rng.random(n) < prob).astype(int), name="res")
    return X, y


@pytest.fixture
def collinear_data():
    """Dataset with a quasi-collinear pair."""
    rng = np.random.default_rng(7)
    n = 100
    x = rng.normal(0, 1, n)
    X = pd.DataFrame({
        "a": x,
        "b": x + rng.normal(0, 0.001, n),  # quasi-collinear
        "c": rng.normal(0, 1, n),
    })
    y = pd.Series(rng.integers(0, 2, n), name="res")
    return X, y


# ---------------------------------------------------------------------------
# check_vif
# ---------------------------------------------------------------------------

def test_check_vif_returns_dataframe_and_list(sample_data):
    X, y = sample_data
    vif_df, to_drop = check_vif(X)
    assert isinstance(vif_df, pd.DataFrame)
    assert "feature" in vif_df.columns
    assert "VIF" in vif_df.columns
    assert isinstance(to_drop, list)


def test_check_vif_no_collinearity(sample_data):
    X, y = sample_data
    vif_df, to_drop = check_vif(X, threshold=10)
    assert len(to_drop) == 0


def test_check_vif_detects_collinearity(collinear_data):
    X, y = collinear_data
    vif_df, to_drop = check_vif(X, threshold=5)
    assert len(to_drop) > 0


# ---------------------------------------------------------------------------
# check_collinearity
# ---------------------------------------------------------------------------

def test_check_collinearity_returns_dataframe(sample_data):
    X, y = sample_data
    result = check_collinearity(X)
    assert isinstance(result, pd.DataFrame)


def test_check_collinearity_detects_pair(collinear_data):
    X, y = collinear_data
    result = check_collinearity(X, threshold=0.99)
    assert not result.empty
    assert "variable_1" in result.columns
    assert "variable_2" in result.columns
    assert "correlation" in result.columns


def test_check_collinearity_no_pairs(sample_data):
    X, y = sample_data
    result = check_collinearity(X, threshold=0.99)
    assert result.empty


# ---------------------------------------------------------------------------
# fit_logistic
# ---------------------------------------------------------------------------

def test_fit_logistic_returns_model(sample_data):
    X, y = sample_data
    model = fit_logistic(X, y)
    assert hasattr(model, "params")
    assert hasattr(model, "pvalues")
    assert model.converged


def test_fit_logistic_params_length(sample_data):
    X, y = sample_data
    model = fit_logistic(X, y)
    # const + 3 features
    assert len(model.params) == X.shape[1] + 1


# ---------------------------------------------------------------------------
# fit_logistic_sklearn
# ---------------------------------------------------------------------------

def test_fit_logistic_sklearn_returns_model(sample_data):
    X, y = sample_data
    model = fit_logistic_sklearn(X, y)
    assert hasattr(model, "predict")
    assert hasattr(model, "coef_")


def test_fit_logistic_sklearn_with_features(sample_data):
    X, y = sample_data
    model = fit_logistic_sklearn(X, y, features=["var_a", "var_b"])
    assert model.coef_.shape[1] == 2


def test_fit_logistic_sklearn_predict_proba(sample_data):
    X, y = sample_data
    model = fit_logistic_sklearn(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# backward_elimination
# ---------------------------------------------------------------------------

def test_backward_elimination_returns_tuple(sample_data):
    X, y = sample_data
    features, model = backward_elimination(X, y)
    assert isinstance(features, list)
    assert hasattr(model, "params")


def test_backward_elimination_retains_significant(sample_data):
    X, y = sample_data
    features, model = backward_elimination(X, y, p_threshold=0.05)
    p_values = model.pvalues.drop("const")
    assert all(p_values[f] <= 0.05 for f in features)


def test_backward_elimination_reduces_features(sample_data):
    X, y = sample_data
    features, model = backward_elimination(X, y, p_threshold=0.05)
    assert len(features) <= X.shape[1]


# ---------------------------------------------------------------------------
# diagnose_model
# ---------------------------------------------------------------------------

def test_diagnose_model_columns(sample_data):
    X, y = sample_data
    model = fit_logistic(X, y)
    diag = diagnose_model(model, list(X.columns))
    expected_cols = {"feature", "coef", "std_err", "p_value", "significant", "sign", "stable"}
    assert set(diag.columns) == expected_cols


def test_diagnose_model_row_count(sample_data):
    X, y = sample_data
    model = fit_logistic(X, y)
    diag = diagnose_model(model, list(X.columns))
    # const + features
    assert len(diag) == X.shape[1] + 1


def test_diagnose_model_sign_values(sample_data):
    X, y = sample_data
    model = fit_logistic(X, y)
    diag = diagnose_model(model, list(X.columns))
    assert set(diag["sign"].unique()).issubset({"positive", "negative"})


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------

def test_save_and_load_model(sample_data, tmp_path):
    X, y = sample_data
    model = fit_logistic_sklearn(X, y)
    features = list(X.columns)
    model_path = str(tmp_path / "model.pkl")
    features_path = str(tmp_path / "features.json")

    save_model(model, features, model_path=model_path, features_path=features_path)

    loaded_model, loaded_features = load_model(
        model_path=model_path, features_path=features_path
    )
    assert loaded_features == features
    np.testing.assert_array_almost_equal(
        loaded_model.coef_, model.coef_
    )


def test_save_model_creates_files(sample_data, tmp_path):
    X, y = sample_data
    model = fit_logistic_sklearn(X, y)
    features = list(X.columns)
    model_path = str(tmp_path / "model.pkl")
    features_path = str(tmp_path / "features.json")

    save_model(model, features, model_path=model_path, features_path=features_path)

    assert Path(model_path).exists()
    assert Path(features_path).exists()

    with open(features_path) as f:
        saved = json.load(f)
    assert saved == features



# Run with: python -m pytest tests/test_model.py