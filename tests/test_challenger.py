"""Tests for src/challenger.py (scikit-learn gradient boosting only)."""


import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from challenger import train_challenger, train_sklearn_gbm, compare_models, gini, predict_pd, roc_curves


def test_train_sklearn_gbm(small_data):
    X, y = small_data
    model = train_sklearn_gbm(X, y)
    pds = predict_pd(model, X)
    assert pds.shape == (len(X),)
    assert np.all((pds >= 0) & (pds <= 1))


@pytest.fixture
def small_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"))
    y = (X["a"] + rng.normal(size=60) > 0).astype(int)
    return X, y


def test_train_and_predict(small_data):
    X, y = small_data
    model = train_challenger(X, y)
    pds = predict_pd(model, X)
    assert pds.shape == (len(X),)
    assert np.all((pds >= 0) & (pds <= 1))


def test_gini(small_data):
    X, y = small_data
    model = train_challenger(X, y)
    pds = predict_pd(model, X)
    assert -1 <= gini(y, pds) <= 1


def test_compare_models_columns(small_data):
    X, y = small_data
    model = train_challenger(X, y)
    pds = predict_pd(model, X)
    df = compare_models(y, pds, pds)
    assert len(df) == 2
    assert {"model", "AUC", "Gini", "total_cost"} <= set(df.columns)


def test_roc_curves(small_data):
    X, y = small_data
    pds = predict_pd(train_challenger(X, y), X)
    curves = roc_curves(y, pds, pds)
    assert set(curves) == {"Champion", "Challenger"}
