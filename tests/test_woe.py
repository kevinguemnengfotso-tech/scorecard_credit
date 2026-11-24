"""Tests for src/woe.py — Phase 6 binning & WoE functions.

Run with: pytest tests/test_woe.py -v
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from woe import (
    split_data,
    bin_single_variable,
    multi_binning,
    select_features_by_iv,
    check_redundancy,
    calc_psi,
    interpret_psi,
    save_woe_artifacts,
    load_woe_artifacts

)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """Small synthetic dataset for testing."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "age": np.random.randint(18, 70, n),
        "income": np.random.randint(1000, 10000, n),
        "category": np.random.choice(["A", "B", "C"], n),
    })
    y = pd.Series(np.random.choice([0, 1], n, p=[0.7, 0.3]))
    return df, y


@pytest.fixture
def mock_bp(sample_data):
    """A fitted BinningProcess for testing."""
    X, y = sample_data
    cat_cols = ["category"]
    bp, X_train_woe, X_test_woe, X_val_woe = None, None, None, None
    # Use multi_binning to create a fitted bp
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    X_train_woe, X_test_woe, X_val_woe, bp = multi_binning(
        X_train, y_train, X_test, X_val, cat_cols
    )
    return bp, X_train_woe, X_test_woe, X_val_woe, y_train, y_test, X_train, X_test


# ---------------------------------------------------------------------------
# Tests: split_data
# ---------------------------------------------------------------------------

class TestSplitData:
    def test_returns_six_elements(self, sample_data):
        X, y = sample_data
        result = split_data(X, y)
        assert len(result) == 6

    def test_train_larger_than_test(self, sample_data):
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        assert len(X_train) > len(X_test)
        assert len(X_train) > len(X_val)

    def test_stratification_preserves_rate(self, sample_data):
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        original_rate = y.mean()
        assert abs(y_train.mean() - original_rate) < 0.05
        assert abs(y_test.mean() - original_rate) < 0.1

    def test_no_overlap(self, sample_data):
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        train_idx = set(X_train.index)
        test_idx = set(X_test.index)
        val_idx = set(X_val.index)
        assert len(train_idx & test_idx) == 0
        assert len(train_idx & val_idx) == 0
        assert len(test_idx & val_idx) == 0

    def test_total_size_preserved(self, sample_data):
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        assert len(X_train) + len(X_val) + len(X_test) == len(X)


# ---------------------------------------------------------------------------
# Tests: bin_single_variable
# ---------------------------------------------------------------------------

class TestBinSingleVariable:
    def test_returns_optb_and_table(self, sample_data):
        X, y = sample_data
        optb, table = bin_single_variable(X["age"], y, name="age")
        assert optb is not None
        assert isinstance(table, pd.DataFrame)

    def test_table_has_woe_column(self, sample_data):
        X, y = sample_data
        optb, table = bin_single_variable(X["age"], y, name="age")
        assert "WoE" in table.columns

    def test_table_has_iv_total(self, sample_data):
        X, y = sample_data
        optb, table = bin_single_variable(X["age"], y, name="age")
        assert "Totals" in table["Bin"].values or "Totals" in table.index

    def test_categorical_variable(self, sample_data):
        X, y = sample_data
        optb, table = bin_single_variable(X["category"], y, name="category", dtype="categorical")
        assert optb is not None
        assert isinstance(table, pd.DataFrame)


# ---------------------------------------------------------------------------
# Tests: multi_binning
# ---------------------------------------------------------------------------

class TestMultiBinning:
    def test_returns_four_elements(self, sample_data):
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        cat_cols = ["category"]
        result = multi_binning(X_train, y_train, X_test, X_val, cat_cols)
        assert len(result) == 4

    def test_woe_dataframes_same_shape(self, sample_data):
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        cat_cols = ["category"]
        train_woe, test_woe, val_woe, bp = multi_binning(X_train, y_train, X_test, X_val, cat_cols)
        assert train_woe.shape[0] == X_train.shape[0]
        assert test_woe.shape[0] == X_test.shape[0]
        assert val_woe.shape[0] == X_val.shape[0]

    def test_woe_values_are_numeric(self, sample_data):
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        cat_cols = ["category"]
        train_woe, test_woe, val_woe, bp = multi_binning(X_train, y_train, X_test, X_val, cat_cols)
        assert train_woe.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()

    def test_bp_has_summary(self, sample_data):
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        cat_cols = ["category"]
        train_woe, test_woe, val_woe, bp = multi_binning(X_train, y_train, X_test, X_val, cat_cols)
        summary = bp.summary()
        assert "name" in summary.columns
        assert "iv" in summary.columns


# ---------------------------------------------------------------------------
# Tests: select_features_by_iv
# ---------------------------------------------------------------------------

class TestSelectFeaturesByIv:
    def test_returns_list_and_dataframe(self, mock_bp):
        bp, *_ = mock_bp
        selected, summary = select_features_by_iv(bp, threshold=0.01)
        assert isinstance(selected, list)
        assert isinstance(summary, pd.DataFrame)

    def test_summary_has_iv_column(self, mock_bp):
        bp, *_ = mock_bp
        selected, summary = select_features_by_iv(bp, threshold=0.01)
        assert "iv" in summary.columns

    def test_threshold_filters(self, mock_bp):
        bp, *_ = mock_bp
        selected_low, _ = select_features_by_iv(bp, threshold=0.0)
        selected_high, _ = select_features_by_iv(bp, threshold=0.99)
        assert len(selected_low) >= len(selected_high)

    def test_all_selected_have_iv_above_threshold(self, mock_bp):
        bp, *_ = mock_bp
        threshold = 0.05
        selected, summary = select_features_by_iv(bp, threshold=threshold)
        for var in selected:
            iv = summary.loc[summary["name"] == var, "iv"].iloc[0]
            assert iv >= threshold


# ---------------------------------------------------------------------------
# Tests: check_redundancy
# ---------------------------------------------------------------------------

class TestCheckRedundancy:
    def test_returns_dict_with_keys(self, mock_bp):
        bp, X_train_woe, *_ = mock_bp
        selected = ["age", "income", "category"]
        # Filter to available columns
        available = [c for c in selected if c in X_train_woe.columns]
        if not available:
            available = list(X_train_woe.columns[:2])
        result = check_redundancy(X_train_woe, available)
        assert "corr_pairs" in result
        assert "vif_df" in result

    def test_vif_df_has_variable_column(self, mock_bp):
        bp, X_train_woe, *_ = mock_bp
        cols = list(X_train_woe.columns[:2])
        result = check_redundancy(X_train_woe, cols)
        assert "variable" in result["vif_df"].columns
        assert "VIF" in result["vif_df"].columns

    def test_corr_pairs_is_dataframe(self, mock_bp):
        bp, X_train_woe, *_ = mock_bp
        cols = list(X_train_woe.columns[:2])
        result = check_redundancy(X_train_woe, cols)
        assert isinstance(result["corr_pairs"], pd.DataFrame)


# ---------------------------------------------------------------------------
# Tests: calc_psi
# ---------------------------------------------------------------------------

class TestCalcPsi:
    def test_identical_distributions_low_psi(self):
        data = np.random.normal(0, 1, 1000)
        psi = calc_psi(data, data.copy())
        assert psi < 0.1

    def test_different_distributions_higher_psi(self):
        data1 = np.random.normal(0, 1, 1000)
        data2 = np.random.normal(2, 1, 1000)
        psi = calc_psi(data1, data2)
        assert psi > 0.1

    def test_returns_float(self):
        data = np.random.normal(0, 1, 100)
        psi = calc_psi(data, data.copy())
        assert isinstance(psi, float)

    def test_psi_non_negative(self):
        data1 = np.random.normal(0, 1, 500)
        data2 = np.random.normal(0.5, 1.2, 500)
        psi = calc_psi(data1, data2)
        assert psi >= 0


# ---------------------------------------------------------------------------
# Tests: interpret_psi
# ---------------------------------------------------------------------------

class TestInterpretPsi:
    def test_stable(self):
        assert interpret_psi(0.05) == "Stable (no drift)"

    def test_moderate(self):
        assert interpret_psi(0.15) == "Moderate drift (monitor)"

    def test_significant(self):
        assert interpret_psi(0.30) == "Significant drift (recalibration needed)"

    def test_boundary_0_1(self):
        assert interpret_psi(0.1) == "Moderate drift (monitor)"

    def test_boundary_0_25(self):
        assert interpret_psi(0.25) == "Significant drift (recalibration needed)"


# ---------------------------------------------------------------------------
# Tests: save/load woe artifacts
# ---------------------------------------------------------------------------

class TestSaveLoadWoE:
    def test_round_trip(self, tmp_path, mock_bp):
        bp, X_train_woe, X_test_woe, X_val_woe, y_train, y_test, X_train, X_test = mock_bp
        base = str(tmp_path / "models")
        data = str(tmp_path / "data")
        save_woe_artifacts(bp, X_train_woe, X_test_woe, X_val_woe, base_path=base, data_path=data)
        bp2, train2, test2, val2 = load_woe_artifacts(base_path=base, data_path=data)
        assert bp2 is not None
        assert train2.shape == X_train_woe.shape
        assert test2.shape == X_test_woe.shape
        assert val2 is not None

    def test_load_without_val(self, tmp_path, mock_bp):
        bp, X_train_woe, X_test_woe, X_val_woe, y_train, y_test, X_train, X_test = mock_bp
        base = str(tmp_path / "models2")
        data = str(tmp_path / "data2")
        save_woe_artifacts(bp, X_train_woe, X_test_woe, base_path=base, data_path=data)
        bp2, train2, test2, val2 = load_woe_artifacts(base_path=base, data_path=data)
        assert val2 is None
