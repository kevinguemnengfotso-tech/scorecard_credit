"""Model building utilities for the credit scorecard project."""

import json
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor


def load_woe_data(
    base_path: str = "../data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load WoE-transformed train/test data and targets.

    Returns
    -------
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.Series
    """
    X_train = pd.read_parquet(f"{base_path}/X_train_woe.parquet")
    X_test = pd.read_parquet(f"{base_path}/X_test_woe.parquet")
    y_train = pd.read_parquet(f"{base_path}/y_train.parquet").squeeze()
    y_test = pd.read_parquet(f"{base_path}/y_test.parquet").squeeze()
    return X_train, X_test, y_train, y_test


def check_vif(X: pd.DataFrame, threshold: float = 10.0) -> Tuple[pd.DataFrame, List[str]]:
    """Compute VIF for each feature and return those above threshold.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (WoE-transformed).
    threshold : float
        VIF threshold above which a variable is flagged.

    Returns
    -------
    vif_df : pd.DataFrame
        VIF sorted descending.
    to_drop : list
        Feature names with VIF > threshold.
    """
    vif_df = pd.DataFrame({
        "feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    }).sort_values("VIF", ascending=False)
    to_drop = vif_df[vif_df["VIF"] > threshold]["feature"].tolist()
    return vif_df, to_drop


def check_collinearity(X: pd.DataFrame, threshold: float = 0.99) -> pd.DataFrame:
    """Detect quasi-collinear feature pairs (|r| >= threshold).

    Returns
    -------
    pd.DataFrame with columns: variable_1, variable_2, correlation.
    """
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    pairs = []
    for col in upper_tri.columns:
        high = upper_tri.index[upper_tri[col] >= threshold].tolist()
        for row in high:
            pairs.append({
                "variable_1": row,
                "variable_2": col,
                "correlation": round(corr_matrix.loc[row, col], 4),
            })
    return pd.DataFrame(pairs)


def fit_logistic(X: pd.DataFrame, y: pd.Series) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit logistic regression with statsmodels (for p-values and inference).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target (0=good, 1=bad).

    Returns
    -------
    statsmodels Logit result.
    """
    X_const = sm.add_constant(X)
    model = sm.Logit(y, X_const).fit(disp=0)
    return model


def fit_logistic_sklearn(
    X: pd.DataFrame,
    y: pd.Series,
    features: Optional[List[str]] = None,
    C: float = 1.0,
    max_iter: int = 1000,
) -> LogisticRegression:
    """Fit logistic regression with sklearn (for production).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target.
    features : list, optional
        Subset of features to use. If None, uses all columns.
    C : float
        Inverse regularization strength.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    sklearn LogisticRegression fitted model.
    """
    if features is not None:
        X = X[features]
    model = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter)
    model.fit(X, y)
    return model


def backward_elimination(
    X: pd.DataFrame, y: pd.Series, p_threshold: float = 0.05
) -> Tuple[List[str], sm.regression.linear_model.RegressionResultsWrapper]:
    """Iteratively remove features with p-value > p_threshold.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target.
    p_threshold : float
        Significance threshold.

    Returns
    -------
    features : list
        Retained feature names.
    model : statsmodels Logit result
        Final fitted model.
    """
    features = list(X.columns)
    while True:
        X_const = sm.add_constant(X[features])
        model = sm.Logit(y, X_const).fit(disp=0)
        p_values = model.pvalues.drop("const")
        worst = p_values.idxmax()
        if p_values[worst] > p_threshold:
            features.remove(worst)
            print(f"Eliminated: {worst} (p={p_values[worst]:.4f})")
        else:
            break
    return features, model


def diagnose_model(model, feature_names: List[str]) -> pd.DataFrame:
    """Build a diagnostic DataFrame from a statsmodels Logit result.

    Returns
    -------
    pd.DataFrame with columns: feature, coef, std_err, p_value, significant, sign, stable.
    """
    coef_df = pd.DataFrame({
        "feature": ["const"] + feature_names,
        "coef": model.params.values,
        "std_err": model.bse.values,
        "p_value": model.pvalues.values,
    })
    coef_df["significant"] = coef_df["p_value"] < 0.05
    coef_df["sign"] = coef_df["coef"].apply(lambda x: "positive" if x > 0 else "negative")
    coef_df["stable"] = coef_df["std_err"] < 1.0
    return coef_df


def save_model(
    model: LogisticRegression,
    features: List[str],
    model_path: str = "models/logistic_model.pkl",
    features_path: str = "models/selected_features.json",
) -> None:
    """Save sklearn model and selected feature list."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    joblib.dump(model, model_path)
    with open(features_path, "w") as f:
        json.dump(features, f)
    print(f"Model saved: {model_path}")
    print(f"Features saved: {features_path}")


def load_model(
    model_path: str = "models/logistic_model.pkl",
    features_path: str = "models/selected_features.json",
) -> Tuple[LogisticRegression, List[str]]:
    """Load sklearn model and selected feature list."""
    model = joblib.load(model_path)
    with open(features_path) as f:
        features = json.load(f)
    return model, features