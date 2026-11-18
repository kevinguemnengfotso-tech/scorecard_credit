"""Model building utilities for the credit scorecard project.

Functions extracted from notebooks/03_model_building.ipynb.
"""

import json
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_woe_data(base_path="../data/processed"):
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


# ---------------------------------------------------------------------------
# Multicollinearity
# ---------------------------------------------------------------------------

def check_vif(X, threshold=10):
    """Compute VIF for each feature and return those above threshold."""
    vif_df = pd.DataFrame({
        "feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    }).sort_values("VIF", ascending=False)
    to_drop = vif_df[vif_df["VIF"] > threshold]["feature"].tolist()
    return vif_df, to_drop


def check_collinearity(X, threshold=0.99):
    """Detect quasi-collinear feature pairs (|r| >= threshold)."""
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


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_logistic(X, y):
    """Fit logistic regression with statsmodels (for p-values and inference)."""
    X_const = sm.add_constant(X)
    model = sm.Logit(y, X_const).fit(disp=0)
    return model


def fit_logistic_sklearn(X, y, features=None, C=1.0, max_iter=1000):
    """Fit logistic regression with sklearn (for production)."""
    if features is not None:
        X = X[features]
    model = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter)
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def backward_elimination(X, y, p_threshold=0.05):
    """Iteratively remove features with p-value > p_threshold."""
    features = list(X.columns)
    while True:
        X_const = sm.add_constant(X[features])
        model = sm.Logit(y, X_const).fit(disp=0)
        p_values = model.pvalues.drop("const")
        worst = p_values.idxmax()
        if p_values[worst] > p_threshold:
            features.remove(worst)
            print(f"Éliminé : {worst} (p={p_values[worst]:.4f})")
        else:
            break
    return features, model


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def diagnose_model(model, feature_names):
    """Build a diagnostic DataFrame from a statsmodels Logit result."""
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


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model, features, model_path="models/logistic_model.pkl",
               features_path="models/selected_features.json"):
    """Save sklearn model and selected feature list."""
    joblib.dump(model, model_path)
    with open(features_path, "w") as f:
        json.dump(features, f)
    print(f"Modèle sauvegardé : {model_path}")
    print(f"Features sauvegardées : {features_path}")


def load_model(model_path="models/logistic_model.pkl",
               features_path="models/selected_features.json"):
    """Load sklearn model and selected feature list."""
    model = joblib.load(model_path)
    with open(features_path) as f:
        features = json.load(f)
    return model, features