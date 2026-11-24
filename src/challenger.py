"""Challenger model for champion/challenger validation.

Uses only scikit-learn gradient boosting.
Two challengers are available:
  - train_challenger()     : fast histogram-based boosting (default)
  - train_sklearn_gbm()    : classical GradientBoostingClassifier
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from cost_matrix import total_portfolio_cost


def train_challenger(X: pd.DataFrame, y: Sequence[int], random_state: int = 42):
    """Default challenger: HistGradientBoostingClassifier."""
    model = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_leaf_nodes=15,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def train_sklearn_gbm(X: pd.DataFrame, y: Sequence[int], random_state: int = 42):
    """Alternative challenger: sklearn GradientBoostingClassifier."""
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def predict_pd(model, X: pd.DataFrame) -> np.ndarray:
    """Predicted probability of default."""
    return model.predict_proba(X)[:, 1]


def gini(y_true: Sequence[int], pd_values: Sequence[float]) -> float:
    """Gini coefficient from AUC."""
    return float(2.0 * roc_auc_score(y_true, pd_values) - 1.0)


def compare_models(
    y_true: Sequence[int],
    pd_champion: Sequence[float],
    pd_challenger: Sequence[float],
    threshold_pd: Optional[float] = None,
) -> pd.DataFrame:
    """Compare champion and challenger on Gini and portfolio cost."""
    from cost_matrix import optimal_pd_threshold

    y = np.asarray(y_true)
    c = np.asarray(pd_champion, dtype=float)
    ch = np.asarray(pd_challenger, dtype=float)
    t = optimal_pd_threshold() if threshold_pd is None else threshold_pd

    rows = []
    for name, pds in [("Scorecard (Champion)", c), ("Challenger", ch)]:
        approved = pds < t
        rows.append({
            "model": name,
            "AUC": float(roc_auc_score(y, pds)),
            "Gini": gini(y, pds),
            "total_cost": total_portfolio_cost(y, pds, approved),
            "approval_rate": float(approved.mean()),
            "bad_accepted": int(np.sum((y == 1) & approved)),
            "good_rejected": int(np.sum((y == 0) & ~approved)),
        })
    return pd.DataFrame(rows)


def roc_curves(y_true, pd_champion, pd_challenger):
    """ROC curve coordinates for both models."""
    out = {}
    for name, pds in [("Champion", pd_champion), ("Challenger", pd_challenger)]:
        fpr, tpr, _ = roc_curve(y_true, pds)
        out[name] = (fpr, tpr, roc_auc_score(y_true, pds))
    return out