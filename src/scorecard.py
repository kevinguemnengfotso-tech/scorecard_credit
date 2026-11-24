"""Scorecard scaling utilities for the credit scorecard project.

Phase 8: Convert logistic regression model into interpretable scorecard points.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def load_phase7_artifacts(
    base_path: str = "models",
    data_path: str = "data/processed",
) -> Tuple[LogisticRegression, List[str], Any, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load all artifacts produced by Phase 7 (model building).

    Returns
    -------
    model : sklearn LogisticRegression
    selected_features : list of str
    bp : optbinning BinningProcess
    X_train_woe, X_test_woe : pd.DataFrame
    y_train, y_test : pd.Series
    """
    model = joblib.load(f"{base_path}/logistic_model.pkl")
    with open(f"{base_path}/selected_features.json") as f:
        selected_features = json.load(f)
    bp = joblib.load(f"{base_path}/binning_process.pkl")
    X_train_woe = pd.read_parquet(f"{data_path}/X_train_woe.parquet")
    X_test_woe = pd.read_parquet(f"{data_path}/X_test_woe.parquet")
    y_train = pd.read_parquet(f"{data_path}/y_train.parquet").squeeze()
    y_test = pd.read_parquet(f"{data_path}/y_test.parquet").squeeze()
    return model, selected_features, bp, X_train_woe, X_test_woe, y_train, y_test


def compute_scaling_params(
    PDO: int = 20, base_score: int = 600, base_odds: int = 20
) -> Tuple[float, float, Dict[str, Any]]:
    """Compute Factor and Offset from PDO parameters.

    Rationale:
    - PDO (Points to Double the Odds): points for a 2x increase in odds.
    - Factor = PDO / ln(2): converts log-odds to points.
    - Offset: ensures base_score corresponds to base_odds.

    Returns
    -------
    Factor, Offset : float
    scaling_params : dict
    """
    Factor = PDO / np.log(2)
    Offset = base_score - Factor * np.log(base_odds)
    scaling_params = {
        "PDO": PDO,
        "BASE_SCORE": base_score,
        "BASE_ODDS": base_odds,
        "Factor": Factor,
        "Offset": Offset,
    }
    return Factor, Offset, scaling_params


def extract_coefficients(
    model: LogisticRegression, selected_features: List[str]
) -> Tuple[Dict[str, float], float]:
    """Extract coefficients and intercept from sklearn logistic model.

    Returns
    -------
    coef_dict : dict {feature: coefficient}
    intercept : float
    """
    coef_dict = dict(zip(selected_features, model.coef_[0]))
    intercept = float(model.intercept_[0])
    return coef_dict, intercept


def build_scorecard(
    bp,
    selected_features: List[str],
    coef_dict: Dict[str, float],
    factor: float,
    offset: float,
    intercept: float,
) -> pd.DataFrame:
    """Build the scorecard: points per bin for each variable.

    Rationale for design choices:
    - Intercept and offset are distributed across variables so each
      variable carries a share of the base score. This makes the
      scorecard self-contained.
    - Points formula: -(coef * woe + intercept_per_var) * factor + offset_per_var
      The negative sign ensures higher WoE (less risk) -> higher score.
    - WoE is rounded to 4 decimals for consistent mapping later.
    - Bins with 0 count or "Total" are skipped.

    Returns
    -------
    pd.DataFrame with columns: variable, bin, woe, count, event_rate, points
    """
    rows = []
    n_features = len(selected_features)

    intercept_per_var = intercept / n_features
    offset_per_var = offset / n_features

    for var in selected_features:
        table = bp.get_binned_variable(var).binning_table.build()
        coef = float(coef_dict.get(var, 0.0))

        for _, row in table.iterrows():
            bin_name = str(row["Bin"])

            if bin_name.lower() == "total" or row["Count"] == 0:
                continue

            raw_woe = row["WoE"]

            if raw_woe is None or str(raw_woe).strip() in ("", "nan"):
                continue

            try:
                woe = float(raw_woe)
                event_rate = float(row["Event rate"])
                count = int(row["Count"])
            except (ValueError, TypeError):
                continue

            points = -((coef * woe + intercept_per_var) * float(factor)) + float(offset_per_var)

            rows.append({
                "variable": var,
                "bin": bin_name,
                "woe": round(woe, 4),
                "count": count,
                "event_rate": round(event_rate, 4),
                "points": round(points, 2),
            })

    return pd.DataFrame(rows)


def calculate_score(
    df_woe: pd.DataFrame,
    selected_features: List[str],
    scorecard: pd.DataFrame,
    factor: float,
    offset: float,
    intercept: float,
) -> pd.Series:
    """Calculate the score for each client.

    Rationale: WoE is rounded to 4 decimals on both sides to avoid
    floating-point mismatch. base_points = offset + factor * intercept.

    Returns
    -------
    pd.Series : score for each client
    """
    base_points = offset + factor * intercept
    scores = pd.Series(base_points, index=df_woe.index)

    for var in selected_features:
        var_card = scorecard[scorecard["variable"] == var]
        woe_to_points = dict(zip(var_card["woe"].round(4), var_card["points"]))
        client_woes = df_woe[var].round(4)
        client_points = client_woes.map(woe_to_points)

        if client_points.isna().any():
            n_missing = client_points.isna().sum()
            unmatched = client_woes[client_points.isna()].unique()
            print(f"Warning: {var}: {n_missing} clients not mapped. Unmatched WoE: {unmatched}")

        scores = scores + client_points

    return scores


def score_distribution(scores: pd.Series, label: str = "train") -> None:
    """Print score distribution statistics and percentiles."""
    print(f"\n=== Score distribution ({label}) ===")
    print(scores.describe())
    print(f"\nPercentiles ({label}) :")
    print(scores.quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))


def save_scorecard_artifacts(
    scorecard: pd.DataFrame,
    scaling_params: Dict[str, Any],
    train_scores: pd.Series,
    test_scores: pd.Series,
    base_path: str = "models",
    data_path: str = "data/processed",
) -> None:
    """Save all Phase 8 artifacts to disk."""
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    scorecard.to_csv(f"{base_path}/scorecard.csv", index=False)
    joblib.dump(scorecard, f"{base_path}/scorecard.pkl")

    with open(f"{base_path}/scaling_params.json", "w") as f:
        json.dump(scaling_params, f, indent=2)

    train_scores.to_frame("score").to_parquet(f"{data_path}/train_scores.parquet")
    test_scores.to_frame("score").to_parquet(f"{data_path}/test_scores.parquet")

    y_train = pd.read_parquet(f"{data_path}/y_train.parquet").squeeze()
    y_test = pd.read_parquet(f"{data_path}/y_test.parquet").squeeze()
    y_train.to_frame("default").join(train_scores.rename("score")).to_parquet(
        f"{data_path}/train_scores_with_target.parquet"
    )
    y_test.to_frame("default").join(test_scores.rename("score")).to_parquet(
        f"{data_path}/test_scores_with_target.parquet"
    )

    print("Scorecard and scores saved")


def load_scaling_params(base_path: str = "models") -> Dict[str, Any]:
    """Load scaling parameters saved by save_scorecard_artifacts."""
    with open(f"{base_path}/scaling_params.json") as f:
        return json.load(f)