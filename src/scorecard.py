"""Scorecard scaling utilities for the credit scorecard project.

Functions extracted from notebooks/04_scorecard.ipynb.
Phase 8: Convert logistic regression model into interpretable scorecard points.

Each function is documented with the rationale behind design choices.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# 1. Artifact loading
# ---------------------------------------------------------------------------

def load_phase7_artifacts(base_path="../models", data_path="../data/processed"):
    """Load all artifacts produced by Phase 7 (model building).

    Rationale: Centralize loading so every notebook/script uses the same
    entry point. Avoids scattered pd.read_parquet / joblib.load calls.

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


# ---------------------------------------------------------------------------
# 2. Scaling parameters
# ---------------------------------------------------------------------------

def compute_scaling_params(PDO=20, base_score=600, base_odds=20):
    """Compute Factor and Offset from PDO parameters.

    Rationale:
    - PDO (Points to Double the Odds): how many points for a 2x increase in odds.
      20 is standard for credit scorecards (easy to communicate).
    - Factor = PDO / ln(2): converts log-odds to points. If odds double,
      log-odds increases by ln(2), so score increases by PDO points.
    - Offset: ensures the base_score corresponds to base_odds.

    Parameters
    ----------
    PDO : int
        Points to Double the Odds (default 20).
    base_score : int
        Score for the average customer (default 600).
    base_odds : int
        Odds (good:bad) for the average customer (default 20:1).

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


# ---------------------------------------------------------------------------
# 3. Coefficient extraction
# ---------------------------------------------------------------------------

def extract_coefficients(model, selected_features):
    """Extract coefficients and intercept from sklearn logistic model.

    Rationale: sklearn stores coef_ as a 2D array (n_classes x n_features).
    We flatten it and map to feature names for readable access.
    The intercept is separated because it becomes the base_points (offset).

    Returns
    -------
    coef_dict : dict {feature: coefficient}
    intercept : float
    """
    coef_dict = dict(zip(selected_features, model.coef_[0]))
    intercept = float(model.intercept_[0])
    return coef_dict, intercept


# ---------------------------------------------------------------------------
# 4. Scorecard construction
# ---------------------------------------------------------------------------

def build_scorecard(bp, selected_features, coef_dict, factor, offset, intercept):
    """Build the scorecard: points per bin for each variable.

    Rationale for design choices:
    - Intercept and offset are distributed across variables (intercept_per_var,
      offset_per_var) so each variable carries a share of the base score.
      This makes the scorecard self-contained: summing all variable points
      gives the final score without a separate constant.
    - Points formula: -(coef * woe + intercept_per_var) * factor + offset_per_var
      The negative sign ensures higher WoE (less risk) → higher score.
    - WoE is rounded to 4 decimals for consistent mapping later.
    - Bins with 0 count or "Total" are skipped (not scoreable).
    - try/except on float conversion handles optbinning's StringArray bins
      that may produce non-numeric WoE values.

    Parameters
    ----------
    bp : optbinning BinningProcess
    selected_features : list of str
    coef_dict : dict {feature: coefficient}
    factor, offset, intercept : float

    Returns
    -------
    pd.DataFrame with columns: variable, bin, woe, count, event_rate, points
    """
    rows = []
    n_features = len(selected_features)

    # Distribute intercept and offset across variables
    # Rationale: each variable gets an equal share of the base score,
    # making the scorecard modular (no separate constant line).
    intercept_per_var = intercept / n_features
    offset_per_var = offset / n_features

    for var in selected_features:
        table = bp.get_binned_variable(var).binning_table.build()
        coef = float(coef_dict.get(var, 0.0))

        for _, row in table.iterrows():
            bin_name = str(row["Bin"])

            # Skip Total row and empty bins
            # Rationale: Total is an aggregate, not a scoreable bin.
            # Empty bins have no WoE and would produce NaN points.
            if bin_name.lower() == "total" or row["Count"] == 0:
                continue

            raw_woe = row["WoE"]

            # Guard against non-numeric WoE (optbinning StringArray bins)
            # Rationale: optbinning sometimes stores WoE as strings or NaN
            # for special bins. We skip those gracefully.
            if raw_woe is None or str(raw_woe).strip() in ("", "nan"):
                continue

            try:
                woe = float(raw_woe)
                event_rate = float(row["Event rate"])
                count = int(row["Count"])
            except (ValueError, TypeError):
                continue  # Skip non-numeric bins

            # Points calculation
            # Rationale: negative sign because higher WoE = less risk = higher score
            points = -((coef * woe + intercept_per_var) * float(factor)) + float(offset_per_var)

            rows.append({
                "variable": var,
                "bin": bin_name,
                "woe": round(woe, 4),
                "count": count,
                "event_rate": round(event_rate, 4),
                "points": round(points, 2),
            })

    scorecard_df = pd.DataFrame(rows)
    return scorecard_df


# ---------------------------------------------------------------------------
# 5. Score calculation
# ---------------------------------------------------------------------------

def calculate_score(df_woe, selected_features, scorecard, factor, offset, intercept):
    """Calculate the score for each client.

    Rationale for approach:
    - We round WoE to 4 decimals on both sides (scorecard and client) to
      avoid floating-point mismatch (the #1 cause of NaN in mapping).
    - We map WoE → points (not bin → points) because the WoE is already
      computed and stored in df_woe. This avoids re-running the binning process.
    - base_points = offset + factor * intercept: converts the logistic
      intercept into points and adds the base offset.

    Parameters
    ----------
    df_woe : pd.DataFrame
        WoE-transformed data (train or test).
    selected_features : list of str
    scorecard : pd.DataFrame
    factor, offset, intercept : float

    Returns
    -------
    pd.Series
        Score for each client.
    """
    base_points = offset + factor * intercept
    scores = pd.Series(base_points, index=df_woe.index)

    for var in selected_features:
        var_card = scorecard[scorecard["variable"] == var]

        # Map WoE (rounded to 4 decimals) → points
        # Rationale: rounding ensures exact match between scorecard WoE
        # and client WoE despite floating-point differences
        woe_to_points = dict(zip(var_card["woe"].round(4), var_card["points"]))
        client_woes = df_woe[var].round(4)
        client_points = client_woes.map(woe_to_points)

        # Safety check: warn if any client couldn't be mapped
        if client_points.isna().any():
            n_missing = client_points.isna().sum()
            unmatched = client_woes[client_points.isna()].unique()
            print(f"⚠️ {var}: {n_missing} clients non mappés. WoE non matchés: {unmatched}")

        scores = scores + client_points

    return scores


# ---------------------------------------------------------------------------
# 6. Score distribution analysis
# ---------------------------------------------------------------------------

def score_distribution(scores, label="train"):
    """Print score distribution statistics and percentiles.

    Rationale: A good scorecard should have approximately normal
    distribution with similar train/test distributions (no drift).
    """
    print(f"\n=== Distribution des scores ({label}) ===")
    print(scores.describe())
    print(f"\nPercentiles ({label}) :")
    print(scores.quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))


# ---------------------------------------------------------------------------
# 7. Persistence
# ---------------------------------------------------------------------------

def load_scaling_params(base_path="../models"):
    """Load scaling parameters saved by save_scorecard_artifacts.

    Rationale: Phase 9 (validation) needs Factor, Offset and intercept
    to reconstruct scores or verify the scorecard. Centralizing the
    loading avoids scattered json.load calls.

    Returns
    -------
    dict with keys: PDO, BASE_SCORE, BASE_ODDS, Factor, Offset, intercept
    """
    with open(f"{base_path}/scaling_params.json") as f:
        return json.load(f)


def save_scorecard_artifacts(scorecard, scaling_params, train_scores, test_scores,
                            base_path="../models", data_path="../data/processed"):
    """Save all Phase 8 artifacts to disk.

    Rationale: Centralize saving so every artifact is stored consistently.
    Scores are saved as parquet (fast, typed). Scorecard as CSV (human-readable)
    and PKL (programmatic access). Scaling params as JSON (auditable).

    Parameters
    ----------
    scorecard : pd.DataFrame
    scaling_params : dict
    train_scores, test_scores : pd.Series
    base_path : str
    data_path : str
    """
    scorecard.to_csv(f"{base_path}/scorecard.csv", index=False)
    joblib.dump(scorecard, f"{base_path}/scorecard.pkl")

    with open(f"{base_path}/scaling_params.json", "w") as f:
        json.dump(scaling_params, f, indent=2)

    train_scores.to_frame("score").to_parquet(f"{data_path}/train_scores.parquet")
    test_scores.to_frame("score").to_parquet(f"{data_path}/test_scores.parquet")

    # Save scores with target for validation phase
    # Rationale: Phase 9 needs both score and default in one DataFrame
    # for KS, AUC, Gini calculations
    y_train = pd.read_parquet(f"{data_path}/y_train.parquet").squeeze()
    y_test = pd.read_parquet(f"{data_path}/y_test.parquet").squeeze()
    y_train.to_frame("default").join(train_scores.rename("score")).to_parquet(
        f"{data_path}/train_scores_with_target.parquet"
    )
    y_test.to_frame("default").join(test_scores.rename("score")).to_parquet(
        f"{data_path}/test_scores_with_target.parquet"
    )

    print("✅ Scorecard et scores sauvegardés")
