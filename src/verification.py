"""Verification and validation utilities for the credit scorecard project.

Functions for computing metrics, stress testing, calibration checks,
rank ordering, leakage detection, and artifact loading.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, roc_curve
from woe import calc_psi, interpret_psi


# ---------------------------------------------------------------------------
# 1. Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame, name: str = "Test") -> Dict[str, Any]:
    """Compute AUC, Gini, KS for a DataFrame with 'score' and 'default'.

    Returns
    -------
    dict with AUC, Gini, KS, KS_pvalue, n_obs, default_rate, score_min, score_max, score_mean
    """
    auc = roc_auc_score(df["default"], -df["score"])
    gini = 2 * auc - 1
    ks_stat, ks_p = ks_2samp(
        df.loc[df["default"] == 0, "score"],
        df.loc[df["default"] == 1, "score"],
    )
    return {
        "dataset": name,
        "n_obs": len(df),
        "default_rate": round(df["default"].mean(), 4),
        "score_min": round(df["score"].min(), 2),
        "score_max": round(df["score"].max(), 2),
        "score_mean": round(df["score"].mean(), 2),
        "AUC": round(auc, 4),
        "Gini": round(gini, 4),
        "KS": round(ks_stat, 4),
        "KS_pvalue": round(ks_p, 6),
    }


def verify_metrics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Verify that metrics meet the thresholds.

    Returns
    -------
    dict with train/test metrics, PSI, and check results
    """
    if thresholds is None:
        thresholds = {"AUC": 0.70, "Gini": 0.40, "KS": 0.30}

    train_m = compute_metrics(train_df, "Train")
    test_m = compute_metrics(test_df, "Test")
    psi = calc_psi(train_df["score"], test_df["score"])

    checks = {
        "AUC test > threshold": test_m["AUC"] > thresholds["AUC"],
        "Gini test > threshold": test_m["Gini"] > thresholds["Gini"],
        "KS test > threshold": test_m["KS"] > thresholds["KS"],
        "PSI < 0.25": psi < 0.25,
        "AUC gap < 0.10": (train_m["AUC"] - test_m["AUC"]) < 0.10,
    }

    for check, result in checks.items():
        status = "OK" if result else "FAIL"
        print(f"  [{status}] {check}")

    return {"train": train_m, "test": test_m, "PSI": round(psi, 4), "checks": checks}


# ---------------------------------------------------------------------------
# 2. Stress testing
# ---------------------------------------------------------------------------

def stress_test_noise(
    df: pd.DataFrame, flip_rate: float = 0.10, seed: int = 42
) -> Dict[str, Any]:
    """Stress test by random noise (label flipping).

    Rationale: simulates data quality degradation by randomly
    flipping X% of labels.
    """
    np.random.seed(seed)
    stress_df = df.copy()
    stress_df["default_stress"] = stress_df["default"].copy()
    flip_mask = np.random.choice([True, False], size=len(stress_df), p=[flip_rate, 1 - flip_rate])
    stress_df.loc[flip_mask, "default_stress"] = 1 - stress_df.loc[flip_mask, "default"]

    auc_original = roc_auc_score(df["default"], -df["score"])
    auc_stress = roc_auc_score(stress_df["default_stress"], -stress_df["score"])

    return {
        "auc_original": round(auc_original, 4),
        "auc_stress": round(auc_stress, 4),
        "flip_rate": flip_rate,
        "n_flipped": int(flip_mask.sum()),
    }


def stress_test_shift(
    train_df: pd.DataFrame, test_df: pd.DataFrame, shift: float = -50
) -> Dict[str, Any]:
    """Stress test by distribution shift."""
    psi = calc_psi(train_df["score"], test_df["score"] + shift)
    return {
        "shift": shift,
        "psi": round(psi, 4),
        "interpretation": interpret_psi(psi),
    }


def stress_test_target_increase(
    df: pd.DataFrame, increase_pct: float = 0.50
) -> Dict[str, Any]:
    """Stress test: exact increase in default rate.

    Rationale: selects non-defaulting clients with the lowest score
    and sets them to default, for a precise increase.
    """
    current_rate = df["default"].mean()
    target_rate = current_rate * (1 + increase_pct)
    n_total = len(df)
    n_target = int(n_total * target_rate)
    n_additional = n_target - int(df["default"].sum())

    non_defaults = df[df["default"] == 0].sort_values("score")
    additional_idx = non_defaults.index[:n_additional]

    stress_df = df.copy()
    stress_df["default_stress"] = stress_df["default"].copy()
    stress_df.loc[additional_idx, "default_stress"] = 1

    auc_original = roc_auc_score(df["default"], -df["score"])
    auc_stress = roc_auc_score(stress_df["default_stress"], -stress_df["score"])

    return {
        "auc_original": round(auc_original, 4),
        "auc_stress": round(auc_stress, 4),
        "original_rate": round(current_rate, 4),
        "stress_rate": round(stress_df["default_stress"].mean(), 4),
        "increase_pct": round((stress_df["default_stress"].mean() / current_rate - 1) * 100, 1),
    }


# ---------------------------------------------------------------------------
# 3. Calibration
# ---------------------------------------------------------------------------

def score_to_proba(score, Factor: float, Offset: float, intercept: float) -> np.ndarray:
    """Convert a score to default probability.

    Rationale: the scorecard distributes intercept and offset across
    variables, so the conversion is (2*Offset + Factor*intercept - score) / Factor.
    """
    log_odds = (2 * Offset + Factor * intercept - score) / Factor
    return 1 / (1 + np.exp(-log_odds))


def check_calibration(
    df: pd.DataFrame, Factor: float, Offset: float, intercept: float
) -> Dict[str, float]:
    """Check model calibration (Brier score)."""
    from sklearn.metrics import brier_score_loss
    proba = score_to_proba(df["score"], Factor, Offset, intercept)
    brier = brier_score_loss(df["default"], proba)

    return {
        "brier_score": round(brier, 4),
        "proba_mean": round(proba.mean(), 4),
        "default_rate": round(df["default"].mean(), 4),
        "ecart": round(abs(proba.mean() - df["default"].mean()), 4),
    }


# ---------------------------------------------------------------------------
# 4. Rank ordering
# ---------------------------------------------------------------------------

def check_rank_ordering(df: pd.DataFrame, n_bins: int = 10) -> Dict[str, Any]:
    """Check rank ordering (default rate decreasing with score)."""
    df = df.copy()
    df["bin"] = pd.qcut(df["score"], n_bins, labels=False, duplicates="drop")
    rank_order = df.groupby("bin")["default"].mean()

    is_monotonic = rank_order.is_monotonic_decreasing
    n_decreasing = (rank_order.diff().dropna() < 0).sum()

    return {
        "is_monotonic": bool(is_monotonic),
        "n_bins": len(rank_order),
        "n_decreasing": int(n_decreasing),
        "first_bin_rate": round(rank_order.iloc[0], 4),
        "last_bin_rate": round(rank_order.iloc[-1], 4),
    }


# ---------------------------------------------------------------------------
# 5. Leakage check
# ---------------------------------------------------------------------------

def check_leakage(
    train_df: pd.DataFrame, test_df: pd.DataFrame, iv_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Check leakage signals."""
    train_m = compute_metrics(train_df, "Train")
    test_m = compute_metrics(test_df, "Test")

    checks = {
        "auc_gap": round(train_m["AUC"] - test_m["AUC"], 4),
        "auc_gap_ok": (train_m["AUC"] - test_m["AUC"]) < 0.10,
    }

    if iv_df is not None:
        checks["max_iv"] = round(iv_df["IV"].max(), 4)
        checks["iv_ok"] = iv_df["IV"].max() < 0.5

    return checks


# ---------------------------------------------------------------------------
# 6. Override analysis
# ---------------------------------------------------------------------------

def override_analysis(
    df: pd.DataFrame, threshold: float, score_col: str = "score", target_col: str = "default"
) -> Dict[str, Any]:
    """Analyze overrides (model vs business disagreements).

    Rationale: overrides reveal model limitations and business rules
    not captured by the model. An override = a client the score rejects
    but business approves (or vice versa).
    """
    overrides = df[(df[score_col] < threshold) & (df[target_col] == 0)]
    return {
        "n_overrides": len(overrides),
        "override_rate": round(len(overrides) / len(df), 4),
        "overrides": overrides,
    }


# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------

def plot_calibration_curve(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    Factor: float,
    Offset: float,
    intercept: float,
    n_bins: int = 10,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """Plot calibration curve (train vs test)."""
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt
    from sklearn.metrics import brier_score_loss

    proba_train = score_to_proba(train_df["score"], Factor, Offset, intercept)
    proba_test = score_to_proba(test_df["score"], Factor, Offset, intercept)

    frac_pos_train, mean_pred_train = calibration_curve(
        train_df["default"], proba_train, n_bins=n_bins, strategy="quantile"
    )
    frac_pos_test, mean_pred_test = calibration_curve(
        test_df["default"], proba_test, n_bins=n_bins, strategy="quantile"
    )

    brier_train = brier_score_loss(train_df["default"], proba_train)
    brier_test = brier_score_loss(test_df["default"], proba_test)

    plt.figure(figsize=(8, 6))
    plt.plot(mean_pred_train, frac_pos_train, "s-", label=f"Train (Brier={brier_train:.4f})", color="blue")
    plt.plot(mean_pred_test, frac_pos_test, "o-", label=f"Test (Brier={brier_test:.4f})", color="red")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Actual default rate")
    plt.title("Calibration curve (train vs test)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return {"brier_train": brier_train, "brier_test": brier_test}


def plot_roc_curve(df: pd.DataFrame, save_path: Optional[str] = None) -> float:
    """Plot ROC curve."""
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(df["default"], -df["score"])
    auc = roc_auc_score(df["default"], -df["score"])

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, "b-", label=f"ROC (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return auc


def plot_score_distribution(
    train_df: pd.DataFrame, test_df: pd.DataFrame, save_path: Optional[str] = None
) -> None:
    """Plot score distribution (train vs test)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(train_df["score"], bins=30, alpha=0.7, color="blue", edgecolor="black")
    axes[0].set_title("Score distribution (train)")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(train_df["score"].mean(), color="red", linestyle="--",
                    label=f"Mean = {train_df['score'].mean():.0f}")
    axes[0].legend()

    axes[1].hist(test_df["score"], bins=30, alpha=0.7, color="red", edgecolor="black")
    axes[1].set_title("Score distribution (test)")
    axes[1].set_xlabel("Score")
    axes[1].set_ylabel("Frequency")
    axes[1].axvline(test_df["score"].mean(), color="blue", linestyle="--",
                    label=f"Mean = {test_df['score'].mean():.0f}")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_rank_ordering(
    df: pd.DataFrame, n_bins: int = 5, save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Plot rank ordering (default rate by score bin)."""
    import matplotlib.pyplot as plt

    df = df.copy()
    df["bin"] = pd.qcut(df["score"], n_bins, labels=False, duplicates="drop")
    rank_order = df.groupby("bin")["default"].mean()

    plt.figure(figsize=(8, 5))
    plt.bar(rank_order.index, rank_order.values, color="steelblue", alpha=0.7)
    plt.xlabel(f"Score bin (0=risky, {n_bins - 1}=safe)")
    plt.ylabel("Default rate")
    plt.title("Rank ordering (test)")
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    is_monotonic = rank_order.is_monotonic_decreasing
    return {"is_monotonic": bool(is_monotonic), "rank_order": rank_order}


# ---------------------------------------------------------------------------
# 8. Chargement des artefacts
# ---------------------------------------------------------------------------

def load_all_artifacts(
    base_path: str = "models", data_path: str = "data/processed"
) -> Dict[str, Any]:
    """Load all model artifacts for verification."""
    artifacts = {}
    artifacts["model"] = joblib.load(f"{base_path}/logistic_model.pkl")
    artifacts["scorecard"] = joblib.load(f"{base_path}/scorecard.pkl")
    artifacts["bp"] = joblib.load(f"{base_path}/binning_process.pkl")

    with open(f"{base_path}/selected_features.json") as f:
        artifacts["selected_features"] = json.load(f)
    with open(f"{base_path}/scaling_params.json") as f:
        artifacts["scaling_params"] = json.load(f)
    with open(f"{base_path}/cutoffs.json") as f:
        artifacts["cutoffs"] = json.load(f)

    artifacts["train_df"] = pd.read_parquet(f"{data_path}/train_scores_with_target.parquet")
    artifacts["test_df"] = pd.read_parquet(f"{data_path}/test_scores_with_target.parquet")

    return artifacts