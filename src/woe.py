"""WoE & IV utilities (Phase 6).

Functions for binning, WoE transformation, IV selection,
redundancy checks, and stability (PSI).
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from optbinning import BinningProcess, OptimalBinning
from statsmodels.stats.outliers_influence import variance_inflation_factor


def load_data_from_db(db_path: str = "../data/processed/german_credit.db") -> pd.DataFrame:
    """Load data from the SQLite database.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.

    Returns
    -------
    pd.DataFrame
        Full DataFrame with all columns.
    """
    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{db_path}")
    df = pd.read_sql_query("SELECT * FROM clients", engine)
    engine.dispose()
    return df


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    val_size: float = 0.5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified split into train / val / test.

    Rationale: stratification preserves the default rate in each split.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    from sklearn.model_selection import train_test_split

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def bin_single_variable(
    series: pd.Series,
    target: pd.Series,
    name: str,
    dtype: str = "numerical",
    solver: str = "cp",
    monotonic_trend: str = "auto",
    min_bin_size: float = 0.05,
) -> Tuple[OptimalBinning, pd.DataFrame]:
    """Bin a single variable with optbinning and return the binning table.

    Rationale: used for inspection and visualization of individual
    variables before the full pipeline.

    Returns
    -------
    optb : OptimalBinning
        Fitted binning object.
    table : pd.DataFrame
        Binning table with WoE and IV.
    """
    optb = OptimalBinning(
        name=name,
        dtype=dtype,
        solver=solver,
        monotonic_trend=monotonic_trend,
        min_bin_size=min_bin_size,
    )
    optb.fit(series, target)
    table = optb.binning_table.build()
    return optb, table


def multi_binning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    X_val: pd.DataFrame,
    cat_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, BinningProcess]:
    """Bin all variables with BinningProcess and transform to WoE.

    Rationale: BinningProcess handles all variables at once, ensuring
    consistent binning across train/val/test.

    Returns
    -------
    df_woe_train, df_woe_test, df_woe_val : pd.DataFrame
    bp : BinningProcess
    """
    bp = BinningProcess(
        variable_names=X_train.columns.tolist(),
        categorical_variables=cat_cols,
        min_bin_size=0.05,
    )
    bp.fit(X_train, y_train)
    df_woe_train = bp.transform(X_train, metric="woe")
    df_woe_test = bp.transform(X_test, metric="woe")
    df_woe_val = bp.transform(X_val, metric="woe")
    return df_woe_train, df_woe_test, df_woe_val, bp


def select_features_by_iv(
    binning_process: BinningProcess, threshold: float = 0.05
) -> Tuple[List[str], pd.DataFrame]:
    """Select features with IV >= threshold.

    Rationale: IV < 0.02 is useless. 0.05 is a reasonable cutoff.

    Returns
    -------
    selected : list of str
    summary : pd.DataFrame
    """
    summary = binning_process.summary()
    selected = summary.loc[summary["iv"] >= threshold, "name"].tolist()
    return selected, summary


def check_redundancy(
    df_woe: pd.DataFrame,
    selected_cols: List[str],
    corr_threshold: float = 0.5,
    vif_threshold: float = 5,
) -> Dict[str, pd.DataFrame]:
    """Check correlation and VIF for redundant features.

    Rationale: correlation catches pairwise redundancy,
    VIF catches multivariable redundancy. Both are needed.

    Returns
    -------
    dict with keys: corr_pairs, vif_df
    """
    df_sel = df_woe[selected_cols]
    corr_matrix = df_sel.corr()

    high_corr = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c_val = corr_matrix.iloc[i, j]
            if abs(c_val) >= corr_threshold:
                high_corr.append({
                    "var_1": cols[i],
                    "var_2": cols[j],
                    "corr": round(c_val, 3),
                })
    corr_pairs = pd.DataFrame(high_corr)

    X_vif = df_sel.assign(const=1.0)
    vif_df = pd.DataFrame({
        "variable": df_sel.columns,
        "VIF": [variance_inflation_factor(X_vif.values, i)
                for i in range(len(df_sel.columns))],
    }).sort_values("VIF", ascending=False)

    return {"corr_pairs": corr_pairs, "vif_df": vif_df}


def calc_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calculate Population Stability Index between two distributions.

    Rationale: PSI measures distribution drift. Uses quantile-based
    binning on expected, with smoothing to avoid log(0).

    Returns
    -------
    float : PSI value
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    exp_counts = np.histogram(expected, bins=breakpoints)[0]
    act_counts = np.histogram(actual, bins=breakpoints)[0]

    exp_pct = (exp_counts + 0.5) / (exp_counts.sum() + 0.5 * len(exp_counts))
    act_pct = (act_counts + 0.5) / (act_counts.sum() + 0.5 * len(act_counts))

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def interpret_psi(psi_value: float) -> str:
    """Interpret PSI value into a stability category."""
    if psi_value < 0.1:
        return "Stable (no drift)"
    elif psi_value < 0.25:
        return "Moderate drift (monitor)"
    else:
        return "Significant drift (recalibration needed)"


def save_woe_artifacts(
    binning_process: BinningProcess,
    df_woe_train: pd.DataFrame,
    df_woe_test: pd.DataFrame,
    df_woe_val: Optional[pd.DataFrame] = None,
    base_path: str = "../models",
    data_path: str = "../data/processed",
) -> None:
    """Save binning process and WoE-transformed datasets."""
    import os
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    joblib.dump(binning_process, f"{base_path}/binning_process.pkl")
    df_woe_train.to_parquet(f"{data_path}/X_train_woe.parquet")
    df_woe_test.to_parquet(f"{data_path}/X_test_woe.parquet")
    if df_woe_val is not None:
        df_woe_val.to_parquet(f"{data_path}/X_val_woe.parquet")

    print("Binning process and WoE data saved")


def load_woe_artifacts(
    base_path: str = "../models",
    data_path: str = "../data/processed",
) -> Tuple[BinningProcess, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Load binning process and WoE-transformed datasets."""
    bp = joblib.load(f"{base_path}/binning_process.pkl")
    X_train_woe = pd.read_parquet(f"{data_path}/X_train_woe.parquet")
    X_test_woe = pd.read_parquet(f"{data_path}/X_test_woe.parquet")
    X_val_woe = None
    try:
        X_val_woe = pd.read_parquet(f"{data_path}/X_val_woe.parquet")
    except FileNotFoundError:
        pass
    return bp, X_train_woe, X_test_woe, X_val_woe


def save_iv_selected_features(selected: List[str], base_path: str = "../models") -> None:
    """Save the list of IV-selected features."""
    import os
    os.makedirs(base_path, exist_ok=True)
    with open(f"{base_path}/iv_selected_features.json", "w") as f:
        import json
        json.dump(selected, f)
    print(f"IV-selected features saved: {len(selected)} features")