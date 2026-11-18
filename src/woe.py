"""WoE & IV utilities (Phase 6)."""

import numpy as np
import pandas as pd
import joblib
from optbinning import BinningProcess, OptimalBinning
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import VarianceThreshold
import json


def load_data_from_db(db_path="../data/processed/german_credit.db"):
    """Load data from the SQLite database."""
    from sqlalchemy import create_engine
    engine = create_engine(f"sqlite:///{db_path}")
    df = pd.read_sql_query("SELECT * FROM clients", engine)
    engine.dispose()
    return df


def split_data(X, y, test_size=0.3, val_size=0.5, random_state=42):
    """Stratified split into train / val / test."""
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def bin_single_variable(series, target, name, dtype="numerical", solver="cp",
                        monotonic_trend="auto", min_bin_size=0.05):
    """Bin a single variable with optbinning and return the binning table."""
    optb = OptimalBinning(
        name=name, dtype=dtype, solver=solver,
        monotonic_trend=monotonic_trend, min_bin_size=min_bin_size,
    )
    optb.fit(series, target)
    table = optb.binning_table.build()
    return optb, table


def multi_binning(X_train, y_train, X_test, X_val, cat_cols):
    """Bin all variables with BinningProcess and transform to WoE."""
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


def select_features_by_iv(binning_process, threshold=0.05):
    """Select features with IV >= threshold."""
    summary = binning_process.summary()
    selected = summary.loc[summary["iv"] >= threshold, "name"].tolist()
    return selected, summary

def select_features_by_variance(X): 
    selector = VarianceThreshold(threshold=0.05)  # variance < 1%
    selector.fit(X)
    cols_quasi_const = X.columns[~selector.get_support()]
    return 
def check_redundancy(df_woe, selected_cols, corr_threshold=0.5, vif_threshold=5):
    """Check correlation and VIF for redundant features."""
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


def calc_psi(expected, actual, bins=10):
    """Calculate Population Stability Index between two distributions."""
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    exp_counts = np.histogram(expected, bins=breakpoints)[0]
    act_counts = np.histogram(actual, bins=breakpoints)[0]

    exp_pct = (exp_counts + 0.5) / (exp_counts.sum() + 0.5 * len(exp_counts))
    act_pct = (act_counts + 0.5) / (act_counts.sum() + 0.5 * len(act_counts))

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)

def save_iv_selected_features(selected, base_path="../models"):
    """Sauvegarde la liste des features sélectionnées par IV."""
    with open(f"{base_path}/iv_selected_features.json", "w") as f:
        json.dump(selected, f)

def interpret_psi(psi_value):
    """Interpret PSI value into a stability category."""
    if psi_value < 0.1:
        return "Stable (pas de drift)"
    elif psi_value < 0.25:
        return "Drift modere (a surveiller)"
    else:
        return "Drift significatif (recalibration necessaire)"


def save_woe_artifacts(binning_process, df_woe_train, df_woe_test, df_woe_val=None,
                       base_path="../models", data_path="../data/processed"):
    """Save binning process and WoE-transformed datasets."""
    import os
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    joblib.dump(binning_process, f"{base_path}/binning_process.pkl")
    df_woe_train.to_parquet(f"{data_path}/X_train_woe.parquet")
    df_woe_test.to_parquet(f"{data_path}/X_test_woe.parquet")
    if df_woe_val is not None:
        df_woe_val.to_parquet(f"{data_path}/X_val_woe.parquet")

    print("Binning process et donnees WoE sauvegardes")


def load_woe_artifacts(base_path="../models", data_path="../data/processed"):
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