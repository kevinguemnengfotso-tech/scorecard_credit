"""Pipeline complete du scorecard de credit."""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from woe import (
    load_data_from_db, split_data, multi_binning,
    select_features_by_iv, check_redundancy, calc_psi, interpret_psi,
    save_woe_artifacts, load_woe_artifacts,save_iv_selected_features
)
from model import (
    load_woe_data, check_vif, fit_logistic, fit_logistic_sklearn,
    backward_elimination, save_model, diagnose_model,
)
from scorecard import (
    load_phase7_artifacts, compute_scaling_params, extract_coefficients,
    build_scorecard, calculate_score, save_scorecard_artifacts,
    score_distribution, load_scaling_params,
)
from verification import (
    verify_metrics, check_calibration, check_rank_ordering,
    check_leakage, override_analysis, plot_calibration_curve,
    plot_roc_curve, plot_score_distribution, plot_rank_ordering,
    load_all_artifacts,
)

from data_loader import load_datas

def phase_1_to_5():
    print("=" * 60)
    print("PHASES 1-5 : EDA & Feature Engineering")
    print("=" * 60)
    import os
    print(os.getcwd())  # chemin absolu du dossier courant

    df = load_datas("data/raw/dataset_original.csv", db_path="data/processed/german_credit.db")
    


  #  df = load_data_from_db("../data/processed/german_credit.db")
    print(f"Raw data: {df.shape}")

    target_col = "res"
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"Train : {X_train.shape}, Val : {X_val.shape}, Test : {X_test.shape}")

    os.makedirs("../data/processed", exist_ok=True)
    X_train.to_parquet("../data/processed/X_train_raw.parquet")
    X_val.to_parquet("../data/processed/X_val_raw.parquet")
    X_test.to_parquet("../data/processed/X_test_raw.parquet")
    y_train.to_frame("res").to_parquet("../data/processed/y_train.parquet")
    y_val.to_frame("res").to_parquet("../data/processed/y_val.parquet")
    y_test.to_frame("res").to_parquet("../data/processed/y_test.parquet")

    print("Phases 1-5 completed")
    return X_train, X_val, X_test, y_train, y_val, y_test


def phase_6(X_train=None, X_val=None, X_test=None, y_train=None):
    print("=" * 60)
    print("PHASE 6 : Optimal binning & WoE")
    print("=" * 60)

    if X_train is None:
        X_train = pd.read_parquet("../data/processed/X_train_raw.parquet")
        X_val = pd.read_parquet("../data/processed/X_val_raw.parquet")
        X_test = pd.read_parquet("../data/processed/X_test_raw.parquet")
        y_train = pd.read_parquet("../data/processed/y_train.parquet").squeeze()
        y_val = pd.read_parquet("../data/processed/y_val.parquet").squeeze()
        y_test = pd.read_parquet("../data/processed/y_test.parquet").squeeze()

    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

    df_woe_train, df_woe_test, df_woe_val, bp = multi_binning(
        X_train, y_train, X_test, X_val, cat_cols
    )
    print(df_woe_train.shape)
    print("\n \n")
    selected, summary = select_features_by_iv(bp, threshold=0.02)
    print(f"Selected by IV: {selected}")
    print(f'Number of selected variables: {len(selected)}')
    save_iv_selected_features(selected)

    redundancy = check_redundancy(df_woe_train, selected)
    print(f"VIF max : {redundancy['vif_df']['VIF'].max():.2f}")

    save_woe_artifacts(bp, df_woe_train, df_woe_test, df_woe_val)

    print("Phase 6 completed")
    return bp, df_woe_train, df_woe_test, df_woe_val


def phase_7():
    print("=" * 60)
    print("PHASE 7 : Model building")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_woe_data()

    with open("../models/iv_selected_features.json","r") as f:
        selected_feat=json.load(f)

    X_train=X_train[selected_feat]
    X_test=X_test[selected_feat]
    


    print(f"New Xtrain : {X_train}")
    vif_df, to_drop = check_vif(X_train)
    print(f'TO DROP : {to_drop}')
    if to_drop:
        X_train = X_train.drop(columns=to_drop)
        X_test = X_test.drop(columns=to_drop)
        print(f"Variables dropped (VIF): {to_drop}")
    print(f'SHAPE XTRAIN : {X_train.shape}')
    print(X_train)
    print('\n \n \n')
    model_sm = fit_logistic(X_train, y_train)

    selected_features, final_model = backward_elimination(X_train, y_train)
    print(f"Variables retenues : {selected_features}")

    model_sk = fit_logistic_sklearn(X_train, y_train, selected_features)

    diag = diagnose_model(final_model, selected_features)
    print(diag.to_string())

    save_model(model_sk, selected_features)

    print("Phase 7 completed")
    return model_sk, selected_features, final_model


def phase_8():
    print("=" * 60)
    print("PHASE 8 : Scorecard scaling")
    print("=" * 60)

    model, selected_features, bp, X_train_woe, X_test_woe, y_train, y_test = load_phase7_artifacts(base_path='models/')

    Factor, Offset, scaling_params = compute_scaling_params(PDO=20, base_score=600, base_odds=20)
    coef_dict, intercept = extract_coefficients(model, selected_features)

    scorecard = build_scorecard(bp, selected_features, coef_dict, Factor, Offset, intercept)
    print(f"Scorecard : {len(scorecard)} bins")

    train_scores = calculate_score(X_train_woe, selected_features, scorecard, Factor, Offset, intercept)
    test_scores = calculate_score(X_test_woe, selected_features, scorecard, Factor, Offset, intercept)

    print(f"Train score - min: {train_scores.min():.1f}, max: {train_scores.max():.1f}, mean: {train_scores.mean():.1f}")
    print(f"Test score - min: {test_scores.min():.1f}, max: {test_scores.max():.1f}, mean: {test_scores.mean():.1f}")

    score_distribution(train_scores, label="train")
    score_distribution(test_scores, label="test")

    scaling_params["intercept"] = intercept
    save_scorecard_artifacts(scorecard, scaling_params, train_scores, test_scores)

    print("Phase 8 completed")
    return scorecard, scaling_params, train_scores, test_scores


def phase_9():
    """Phase 9 : validation statistique complete."""
    print("=" * 60)
    print("PHASE 9 : Validation")
    print("=" * 60)

    artifacts = load_all_artifacts()
    train_df = artifacts["train_df"]
    test_df = artifacts["test_df"]
    params = artifacts["scaling_params"]

    results = verify_metrics(train_df, test_df)
    print(f"Train : AUC={results['train']['AUC']}, Gini={results['train']['Gini']}, KS={results['train']['KS']}")
    print(f"Test  : AUC={results['test']['AUC']}, Gini={results['test']['Gini']}, KS={results['test']['KS']}")
    print(f"PSI   : {results['PSI']}")

    cal = check_calibration(test_df, params["Factor"], params["Offset"], params["intercept"])
    print(f"Calibration : Brier={cal['brier_score']}, proba_mean={cal['proba_mean']}, default_rate={cal['default_rate']}")

    ro = check_rank_ordering(test_df)
    print(f"Rank ordering : monotone={ro['is_monotonic']}, bins={ro['n_bins']}, decreasing={ro['n_decreasing']}")

    leakage = check_leakage(train_df, test_df)
    print(f"Leakage : auc_gap={leakage['auc_gap']}, auc_gap_ok={leakage['auc_gap_ok']}")

    overrides = override_analysis(test_df, threshold=950)
    print(f"Overrides : {overrides['n_overrides']} ({overrides['override_rate']:.1%})")

    import os
    os.makedirs("../models", exist_ok=True)
    plot_calibration_curve(train_df, test_df, params["Factor"], params["Offset"], params["intercept"],
                           save_path="../models/calibration_curve.png")
    plot_roc_curve(test_df, save_path="../models/roc_curve.png")
    plot_score_distribution(train_df, test_df, save_path="../models/score_distribution.png")
    plot_rank_ordering(test_df, n_bins=5, save_path="../models/rank_ordering.png")

    print("Phase 9 completed")
    return results

def run_pipeline(start_phase=1):
    if start_phase <= 5:
        X_train, X_val, X_test, y_train, y_val, y_test = phase_1_to_5()
        phase_6(X_train, X_val, X_test, y_train)
    elif start_phase == 6:
        phase_6()
    if start_phase <= 7:
        phase_7()
    if start_phase <= 8:
        phase_8()
    if start_phase <= 9:
        phase_9()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline scorecard credit")
    parser.add_argument("--phase", type=int, default=1, help="Phase de depart (1-9)")
    args = parser.parse_args()
    run_pipeline(start_phase=args.phase)