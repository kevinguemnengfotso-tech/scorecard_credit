# Credit Scorecard — German Credit Dataset

An end-to-end credit risk scorecard built on the German Credit dataset (UCI / Statlog).
The pipeline covers EDA, SQLite storage, optimal binning, WoE/IV selection, logistic regression,
scorecard scaling, statistical validation, **cost-sensitive decision making**, and a
**champion/challenger comparison** against two scikit-learn gradient boosting models.

> Headline result: test **AUC 0.8154**, **Gini 0.6309**, **PSI 0.0375**, and a portfolio
> **cost of 79 units** under the dataset's official 1:5 cost matrix — better than both challengers.

---

## Project Structure

```
.
├── src/                          # Reusable modules
│   ├── data_loader.py            # CSV -> SQLite loading, renaming, target creation
│   ├── woe.py                    # Optimal binning (optbinning), WoE, IV, PSI
│   ├── model.py                  # VIF, logistic regression (statsmodels + sklearn)
│   ├── scorecard.py              # PDO scaling, points, score computation
│   ├── verification.py           # AUC/Gini/KS, calibration, rank ordering, leakage
│   ├── cost_matrix.py            # UCI 1:5 cost matrix, break-even PD, portfolio cost
│   ├── decision.py               # Score <-> PD conversion, optimal score cutoff
│   ├── challenger.py             # Two sklearn gradient boosting challengers
│   └── pipeline.py               # Phases 1-9 orchestrator (--phase argument)
├── notebooks/
│   ├── EDA_test.ipynb            # Exploratory analysis
│   ├── 02_woe_iv.ipynb           # Binning, WoE, IV selection
│   ├── 03_model_building.ipynb   # Logistic regression, VIF, backward elimination
│   ├── 04_scorecard.ipynb        # Scaling + derived decision cutoff (cell 8.8)
│   ├── 05_cost_decision.ipynb    # Cost matrix, break-even PD, cost curve, stress test
│   ├── 06_challenger.ipynb       # Champion vs two challengers (financial comparison)
│   └── verification.ipynb        # Calibration, rank ordering, cutoffs, impact analysis
├── tests/                        # 116 unit tests
│   ├── test_data_loader.py
│   ├── test_woe.py
│   ├── test_model.py
│   ├── test_scorecard.py
│   ├── test_verification.py
│   ├── test_cost_matrix.py
│   ├── test_decision.py
│   └── test_challenger.py
├── data/                         # Raw and processed data (gitignored)
├── models/                       # Saved artifacts (gitignored for .pkl)
├── reports/
│   ├── report.md                 # Technical report (French)
│   └── figures/                  # EDA figures
├── .github/workflows/ci.yml      # GitHub Actions CI
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the pipeline (phases 1 to 9)
python src/pipeline.py --phase 1

# 3. Run the full test suite
pytest tests/ -v
```

Notebooks are run from the `notebooks/` directory (they use `../src`, `../data`, `../models`).

---

## Pipeline Phases

| Phase | Description | Module |
|-------|-------------|--------|
| 1-5 | EDA, CSV loading, SQLite, train/val/test split | `data_loader.py` |
| 6 | Optimal binning, WoE transformation, IV selection, PSI | `woe.py` |
| 7 | Logistic regression, VIF, backward elimination | `model.py` |
| 8 | Scorecard scaling (PDO = 20, base score 600, base odds 20) | `scorecard.py` |
| 9 | Validation: AUC, Gini, KS, calibration, rank ordering, leakage | `verification.py` |

Business decision and challenger analysis are performed in notebooks **05** and **06**,
not in `pipeline.py`.

---

## Model Performance

| Metric | Train | Test | Threshold |
|--------|------:|-----:|----------:|
| AUC | 0.8377 | 0.8154 | > 0.70 |
| Gini | 0.6754 | 0.6309 | > 0.40 |
| KS | 0.5592 | 0.5524 | > 0.30 |
| PSI | — | 0.0375 | < 0.25 |

Selected variables (10): `checking_status`, `duration`, `credit_history`, `purpose`,
`credit_amount`, `savings`, `employment`, `installment_rate`, `personal_status_sex`,
`other_installment_plans`.

---

## Business Decision — Cost Matrix

The German Credit dataset provides an official asymmetric cost matrix:

|                       | Accept | Reject |
|-----------------------|:------:|:------:|
| **Good customer**     | 0      | 1      |
| **Bad customer**      | 5      | 0      |

Break-even rule: `5 * PD = 1 * (1 - PD)` → **PD < 1/6 (16.67%)**.

| Artifact | Value |
|---|---|
| Optimal PD cutoff | 0.1667 |
| Equivalent score cutoff | 1049.11 |
| Business baseline cutoffs | refus 950 / revue 1020 / approbation 1080 |

The derived optimum falls inside the manual "approbation" band — the business baselines are consistent.

---

## Champion vs Challengers

Both challengers are trained on the 700-row training set and evaluated on the 150-row test set
using the same PD cutoff (1/6).

| Model | AUC | Gini | Portfolio cost (units) |
|---|---:|---:|---:|
| **Scorecard (Champion)** | **0.8154** | **0.6309** | **79** |
| GradientBoostingClassifier (challenger) | 0.8022 | 0.6044 | 87 |
| HistGradientBoostingClassifier (challenger) | 0.7659 | 0.5319 | 98 |

**Verdict:** the interpretable scorecard outperforms both non-linear challengers on this dataset.
The challengers remain benchmarks; they are not deployed.

---

## Validation

- Calibration: Brier score and reliability diagram
- Rank ordering: monotonic decrease of default rate across score bins
- Leakage check: train/test AUC gap
- Override analysis: decisions deviating from the threshold
- PSI: 0.0375 (stable)
- Stress test: +20% defaults among accepted clients → cost 79 → 139 (**+75.9%**)

---

## Methodology

- **Framework**: CRISP-DM
- **Binning**: optimal binning (optbinning), monotone constraints
- **WoE**: `ln(% Good / % Bad)`
- **Model**: logistic regression (statsmodels for inference, sklearn for production)
- **Selection**: IV ≥ 0.02 → VIF → backward elimination (p < 0.05)
- **Scaling**: PDO = 20, Factor = 28.85, Offset = 513.56
- **Decision**: cost-sensitive threshold derived from the dataset's cost matrix

---

## Known Limitations

- Small dataset (1000 observations); no out-of-time split available (no dates)
- Reject inference not implemented (model trained on accepted applicants only)
- Sensitive attributes (sex, age) retained; covered by an impact analysis
- Lift figure in notebook 06 currently plots overlapping curves; to be consolidated
- CI workflow present but its lint path still refers to another project

---

## Author

Credit Risk Scorecard Project — individual academic work.
