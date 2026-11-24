# Credit Scorecard — German Credit Dataset

A complete credit risk scorecard pipeline built on the German Credit dataset (UCI).  
Covers data loading, EDA, WoE/IV binning, logistic regression, scorecard scaling, validation, and business decision strategy.

## Project Structure

```
├── src/                    # Reusable Python modules
│   ├── __init__.py
│   ├── data_loader.py      # Data loading (CSV → SQLite)
│   ├── woe.py              # Binning, WoE, IV selection, PSI
│   ├── model.py            # Logistic regression, feature selection
│   ├── scorecard.py        # Scorecard scaling (PDO, points)
│   ├── verification.py     # Metrics, stress testing, calibration
│   └── pipeline.py         # End-to-end pipeline orchestrator
├── notebooks/              # Jupyter notebooks (exploration)
│   ├── 02_woe_iv.ipynb
│   ├── 03_model_building.ipynb
│   ├── 04_scorecard.ipynb
│   └── verification.ipynb
├── tests/                  # Unit tests
│   ├── test_data_loader.py
│   ├── test_woe.py
│   ├── test_model.py
│   ├── test_scorecard.py
│   └── test_verification.py
├── models/                 # Saved artifacts (gitignored)
├── data/                   # Raw and processed data (gitignored)
├── reports/                # Technical reports
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python src/pipeline.py

# 3. Run tests
pytest tests/ -v
```

## Pipeline Phases

| Phase | Description | Module |
|-------|-------------|--------|
| 1-5 | EDA & Feature Engineering | `data_loader.py` |
| 6 | Binning & WoE | `woe.py` |
| 7 | Model Building (Logistic Regression) | `model.py` |
| 8 | Scorecard Scaling (PDO, points) | `scorecard.py` |
| 9 | Validation (KS, AUC, Gini, PSI) | `verification.py` |
| 10 | Business Strategy (cut-offs, stress testing) | `pipeline.py` |

## Results

| Metric | Train | Test | Threshold |
|--------|-------|------|-----------|
| AUC | 0.8377 | 0.8154 | > 0.70 |
| Gini | 0.6754 | 0.6309 | > 0.40 |
| KS | 0.5592 | 0.5524 | > 0.30 |
| PSI | — | 0.0375 | < 0.25 |

## Methodology

- **Framework**: CRISP-DM
- **Binning**: Optimal binning (optbinning) with monotonicity constraints
- **WoE**: Weight of Evidence transformation
- **Model**: Logistic regression (interpretable, auditable, regulatory-compliant)
- **Selection**: IV ≥ 0.05 → backward elimination (p < 0.05)
- **Scaling**: PDO = 20, Factor = 28.85, Offset = 513.56

## Author

Data Science Team — Credit Risk Scorecard Project