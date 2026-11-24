"""Credit scorecard package.

Provides utilities for credit risk scorecard development:
- data_loader: data loading and SQLite management
- woe: binning, WoE transformation, IV selection
- model: logistic regression, feature selection
- scorecard: scorecard scaling and scoring
- verification: metrics, stress testing, calibration
- pipeline: end-to-end pipeline orchestration
"""

__version__ = "1.0.0"
__author__ = "Kevin GUEMNENG FOTSO"