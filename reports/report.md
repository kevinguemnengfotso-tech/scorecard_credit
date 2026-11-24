# Rapport technique — Scorecard de crédit (German Credit Dataset)

## 1. Résumé exécutif

Ce projet développe une scorecard de crédit complète sur le dataset German Credit (UCI).  
Le modèle obtient un **AUC de 0.8154** (test), un **Gini de 0.6309** et un **KS de 0.5524** — des performances excellentes pour ce dataset.

## 2. Données

- **Source** : German Credit Dataset (UCI Machine Learning Repository)
- **Taille** : 1000 observations (700 train, 150 val, 150 test)
- **Variables brutes** : 20 attributs (catégoriels et numériques)
- **Cible** : défaut (90+ jours de retard dans 12 mois) — taux de défaut = 30%
- **Limitations** : dataset petit, pas de dates, pas de données bureau externes

## 3. Méthodologie

- **Framework** : CRISP-DM
- **Binning** : optbinning (optimal, monotone, min_bin_size=0.05)
- **WoE** : ln(% Good / % Bad)
- **Modèle** : régression logistique (statsmodels pour p-values, sklearn pour production)
- **Sélection** : IV ≥ 0.05 → backward elimination (p < 0.05)
- **Scaling** : PDO = 20, Factor = 28.85, Offset = 513.56

## 4. Variables retenues (10)

checking_status, duration, credit_history, purpose, credit_amount, savings, employment, installment_rate, personal_status_sex, other_installment_plans

## 5. Performance

| Métrique | Train | Test | Seuil |
|----------|-------|------|-------|
| AUC | 0.8377 | 0.8154 | > 0.70 |
| Gini | 0.6754 | 0.6309 | > 0.40 |
| KS | 0.5592 | 0.5524 | > 0.30 |
| Brier | 0.1475 | 0.1500 | < 0.20 |
| PSI | — | 0.0375 | < 0.25 |
| Écart AUC | 0.0223 | — | < 0.10 |

## 6. Hypothèses et limitations

- Dataset petit (1000 obs) → stabilité limitée
- Pas de validation temporelle (pas de dates dans le dataset)
- Variables sensibles (sexe, âge) conservées (Option B — analyse d'impact)
- Pas de données bureau externes
- Pas de stress testing macroéconomique

## 7. Gouvernance

- **Développé par** : Data Science Team
- **Framework** : CRISP-DM + méthodologie scorecard (Siddiqi)
- **Prochaine revue** : à définir