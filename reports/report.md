# Rapport technique — Scorecard de crédit (German Credit Dataset)

## 1. Résumé exécutif

Ce projet développe une scorecard de crédit complète sur le dataset German Credit (UCI / Statlog).
Le modèle obtient un **AUC de 0.8154**, un **Gini de 0.6309** et un **KS de 0.5524** sur le test set.

Au-delà de la performance statistique, la décision d'octroi est optimisée avec la **matrice de coût officielle**
du dataset (refuser un bon client = 1 unité, accepter un mauvais = 5 unités). Le seuil optimal en découle
mathématiquement : **PD < 1/6**, soit un score de **1049,11**.

Le scorecard est comparé à deux challengers sklearn. **Il reste champion sur les trois mesures**
(Gini et coût inclus), ce qui conforte le choix d'un modèle linéaire interprétable en contexte réglementé.

---

## 2. Données

- **Source** : German Credit Dataset (UCI Machine Learning Repository)
- **Taille** : 1000 observations — 700 train / 150 validation / 150 test
- **Variables brutes** : 20 attributs (catégoriels et numériques)
- **Cible** : défaut de paiement — taux de défaut = 30 %
- **Stockage** : CSV source → SQLite (`data/processed/german_credit.db`) → parquet WoE
- **Limitations** : dataset petit, aucune date, pas de données bureau externes

---

## 3. Méthodologie

- **Framework** : CRISP-DM
- **Binning** : optbinning (optimal, contrainte de monotonie, `min_bin_size=0.05`)
- **WoE** : `ln(% Good / % Bad)`
- **Modèle** : régression logistique (statsmodels pour les p-values, sklearn pour la production)
- **Sélection** : IV ≥ 0.02 → VIF → élimination ascendante (p < 0.05)
- **Scaling** : PDO = 20, base score = 600, base odds = 20 → Factor = 28.85, Offset = 513.56
- **Décision** : seuil dérivé de la matrice de coût (et non fixé manuellement)

---

## 4. Variables retenues (10)

`checking_status`, `duration`, `credit_history`, `purpose`, `credit_amount`, `savings`,
`employment`, `installment_rate`, `personal_status_sex`, `other_installment_plans`

---

## 5. Performance statistique

| Métrique | Train | Test | Seuil |
|----------|------:|-----:|------:|
| AUC | 0.8377 | 0.8154 | > 0.70 |
| Gini | 0.6754 | 0.6309 | > 0.40 |
| KS | 0.5592 | 0.5524 | > 0.30 |
| Brier | 0.1475 | 0.1500 | < 0.20 |
| PSI | — | 0.0375 | < 0.25 |
| Écart AUC | 0.0223 | — | < 0.10 |

---

## 6. Décision économique

### 6.1 Matrice de coût

|                        | Accepter | Refuser |
|------------------------|:--------:|:-------:|
| **Bon client**         | 0        | 1       |
| **Mauvais client**     | 5        | 0       |

### 6.2 Seuil optimal

```
cout(accepter) = 5 * PD
cout(refuser)  = 1 * (1 - PD)

5 * PD = 1 - PD   →   PD = 1/6 = 0.1667
```

Le score correspondant est **1049,11** (formule : `score = 2*Offset + Factor*intercept - Factor*logit`).

| Seuil | Valeur |
|---|---|
| PD optimal | 0,1667 |
| Score optimal | 1049,11 |
| Repères métier | refus 950 / revue 1020 / approbation 1080 |

Le seuil théorique se situe dans la bande « approbation » définie manuellement : les deux approches sont cohérentes.

### 6.3 Analyse d'impact (seuils métier)

| Cutoff | Taux d'approbation | Défaut chez les approuvés | Défauts évités |
|-------:|-------------------:|--------------------------:|---------------:|
| 950 | 98,7 % | 29,05 % | 2 |
| 1020 | 68,0 % | 13,73 % | 31 |
| 1080 | 20,7 % | 6,45 % | 43 |

La métrique « défauts évités » est volontairement conservée à titre de référence historique,
mais elle est **économiquement biaisée** (elle récompenserait un refus systématique).
Le coût total est la métrique de décision retenue.

---

## 7. Champion vs Challengers

Les deux challengers sont entraînés sur les 700 lignes du train et évalués sur les 150 lignes du test,
avec les mêmes 10 variables et le même seuil PD = 1/6.

| Modèle | AUC | Gini | Coût total | Taux d'approbation |
|---|---:|---:|---:|---:|
| **Scorecard (Champion)** | **0.8154** | **0.6309** | **79** | 45,3 % |
| GradientBoostingClassifier | 0.8022 | 0.6044 | 87 | 48,0 % |
| HistGradientBoostingClassifier | 0.7659 | 0.5319 | 98 | 56,7 % |

**Verdict.** Le champion domine sur les trois dimensions. Le meilleur challenger réduit l'écart
à 8 unités de coût, mais reste derrière. Aucun challenger ne justifie une perte d'interprétabilité
en environnement réglementé : ils servent de benchmarks.

 Figures produites : `models/roc_comparison.png`, `models/lift_comparison.png`,
`models/cost_curve.png`, `models/cost_matrix.png`.

---

## 8. Validation complémentaire

- **Calibration** : Brier score et courbe de fiabilité (`src/verification.py`)
- **Rank ordering** : décroissance monotone du taux de défaut par classe de score
- **Leakage** : écart AUC train/test = 0,0223 (acceptable)
- **Stabilité** : PSI global = 0,0375 ; PSI par variable disponible dans `src/woe.py`
- **Stress test** : +20 % de défauts parmi les clients acceptés → 12 défauts supplémentaires,
  coût de 79 à 139 unités (**+75,9 %**)
- **Overrides** : analyse des décisions s'écartant du seuil (`override_analysis`)

---

## 8.5 Adverse-impact et reason codes

Le modèle conserve `personal_status_sex` (variable UCI mêlant statut marital et sexe).
Une analyse par groupe a été ajoutée dans le notebook 07 (`src/credit_risk.py::adverse_impact_by_group`).

| Groupe | n | Taux d'approbation | Taux de défaut | Coût moyen |
|---|---:|---:|---:|---:|
| A91 (homme marié) | 3 | 33,3 % | 66,7 % | 0,000 |
| A92 (femme divorcée) | 54 | 40,7 % | 35,2 % | 0,463 |
| A93 (homme célibataire) | 80 | 50,0 % | 25,0 % | 0,550 |
| A94 (femme mariée) | 13 | 38,5 % | 30,8 % | 0,769 |

Écart maximal de coût moyen entre groupes : **0,769 unité**. Sur ce jeu de test,
l'écart reste modeste mais le groupe A91 n'a que 3 observations — résultat non significatif.
Le printemps suivant l'octroi devra faire l'objet d'un monitoring dédié avant toute mise en production.

**Reason codes.** Pour chaque refus, les trois variables ayant retiré le plus de points sont
extraites de `models/scorecard.csv` (mapping WoE → points par bin, arrondi à 4 décimales).
Exemples : `duration`, `credit_amount`, `credit_history` pour le client 915.
La fonction `reason_codes_for_client()` remplace l'ancienne approximation par moyenne de variable.

## 9. Hypothèses et limitations

- Dataset petit (1000 observations) → stabilité limitée
- Pas de validation temporelle (Out-of-Time) : le dataset ne contient pas de dates
- **Reject inference non implémentée** : le modèle est entraîné sur les seuls clients acceptés
- Variables sensibles (sexe, âge) conservées avec analyse d'impact
- Pas de scénarios macroéconomiques complets ; seul un choc sur le taux de défaut est simulé
- Le minimum empirique de la courbe de coût (PD ≈ 0,26 sur 150 clients) diffère de l'optimum
  théorique (1/6) ; le seuil théorique est retenu car plus stable
- Figure de lift à consolider (les deux courbes se superposent actuellement)
- Le workflow CI contient un chemin de lint hérité d'un autre projet (`pricing_engine/`)

---

## 10. Gouvernance

- **Développement** : projet individuel, démarche CRISP-DM et scorecard (Siddiqi)
- **Tests** : 116 tests unitaires, tous passing
- **Versionnage** : tags Git recommandés à partir de la baseline (`v1.0.0`)
- **Prochaine étape** : version v2 avec variables métier dérivées, comparée sur la matrice de coût
