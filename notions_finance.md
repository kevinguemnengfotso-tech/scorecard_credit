Voici les connaissances en finance, modèles et mathématiques nécessaires pour le projet P9 (moteur de pricing : Python/C++ pour options, Black-Scholes, Monte Carlo, Greeks, P&L/IPV/FairVA, dashboard). Je suis honnête et objectif sur ce qui est indispensable vs nice-to-have.

---

## 1. Finance de marché — Produits et concepts

### 1.1 Produits dérivés (obligatoire)

| Produit | Définition | Pourquoi c'est indispensable |
|---|---|---|
| **Option call / put** | Droit (mais pas obligation) d'acheter/vendre un sous-jacent à un prix fixé (strike) à une date donnée (maturité). | C'est le produit de base du projet. Tout le pricing tourne autour des options. |
| **Option européenne** | Exerçable uniquement à la maturité. | Plus simple à pricer (Black-Scholes exact). Point de départ. |
| **Option américaine** | Exerçable à tout moment avant la maturité. | Plus complexe (Monte Carlo ou arbres binomiaux). Mentionner comme extension. |
| **Option asiatique** | Payoff dépend de la moyenne du sous-jacent sur la période. | Plus complexe (Monte Carlo). Mentionner comme extension. |
| **Option barrière** | Activée/désactivée si le sous-jacent franchit un seuil. | Plus complexe. Mentionner comme extension. |
| **Forward / Future** | Engagement d'acheter/vendre un sous-jacent à une date future à un prix fixé. | Produit de base, plus simple que l'option. Utile pour comprendre le pricing. |
| **Swap** | Échange de flux financiers (taux fixe vs variable, par exemple). | Mentionner comme contexte (Natixis pricing de swaps). |

### 1.2 Concepts de marché (obligatoire)

| Concept | Définition | Pourquoi c'est indispensable |
|---|---|---|
| **Sous-jacent (underlying)** | L'actif sur lequel le dérivé est basé (action, indice, taux, devise, matière première). | C'est la variable S dans Black-Scholes. |
| **Strike (K)** | Prix d'exercice de l'option. | Paramètre clé du payoff. |
| **Maturité (T)** | Date d'expiration de l'option (en années). | Paramètre clé du pricing. |
| **Volatilité (σ)** | Écart-type annualisé des rendements du sous-jacent. Mesure le risque. | Paramètre clé de Black-Scholes. C'est la "difficulté" du pricing. |
| **Taux sans risque (r)** | Taux d'intérêt théoriquement sans risque (ex : OAT, Bund). | Paramètre clé de Black-Scholes. |
| **Dividendes (q)** | Rendement du dividende du sous-jacent. | Ajustement du pricing pour les actions qui versent des dividendes. |
| **Payoff** | Gain à la maturité de l'option. Call : max(S_T - K, 0). Put : max(K - S_T, 0). | C'est ce qu'on price. |
| **Prix de l'option (V)** | Valeur actuelle du payoff espéré. | C'est la sortie du modèle. |
| **Hedging** | Couvrir le risque d'une position en prenant la position opposée. | Les Greeks servent au hedging. |
| **Arbitrage** | Profiter d'une incohérence de prix sans risque. | Fondement de la théorie du pricing (pas d'arbitrage = prix unique). |

### 1.3 Concepts de valorisation (obligatoire)

| Concept | Définition | Pourquoi c'est indispensable |
|---|---|---|
| **Valorisation (pricing)** | Calculer la valeur actuelle d'un instrument financier. | C'est le cœur du projet. |
| **P&L (Profit & Loss)** | Gain ou perte sur une position. | Mesurer la performance du portefeuille. |
| **IPV (Independent Price Verification)** | Vérification indépendante des prix par une équipe différente de celle qui les calcule. | Mentionné dans Natixis Valuation Group. |
| **FairVA (Fair Value Adjustment)** | Ajustement de juste valeur pour refléter le risque de crédit, la liquidité, etc. | Mentionné dans Natixis Valuation Group. |
| **xVA (Valuation Adjustments)** | Ajustements de valorisation : CVA (Credit Valuation Adjustment), DVA (Debit Valuation Adjustment), FVA (Funding Valuation Adjustment), MVA (Margin Valuation Adjustment), KVA (Capital Valuation Adjustment). | Pour aller plus loin — mentionner dans le rapport. |
| **Courbe de taux (yield curve)** | Relation entre le taux d'intérêt et la maturité. | Utilisé pour le discounting. |
| **Discounting** | Actualiser un flux futur à sa valeur présente. | Fondement du pricing. |

---

## 2. Mathématiques — Probabilités et stochastique

### 2.1 Probabilités (obligatoire)

| Concept | Définition | Pourquoi c'est indispensable |
|---|---|---|
| **Espérance (E[X])** | Valeur moyenne d'une variable aléatoire. | Le prix de l'option = espérance du payoff actualisé. |
| **Variance (Var[X])** | Mesure de la dispersion. | Utilisé pour la volatilité. |
| **Distribution normale (Gaussienne)** | Distribution en cloche. N(μ, σ²). | Black-Scholes suppose que les rendements sont normalement distribués. |
| **Distribution log-normale** | Si X est normal, alors e^X est log-normal. | Le prix du sous-jacent suit une distribution log-normale dans Black-Scholes. |
| **Probabilité conditionnelle** | P(A|B) = P(A∩B) / P(B). | Utilisé dans les formules de pricing conditionnel. |
| **Espérance conditionnelle** | E[X|Y]. | Utilisé dans le pricing des options asiatiques et barrières. |

### 2.2 Calcul stochastique (obligatoire pour comprendre, pas pour coder)

| Concept | Définition | Pourquoi c'est indispensable |
|---|---|---|
| **Mouvement brownien (W_t)** | Processus stochastique avec accroissements indépendants, normalement distribués. | Modélise le prix du sous-jacent dans Black-Scholes. |
| **Mouvement brownien géométrique (GBM)** | dS_t = μ S_t dt + σ S_t dW_t. Le prix du sous-jacent suit un GBM. | Équation différentielle stochastique de base. |
| **Lemme d'Itô** | Règle de dérivation pour les processus stochastiques. df = (∂f/∂t + μ S ∂f/∂S + ½ σ² S² ∂²f/∂S²) dt + σ S ∂f/∂S dW. | Fondement de Black-Scholes. |
| **Mesure de probabilité risque-neutre (Q)** | Mesure sous laquelle le prix actualisé du sous-jacent est une martingale. | Le pricing se fait sous la mesure risque-neutre (pas sous la mesure réelle P). |
| **Martingale** | Processus dont l'espérance future conditionnelle est égale à la valeur actuelle. | Fondement du pricing : le prix = espérance risque-neutre du payoff actualisé. |

> **Honnêteté** : tu n'as pas besoin de maîtriser le calcul stochastique pour coder Black-Scholes ou Monte Carlo. Tu dois juste **comprendre les concepts** (GBM, mesure risque-neutre, martingale) pour savoir ce que tu fais. Le code est simple : c'est des formules.

### 2.3 Intégration et dérivation (obligatoire pour comprendre)

| Concept | Définition | Pourquoi c'est indispensable |
|---|---|---|
| **Dérivée partielle (∂V/∂S, ∂V/∂t)** | Taux de variation de la valeur de l'option par rapport au sous-jacent ou au temps. | C'est les Greeks (Delta, Theta). |
| **Équation aux dérivées partielles (EDP)** | Équation qui relie les dérivées partielles. | L'équation de Black-Scholes est une EDP. |
| **Intégrale (E[X])** | Aire sous la courbe. | Le prix = intégrale du payoff sous la mesure risque-neutre. |

---

## 3. Modèles de pricing

### 3.1 Black-Scholes (obligatoire)

**Formule** :
```
C = S₀ × N(d₁) - K × e^(-rT) × N(d₂)
P = K × e^(-rT) × N(-d₂) - S₀ × N(-d₁)

d₁ = [ln(S₀/K) + (r - q + σ²/2) × T] / (σ × √T)
d₂ = d₁ - σ × √T
```

Où :
- S₀ = prix du sous-jacent
- K = strike
- r = taux sans risque
- q = dividende
- σ = volatilité
- T = maturité (en années)
- N(x) = fonction de répartition de la normale standard

**Hypothèses** :
1. Le sous-jacent suit un GBM (mouvement brownien géométrique).
2. Pas d'arbitrage (marché complet).
3. Pas de coûts de transaction.
4. Volatilité constante.
5. Taux sans risque constant.
6. Pas de dividendes (ou dividendes continus).

**Limites** :
- Volatilité constante (en réalité, la volatilité varie → smile de volatilité).
- Pas de sauts (le sous-jacent peut sauter → modèles à sauts).
- Marché complet (en réalité, certains risques ne sont pas couvrables).

**Qui l'utilise** : Tout le monde. C'est le modèle de base. Même si les banques utilisent des modèles plus avancés, Black-Scholes reste la référence.

**Offres qui le demandent** : Natixis (IT Quant Equity Pricing), Natixis (Développeur Python Finance), Forvis Mazars (pricing de dérivés), HSBC (produits structurés).

### 3.2 Monte Carlo (obligatoire)

**Principe** : simuler de nombreux chemins du sous-jacent (selon le GBM), calculer le payoff pour chaque chemin, puis faire la moyenne.

```
S_T = S₀ × exp((r - q - σ²/2) × T + σ × √T × Z)
```
Où Z ~ N(0,1).

**Prix = e^(-rT) × (1/N) × Σ payoff(S_T^i)**

**Quand l'utiliser** :
- Options asiatiques (payoff dépend de la moyenne)
- Options barrières (payoff dépend du chemin)
- Options américaines (avec Longstaff-Schwartz)
- Produits exotiques

**Avantages** : flexible, peut pricer n'importe quel payoff.
**Inconvénients** : lent, erreur statistique (convergence en 1/√N), difficile pour les options américaines.

**Code** :
```python
import numpy as np
N = 100000  # nombre de simulations
Z = np.random.standard_normal(N)
S_T = S0 * np.exp((r - q - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
payoff = np.maximum(S_T - K, 0)  # call
price = np.exp(-r*T) * np.mean(payoff)
```

### 3.3 Arbres binomiaux (optionnel — pour les options américaines)

**Principe** : discrétiser le temps en étapes, et à chaque étape, le sous-jacent peut monter ou descendre. On calcule le payoff à la maturité, puis on remonte l'arbre en actualisant.

**Quand l'utiliser** : options américaines (exercice anticipé), options avec dividendes discrets.

**Inconvénients** : lent pour beaucoup d'étapes, difficile en haute dimension.

### 3.4 Méthodes de différences finies (optionnel — pour les EDP)

**Principe** : résoudre l'équation de Black-Scholes (EDP) numériquement sur une grille.

**Quand l'utiliser** : options américaines, options avec barrières, produits exotiques.

**Inconvénients** : complexe à implémenter, lent en haute dimension.

---

## 4. Les Greeks (obligatoire)

Les **Greeks** sont les dérivées partielles du prix de l'option par rapport aux paramètres. Ils mesurent la sensibilité du prix.

| Greek | Définition | Formule (Black-Scholes call) | Interprétation | Utilisation |
|---|---|---|---|---|
| **Delta (Δ)** | ∂V/∂S | N(d₁) | Sensibilité au prix du sous-jacent. | Hedging : acheter/vendre Δ actions pour être neutre. |
| **Gamma (Γ)** | ∂²V/∂S² | N'(d₁) / (S × σ × √T) | Sensibilité du Delta au prix du sous-jacent. | Risque de convexité. |
| **Theta (Θ)** | ∂V/∂t | -[S × N'(d₁) × σ / (2√T)] - r × K × e^(-rT) × N(d₂) | Sensibilité au temps (décroissance temporelle). | Perte de valeur avec le temps. |
| **Vega (ν)** | ∂V/∂σ | S × √T × N'(d₁) | Sensibilité à la volatilité. | Risque de volatilité. |
| **Rho (ρ)** | ∂V/∂r | K × T × e^(-rT) × N(d₂) | Sensibilité au taux sans risque. | Risque de taux. |

**Pourquoi c'est indispensable** : les Greeks sont utilisés pour le **hedging** (couverture du risque) et la **gestion du risque**. Un desk de trading utilise les Greeks pour savoir combien d'actions acheter/vendre pour être neutre au risque.

**Code** :
```python
from scipy.stats import norm
import numpy as np

def black_scholes_greeks(S, K, r, sigma, T, q=0):
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    delta = np.exp(-q*T) * norm.cdf(d1)
    gamma = np.exp(-q*T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-S*np.exp(-q*T)*norm.pdf(d1)*sigma/(2*np.sqrt(T)) 
             + q*S*np.exp(-q*T)*norm.cdf(d1) 
             - r*K*np.exp(-r*T)*norm.cdf(d2))
    vega = S * np.exp(-q*T) * np.sqrt(T) * norm.pdf(d1)
    rho = K * T * np.exp(-r*T) * norm.cdf(d2)
    return delta, gamma, theta, vega, rho
```

---

## 5. Valuation Adjustments (xVA) — obligatoire pour comprendre, optionnel pour coder

### 5.1 CVA (Credit Valuation Adjustment)

**Définition** : ajustement du prix de l'option pour tenir compte du risque que la contrepartie fasse défaut.

```
CVA = E[ (1 - Recovery Rate) × Exposure × 1_{default} ]
```

**Pourquoi c'est indispensable** : les banques doivent valoriser leurs dérivés en tenant compte du risque de crédit de la contrepartie. C'est une exigence réglementaire (Bâle III).

**Offres qui le demandent** : Forvis Mazars ("calcul d'ajustements xVA"), Natixis (valuation group), SG (pricing de dérivés).

### 5.2 DVA (Debit Valuation Adjustment)

**Définition** : ajustement pour tenir compte du risque que la banque elle-même fasse défaut (symétrique du CVA).

### 5.3 FVA (Funding Valuation Adjustment)

**Définition** : ajustement pour tenir compte du coût de financement de la position.

### 5.4 xVA dans le contexte du projet

Pour P9 (moteur de pricing), tu n'as **pas besoin** de coder le CVA/DVA/FVA. Mais tu dois **comprendre le concept** et le **mentionner dans le rapport** comme perspective.

> **Honnêteté** : le CVA est un sujet à part entière. Coder un moteur de CVA nécessite des simulations Monte Carlo, des modèles de défaut, et des courbes de crédit. C'est un projet à part. Pour P9, mentionne-le comme "perspective" dans le rapport.

---

## 6. Ce que tu n'as PAS besoin de savoir (pour P9)

| Notion | Pourquoi tu peux l'ignorer pour P9 |
|---|---|
| **Calcul stochastique avancé** (Itô, Girsanov, martingales) | Tu dois comprendre les concepts (GBM, mesure risque-neutre) mais pas les démonstrations. Le code est simple : des formules. |
| **Modèles de volatilité** (Heston, SABR, local vol) | Black-Scholes suppose une volatilité constante. Mentionne le "smile de volatilité" comme limite mais ne l'implémente pas. |
| **Modèles à sauts** (Merton, Kou) | Pour les marchés avec sauts. Pas nécessaire pour P9. Mentionne comme perspective. |
| **Modèles de taux** (Hull-White, Vasicek, CIR) | Pour le pricing de produits de taux (swaps, bonds). Pas nécessaire pour P9 (options sur actions). |
| **Arbres binomiaux** | Pour les options américaines. Mentionne comme extension mais implémente Monte Carlo. |
| **Méthodes de différences finies** | Pour résoudre les EDP numériquement. Pas nécessaire pour P9. |
| **CVA/DVA/FVA en détail** | Mentionne comme perspective dans le rapport. Ne code pas. |
| **Marchés de taux, courbes de taux, bootstrapping** | Pour le discounting avancé. Pour P9, utilise un taux sans risque constant. |
| **Produits exotiques** (barrières, asiatiques, lookback) | Mentionne comme extension. Implémente les options européennes standard. |

---

## 7. Ressources de révision (par priorité)

### Priorité 1 — À lire/voir absolument (indispensable)

| Ressource | Type | Durée | Pourquoi |
|---|---|---|---|
| **"Options, Futures, and Other Derivatives"** (Hull, 11e édition) — Chapitres 1-5, 13-15 | Livre | ~200 pages | La bible du pricing. Chapitres 1-5 : introduction aux dérivés. Chapitres 13-15 : Black-Scholes, Greeks, Monte Carlo. |
| **"Paul Wilmott Introduces Quantitative Finance"** (Wilmott, 2007) | Livre | ~150 pages | Explications simples et intuitives. Parfait pour débuter. |
| **StatQuest (YouTube)** — "Black-Scholes", "Greeks", "Monte Carlo" | Vidéos | ~2h | Explications visuelles et simples. |
| **"Python for Finance"** (Yves Hilpisch, 2e édition) — Chapitres 1-5 | Livre | ~100 pages | Python appliqué à la finance (Black-Scholes, Monte Carlo, Greeks). |
| **Documentation scipy.stats** — `norm.cdf`, `norm.pdf` | Docs | ~1h | Référence technique pour Black-Scholes. |

### Priorité 2 — À lire si tu as le temps (important)

| Ressource | Type | Durée | Pourquoi |
|---|---|---|---|
| **"Stochastic Calculus for Finance I"** (Shreve, 2004) — Chapitres 1-4 | Livre | ~100 pages | Probabilités, mouvement brownien, GBM. Pour comprendre (pas pour coder). |
| **"Monte Carlo Methods in Financial Engineering"** (Glasserman, 2003) — Chapitres 1-4 | Livre | ~100 pages | Monte Carlo pour le pricing. |
| **"Volatility and Correlation"** (Rebonato, 2004) — Chapitres 1-3 | Livre | ~50 pages | Volatilité, smile de volatilité. |
| **QuantLib (documentation)** | Docs | ~2h | Librairie Python/C++ pour le pricing. Référence pour les modèles. |
| **"The Concepts and Practice of Mathematical Finance"** (Mark Joshi, 2008) — Chapitres 1-6 | Livre | ~100 pages | Mathématiques de la finance (Black-Scholes, Greeks, Monte Carlo). |

### Priorité 3 — Pour aller plus loin (avancé)

| Ressource | Type | Durée | Pourquoi |
|---|---|---|---|
| **"Stochastic Calculus for Finance II"** (Shreve, 2004) — Chapitres 1-5 | Livre | ~150 pages | Calcul stochastique avancé (Itô, Girsanov, martingales). |
| **"Monte Carlo Methods in Financial Engineering"** (Glasserman, 2003) — Chapitres 5-9 | Livre | ~150 pages | Monte Carlo avancé (variance reduction, quasi-Monte Carlo). |
| **"Counterparty Credit Risk and Credit Value Adjustment"** (Gregory, 2015) | Livre | ~100 pages | CVA/DVA/FVA en détail. |
| **"The Volatility Surface"** (Gatheral, 2006) | Livre | ~100 pages | Smile de volatilité, modèles de volatilité. |
| **QuantLib (code source)** | Code | ~5h | Librairie open source pour le pricing. Référence pour l'implémentation. |

---

## 8. Résumé : ce que tu dois vraiment retenir

**Pour faire P9, tu dois savoir :**

1. **Ce qu'est une option** (call/put, européenne/américaine, strike, maturité, payoff).
2. **Ce qu'est la volatilité** (σ) et le taux sans risque (r).
3. **Black-Scholes** : la formule, les hypothèses, les limites.
4. **Monte Carlo** : simuler des chemins, calculer le payoff, faire la moyenne.
5. **Les Greeks** (Delta, Gamma, Theta, Vega, Rho) et leur utilisation pour le hedging.
6. **Le concept de mesure risque-neutre** (le pricing se fait sous Q, pas sous P).
7. **Les xVA** (CVA, DVA, FVA) — au moins le concept, pas le code.
8. **Le hedging** (couvrir le risque avec le sous-jacent).

**Tu n'as PAS besoin de savoir :**

- Le calcul stochastique avancé (Itô, Girsanov, martingales) — comprendre les concepts suffit.
- Les modèles de volatilité (Heston, SABR) — mentionne le smile comme limite.
- Les modèles à sauts — mentionne comme perspective.
- Les modèles de taux (Hull-White, etc.) — pas pour les options sur actions.
- Les arbres binomiaux — mentionne comme extension.
- Les différences finies — pas nécessaire pour P9.
- Le CVA/DVA/FVA en détail — mentionne comme perspective.
- Les marchés de taux, courbes de taux, bootstrapping — pour le discounting avancé.
- Les produits exotiques (barrières, asiatiques, lookback) — mentionne comme extension.

**Temps de révision estimé** : ~30-40 heures de lecture (priorité 1 + priorité 2) suffisent pour avoir les notions nécessaires. Le reste s'apprend en faisant le projet.

---

*Note : ces notions sont celles demandées par les offres Natixis (IT Quant Equity Pricing, Développeur Python Finance, IT Quant Equity), Forvis Mazars (dérivés complexes, xVA, ML/DL), EY (pricing dérivés, xVAs, FRTB, SA-CCR), HSBC (produits structurés), SG (Fund Finance, Capital Call Financing), Amundi (structuration ETF & Indiciel), SCOR (réassurance, IFRS 17, Solvency II), Orange Lease (modélisation financière).*