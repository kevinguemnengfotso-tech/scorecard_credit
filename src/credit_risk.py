"""Expected Loss, IFRS 9 staging, adverse-action reasons and fairness checks.

German Credit has no collateral, no maturity ladder and no undrawn
commitments: LGD/EAD/CCF are therefore explicit assumptions, not estimates.
"""

import numpy as np
import pandas as pd


def expected_loss(pd_: np.ndarray, lgd: float, ead: np.ndarray) -> np.ndarray:
    """Calcule la perte attendue en euros : PD × LGD × EAD.

    Exemple : PD=0,02, LGD=0,45, EAD=10 000 -> EL = 90 EUR.
    """
    return np.asarray(pd_) * lgd * np.asarray(ead)


def stressed_lgd(lgd: float, factor: float = 1.25) -> float:
    """Aggrave la LGD en cas de récession (ex. 0,45 -> 0,5625), plafonné à 1."""
    return min(1.0, lgd * factor)


def ifrs9_stage(pd_12m: np.ndarray, pd_lifetime: np.ndarray,
                sicr_threshold: float = 2.0,
                default_flag: np.ndarray | None = None) -> np.ndarray:
    """Classe chaque client dans un stade comptable IFRS 9.

    1 = risque normal ; 2 = détérioration (SICR) ; 3 = défaut.
    La règle utilisée ici : lifetime PD > 2 × PD 12 mois -> stade 2.
    """
    stage = np.ones(len(pd_12m), dtype=int)
    stage[pd_lifetime > sicr_threshold * pd_12m] = 2
    if default_flag is not None:
        stage[np.asarray(default_flag) == 1] = 3
    return stage


def provision(pd_12m, pd_lifetime, lgd, ead, stage) -> np.ndarray:
    """Provision à mettre de côté : EL 12 mois en stade 1, EL lifetime sinon."""
    el_12 = expected_loss(pd_12m, lgd, ead)
    el_life = expected_loss(pd_lifetime, lgd, ead)
    return np.where(stage == 1, el_12, el_life)


def top_reason_codes(client_woe: np.ndarray, features: list,
                     scorecard_points: dict, n: int = 3) -> list:
    """Retourne les n variables qui ont coûté le plus de points à un client refusé.

    Scorecard_points : mapping {variable: points} issu de models/scorecard.csv.
    """
    contrib = sorted(
        [(var, float(scorecard_points.get(var, 0.0))) for var in features],
        key=lambda x: x[1],
    )
    return contrib[:n]



def adverse_impact_by_group(groups, y_true, approved) -> pd.DataFrame:
    """Per-group approval rate, default rate and cost.

    Cost per client: 5 if a bad client is accepted, 1 if a good client is rejected.
    """
    y = np.asarray(y_true)
    dec = np.asarray(approved, dtype=bool)
    g = np.asarray(groups)
    cost = 5 * ((y == 1) & dec) + ((y == 0) & ~dec)
    rows = []
    for grp in np.unique(g):
        m = g == grp
        rows.append({
            "group": int(grp) if isinstance(grp, (int, np.integer)) else str(grp),
            "n": int(m.sum()),
            "approval_rate": float(dec[m].mean()),
            "default_rate": float(y[m].mean()),
            "mean_cost": float(cost[m].mean()),
            "total_cost": float(cost[m].sum()),
        })
    return pd.DataFrame(rows).sort_values("mean_cost")


def reason_codes_for_client(client_woe, features: list,
                            scorecard_df: pd.DataFrame, n: int = 3) -> list:
    """Top n variables with the lowest points for one client.

    Uses the exact bin mapping from scorecard.csv (WoE rounded to 4 decimals),
    not a per-variable average.
    """
    rows = []
    for var in features:
        sub = scorecard_df[scorecard_df["variable"] == var]
        mapping = dict(zip(sub["woe"].round(4), sub["points"]))
        val = round(float(client_woe[var]), 4)
        rows.append((var, float(mapping.get(val, 0.0))))
    return sorted(rows, key=lambda x: x[1])[:n]
