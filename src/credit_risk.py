"""Expected Loss, IFRS 9 staging and adverse-action reasons.

German Credit has no collateral, no maturity ladder and no undrawn
commitments: LGD/EAD/CCF are therefore explicit assumptions, not estimates.
"""

import numpy as np


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