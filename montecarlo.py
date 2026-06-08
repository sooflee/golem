"""
Monte Carlo driver: run the tournament many times and aggregate the odds.
"""

from random import Random

import data
import tournament
from tournament import ROUND_NAMES


def run(n_sims=50000, seed=0, elo=None):
    """Simulate n_sims tournaments. Returns per-team counts of how often each
    team reached each round (R32, R16, QF, SF, Final, Champion).

    Pass `elo` to simulate with custom ratings (e.g. market-calibrated ones)."""
    rng = Random(seed)
    teams = [t for grp in data.GROUPS.values() for t in grp]
    # counts[team] = [r32, r16, qf, sf, final, champ]  (cumulative: reaching SF
    # implies reaching QF, etc.)
    counts = {t: [0, 0, 0, 0, 0, 0] for t in teams}

    for _ in range(n_sims):
        result = tournament.simulate(elo=elo, rng=rng)
        for team, depth in result["reached"].items():
            # depth is index into ROUND_NAMES (0=Group ... 6=Champion).
            # Credit every knockout round up to the one reached.
            for r in range(1, min(depth, 6) + 1):
                counts[team][r - 1] += 1

    return counts, n_sims


def probabilities(counts, n_sims):
    """Convert raw counts to a sorted list of per-team probability rows."""
    rows = []
    for team, c in counts.items():
        rows.append({
            "team": team,
            "R32": c[0] / n_sims,
            "R16": c[1] / n_sims,
            "QF": c[2] / n_sims,
            "SF": c[3] / n_sims,
            "Final": c[4] / n_sims,
            "Champion": c[5] / n_sims,
        })
    rows.sort(key=lambda r: (r["Champion"], r["Final"], r["SF"]), reverse=True)
    return rows
