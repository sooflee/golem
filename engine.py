"""
The match engine: turn two teams' ratings into a result.

Per-match randomness is no longer guessed — it's fit to ~31k real internationals
by fit_variance.py and the fitted constants are pasted in below. The model is
scale-free: it converts the Elo gap to a win-expectancy (the standard /400
logistic, identical across Elo scales) and maps that to two Poisson goal means.

    we     = 1 / (1 + 10^(-(ea-eb)/400))          # win expectancy
    lam_a  = BASE_GOALS * exp( ELO_SUPREMACY*(we-0.5))
    lam_b  = BASE_GOALS * exp(-ELO_SUPREMACY*(we-0.5))
    goals ~ Poisson, with a Dixon-Coles low-score correction (DRAW_RHO) so the
    draw rate matches reality.

Re-fit anytime with `python3 fit_variance.py` and update the three constants.
"""

import math

# Rating bump for a host nation (USA / Canada / Mexico) in any match it plays.
# The full home-field edge fit from data is ~96 Elo (fit_variance.home_advantage_fit,
# 17,896 non-neutral internationals) — but that's qualifier-style home advantage;
# a World Cup host's edge is milder (neutral-ish crowds, opponents also far from
# home), so the default is a reduced 60. The report lets you toggle none/60/96.
HOME_ADVANTAGE_ELO = 60.0

# --- fitted by fit_variance.py on 24,787 internationals (2000+, recent only) ---
BASE_GOALS = 1.178       # expected goals per team in an evenly matched game
ELO_SUPREMACY = 1.980    # how strongly a rating edge converts into goals
DRAW_RHO = -0.149        # Dixon-Coles draw correction (negative => more draws)


def adjusted_elo(team, elo, hosts):
    """A team's effective Elo, including home advantage if it's a host."""
    base = elo[team]
    return base + HOME_ADVANTAGE_ELO if team in hosts else base


def win_expectancy(elo_a, elo_b):
    """Probability team A beats B given adjusted Elos (standard logistic)."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def _poisson(lam, rng):
    """Sample from a Poisson(lam) distribution (Knuth's algorithm)."""
    target = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= target:
            return k - 1


def _tau(x, y, lam_a, lam_b):
    """Dixon-Coles low-score correction factor (only differs from 1 for 0/1)."""
    if x == 0 and y == 0:
        return 1.0 - lam_a * lam_b * DRAW_RHO
    if x == 0 and y == 1:
        return 1.0 + lam_a * DRAW_RHO
    if x == 1 and y == 0:
        return 1.0 + lam_b * DRAW_RHO
    if x == 1 and y == 1:
        return 1.0 - DRAW_RHO
    return 1.0


def _lambdas(ea, eb):
    wm = win_expectancy(ea, eb) - 0.5
    return (BASE_GOALS * math.exp(ELO_SUPREMACY * wm),
            BASE_GOALS * math.exp(-ELO_SUPREMACY * wm))


def match_goals(ea, eb, rng):
    """Simulate a scoreline from two already-adjusted Elos. Returns (ga, gb).
    Uses Dixon-Coles-corrected Poisson via rejection sampling."""
    lam_a, lam_b = _lambdas(ea, eb)
    if DRAW_RHO == 0.0:
        return _poisson(lam_a, rng), _poisson(lam_b, rng)
    # Upper bound on tau for rejection sampling (tau>1 only at (0,0) and (1,1)).
    ceil = max(1.0, 1.0 - DRAW_RHO, 1.0 - lam_a * lam_b * DRAW_RHO)
    while True:
        x = _poisson(lam_a, rng)
        y = _poisson(lam_b, rng)
        if rng.random() * ceil <= _tau(x, y, lam_a, lam_b):
            return x, y


def knockout_winner(team_a, ea, team_b, eb, rng):
    """Play a knockout match; a draw is resolved by extra time / penalties,
    which we model as a coin flip weighted by win expectancy."""
    ga, gb = match_goals(ea, eb, rng)
    if ga > gb:
        return team_a
    if gb > ga:
        return team_b
    return team_a if rng.random() < win_expectancy(ea, eb) else team_b
