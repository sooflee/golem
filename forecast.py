"""
Build the calibration target by blending independent forecasts (the ensemble).

The most reliable accuracy gain isn't a better single model — it's combining
independent forecasts so their errors partly cancel. Here we blend:

  - the betting market (optionally favorite-longshot corrected), and
  - an independent pure-Elo Monte Carlo forecast.

market_weight = 1.0  -> pure market (what the market thinks)
market_weight = 0.0  -> pure ratings model (Goldman-style, ignores the market)
market_weight ~ 0.7  -> a sensible ensemble (recommended default)

NOTE on the weight: the *right* blend weight should be chosen by backtesting
(see the planned backtest harness), not by eye. 0.7 is a defensible prior that
leans on the market (historically the best single forecast) while letting an
independent model pull mispriced teams toward their rating-implied strength.
"""

import data
import market
import montecarlo


def elo_title_probs(sims=20000, seed=12345):
    """Independent title probabilities from a pure-Elo Monte Carlo."""
    counts, n = montecarlo.run(sims, seed, elo=data.ELO)
    return {t: counts[t][5] / n for t in counts}


def build_target(market_weight=0.85, bias_k=1.05, elo_probs=None, verbose=False):
    """Return the de-vigged, bias-corrected, market+Elo blended target probs."""
    m = market.implied_probabilities(bias_k=bias_k)
    if market_weight >= 0.999:
        return m
    if elo_probs is None:
        if verbose:
            print("  running independent Elo forecast for the ensemble...")
        elo_probs = elo_title_probs()
    teams = set(m) | set(elo_probs)
    blended = {
        t: market_weight * m.get(t, 0.0)
        + (1.0 - market_weight) * elo_probs.get(t, 0.0)
        for t in teams
    }
    s = sum(blended.values())
    return {t: p / s for t, p in blended.items()}


if __name__ == "__main__":
    tgt = build_target(verbose=True)
    print("\nEnsemble target title odds (top 12):")
    for t, p in sorted(tgt.items(), key=lambda x: -x[1])[:12]:
        print(f"  {t:<24}{100 * p:5.1f}%")
