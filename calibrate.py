"""
Market calibration — the accuracy upgrade.

Plain Elo gives reasonable match dynamics but its title odds won't match the
betting market, which is the best available forecast. Here we adjust each team's
rating until the simulation's championship probabilities reproduce the market's
de-vigged title odds.

Method (fixed-point iteration):
  1. Simulate the tournament with current ratings -> model title prob per team.
  2. For each team, nudge its rating by  lr * ln(market_prob / model_prob).
     Under-rated teams (market > model) move up; over-rated teams move down.
  3. Repeat. Title odds depend only on *relative* ratings, so this converges to
     a rating vector whose simulated title odds match the market.

The calibrated ratings keep Elo's full match-level behavior (scorelines, upsets,
home advantage) but are anchored to market-grade title probabilities, and the
simulation then yields per-round odds the market doesn't publish.
"""

import json
import math
import os

import data
import forecast
import market
import montecarlo

_DIR = os.path.dirname(__file__)

# Laplace smoothing so a team with zero simulated titles doesn't blow up the log.
_SMOOTH = 0.5


def _model_title_probs(elo, sims, seed):
    counts, n = montecarlo.run(sims, seed, elo=elo)
    return {t: (counts[t][5] + _SMOOTH) / (n + _SMOOTH) for t in counts}


def calibrate(target=None, iterations=24, sims=25000, lr=60.0, max_step=60.0,
              avg_last=6, seed=0, verbose=True):
    """Return ratings whose simulated title odds match `target` (a {team: prob}
    forecast; defaults to the recommended market+Elo ensemble).

    Convergence aids: the learning rate decays over iterations, and the final
    ratings are a Polyak average of the last `avg_last` iterations to cancel
    Monte Carlo noise. `max_step` clamps per-iteration Elo change."""
    if target is None:
        target = forecast.build_target(verbose=verbose)
    ratings = dict(data.ELO)
    history = []

    for it in range(iterations):
        model = _model_title_probs(ratings, sims, seed + it)
        err = sum(abs(model[t] - target.get(t, 0.0)) for t in model)
        if verbose:
            print(f"  iter {it + 1:2d}/{iterations}  "
                  f"total abs error = {err:.4f}")
        lr_t = lr / (1.0 + 0.15 * it)  # decay to damp oscillation
        for t, pt in target.items():
            pm = model.get(t, _SMOOTH / sims)
            step = lr_t * math.log(max(pt, 1e-6) / max(pm, 1e-6))
            step = max(-max_step, min(max_step, step))
            ratings[t] += step
        if it >= iterations - avg_last:
            history.append(dict(ratings))

    # Polyak average of the last few iterations -> denoised final ratings.
    avg = {t: sum(h[t] for h in history) / len(history) for t in ratings}
    return avg


def load_or_calibrate(market_weight=0.85, bias_k=1.05, force=False, verbose=True,
                      **kw):
    """Load cached calibrated ratings for this config, or compute and cache."""
    tag = f"w{market_weight:.2f}_k{bias_k:.2f}"
    path = os.path.join(_DIR, f"calibrated_elo_{tag}.json")
    if not force and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    target = forecast.build_target(market_weight, bias_k, verbose=verbose)
    ratings = calibrate(target=target, verbose=verbose, **kw)
    with open(path, "w") as f:
        json.dump(ratings, f, indent=2, sort_keys=True)
    return ratings


def _report(ratings, target=None):
    target = target if target is not None else forecast.build_target()
    model = _model_title_probs(ratings, 60000, seed=999)
    print("\n  Calibrated vs target title odds (top 16):")
    print(f"  {'Team':<24}{'model':>8}{'target':>8}{'Elo':>8}")
    rows = sorted(target.items(), key=lambda x: -x[1])[:16]
    for t, pt in rows:
        print(f"  {t:<24}{100*model[t]:7.1f}%{100*pt:7.1f}%"
              f"{ratings[t]:8.0f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Calibrate ratings to a forecast")
    p.add_argument("--market-weight", type=float, default=0.7)
    p.add_argument("--fav-bias", type=float, default=1.05)
    args = p.parse_args()
    print(f"Calibrating (market_weight={args.market_weight}, "
          f"fav_bias={args.fav_bias})...")
    tgt = forecast.build_target(args.market_weight, args.fav_bias, verbose=True)
    r = load_or_calibrate(args.market_weight, args.fav_bias, force=True)
    _report(r, tgt)
