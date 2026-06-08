"""
Backtest: does the model actually predict real World Cup matches well?

We replay history to get each team's pre-match Elo, then take every World Cup
match (1990-2022) and score the model's predicted Win/Draw/Loss probabilities
against what really happened, using:

  - log-loss  (lower is better; punishes confident wrong calls)
  - Brier     (mean squared error of the probability vector)
  - accuracy  (did the favorite-by-probability win outright?)

Two things this validates:
  1. Calibration end-to-end on tournament football (not just all internationals).
  2. The fitted per-match variance: we sweep GAMMA and confirm the value fit in
     fit_variance.py (~1.99) also minimizes out-of-sample World Cup log-loss.
     If the best GAMMA on WC matches matches the one fit on all internationals,
     the variance isn't overfit — it generalizes to the matches we care about.

NOTE: this backtests the *ratings model* (Elo + fitted variance). It does not
test the market ensemble weight — that would need historical betting odds, which
we don't have. So it confirms the engine/variance, not --market-weight.
"""

import math

import engine
import fit_variance as fv
import histelo


def wc_matches(min_year=1990):
    """Settled World Cup matches (from min_year on) with pre-match Elo diff and
    90-minute result. We restrict to the modern era by default: early-history
    reconstructed Elo is thin, and 1930s-80s football isn't representative of
    how the 2026 model will be used."""
    out = []
    for r in histelo.replay():
        t = r["tournament"].lower()
        if (r["settled"] and "fifa world cup" in t and "qual" not in t
                and int(r["date"][:4]) >= min_year):
            out.append(r)
    return out


def _wdl(diff, gamma, base, rho):
    we = 1.0 / (1.0 + 10 ** (-diff / 400.0))
    wm = we - 0.5
    return fv._wdl(base * math.exp(gamma * wm),
                   base * math.exp(-gamma * wm), rho)


def _outcome(m):
    """0 = home win, 1 = draw, 2 = away win (90-minute result)."""
    if m["ga"] > m["gb"]:
        return 0
    return 1 if m["ga"] == m["gb"] else 2


def score(matches, gamma, base, rho):
    """Return (log_loss, brier, accuracy) for the model over `matches`."""
    ll = brier = 0.0
    correct = decisive = 0
    for m in matches:
        probs = _wdl(m["diff"], gamma, base, rho)
        y = _outcome(m)
        ll -= math.log(max(probs[y], 1e-12))
        brier += sum((probs[k] - (1.0 if k == y else 0.0)) ** 2 for k in range(3))
        # Accuracy on decisive games: did the more-likely side win?
        if probs[0] != probs[2]:
            pick = 0 if probs[0] > probs[2] else 2
            if y != 1:
                decisive += 1
                if pick == y:
                    correct += 1
    n = len(matches)
    return ll / n, brier / n, (correct / decisive if decisive else 0.0)


def baseline_logloss(matches):
    """Naive predictor: always forecast the dataset's base-rate W/D/L."""
    n = len(matches)
    counts = [0, 0, 0]
    for m in matches:
        counts[_outcome(m)] += 1
    rates = [c / n for c in counts]
    return -sum(math.log(max(rates[_outcome(m)], 1e-12)) for m in matches) / n


def calibration(matches, gamma, base, rho):
    """Predicted stronger-side win prob vs actual, bucketed."""
    buckets = [(0.33, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 0.75),
               (0.75, 1.01)]
    rows = []
    for lo, hi in buckets:
        sel = []
        for m in matches:
            w, d, l = _wdl(m["diff"], gamma, base, rho)
            pwin = max(w, l)  # stronger side's win prob
            if lo <= pwin < hi:
                # did the stronger side win?
                stronger_home = w >= l
                y = _outcome(m)
                won = (y == 0) if stronger_home else (y == 2)
                sel.append((pwin, won))
        if sel:
            rows.append((lo, hi, len(sel),
                         sum(p for p, _ in sel) / len(sel),
                         sum(1 for _, wnt in sel if wnt) / len(sel)))
    return rows


def main():
    matches = wc_matches()
    g, b, r = engine.ELO_SUPREMACY, engine.BASE_GOALS, engine.DRAW_RHO
    print(f"Backtesting on {len(matches):,} World Cup matches (1990-2022).\n")

    ll, brier, acc = score(matches, g, b, r)
    base_ll = baseline_logloss(matches)
    print(f"Model (fitted GAMMA={g}, BASE={b}, RHO={r}):")
    print(f"  log-loss   {ll:.4f}   (base-rate baseline {base_ll:.4f})")
    print(f"  Brier      {brier:.4f}")
    print(f"  favorite accuracy on decisive games  {100*acc:.1f}%")
    print(f"  log-loss improvement over baseline    "
          f"{100*(base_ll-ll)/base_ll:.1f}%")

    print("\nGAMMA sweep — out-of-sample log-loss on World Cup matches:")
    best = None
    for g10 in range(10, 36, 2):  # 1.0 .. 3.4
        gg = g10 / 10.0
        gl, _, _ = score(matches, gg, b, r)
        mark = ""
        if best is None or gl < best[1]:
            best = (gg, gl)
        print(f"  GAMMA {gg:.1f}   log-loss {gl:.4f}")
    print(f"  -> WC-optimal GAMMA ~ {best[0]:.1f}  "
          f"(fit on all internationals: {g})")

    print("\nCalibration (stronger side): predicted vs actual win rate")
    print(f"  {'bucket':<14}{'n':>6}{'pred':>8}{'actual':>8}")
    for lo, hi, n, pred, act in calibration(matches, g, b, r):
        print(f"  {lo:.2f}-{hi:.2f}  {n:>6}{100*pred:7.0f}%{100*act:7.0f}%")


if __name__ == "__main__":
    main()
