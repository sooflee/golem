"""
Measure per-match randomness from 50 years of real results.

We model a match as two independent Poisson goal counts whose means depend on
the win-expectancy implied by the Elo gap (scale-free, so the result transfers
to any Elo scale):

    we     = 1 / (1 + 10^(-dr/400))            # win expectancy from Elo gap
    lam_h  = BASE * exp( GAMMA * (we - 0.5) )   # home/stronger side's goals
    lam_a  = BASE * exp(-GAMMA * (we - 0.5) )   # away/weaker side's goals

GAMMA is the key knob: it says how strongly a rating edge turns into goals.
A *small* GAMMA means even big favorites barely outscore weak teams -> lots of
per-match randomness. A *large* GAMMA means strength dominates -> little
randomness. We fit GAMMA and BASE by maximum likelihood on the real results,
so the model's randomness matches football's actual randomness.

For a fixed GAMMA, the MLE for BASE is closed-form, so we only search GAMMA.
"""

import math

import histelo

# Only fit on recent football: matches in the last ~few decades, so the variance
# reflects the modern game rather than 1900s results. Change this to widen/narrow
# the window (older data = more matches but less representative of today).
MIN_YEAR = 2000

# Goal range used for analytic W/D/L probabilities (Poisson tail beyond ~12 is
# negligible for international football).
_MAXG = 14
_LOGFACT = [math.lgamma(k + 1) for k in range(_MAXG + 1)]


def _load():
    rows = list(histelo.reconstruct(min_year=MIN_YEAR))
    we_minus = []  # we - 0.5
    ga, gb = [], []
    for r in rows:
        we = 1.0 / (1.0 + 10 ** (-r["diff"] / 400.0))
        we_minus.append(we - 0.5)
        ga.append(r["ga"])
        gb.append(r["gb"])
    return we_minus, ga, gb


def _base_for_gamma(gamma, we_minus, ga, gb):
    """Closed-form Poisson MLE for BASE given GAMMA."""
    sum_goals = 0
    sum_means = 0.0
    for wm, x, y in zip(we_minus, ga, gb):
        m_h = math.exp(gamma * wm)
        m_a = math.exp(-gamma * wm)
        sum_goals += x + y
        sum_means += m_h + m_a
    return sum_goals / sum_means


def _loglik(gamma, base, we_minus, ga, gb):
    ll = 0.0
    for wm, x, y in zip(we_minus, ga, gb):
        lam_h = base * math.exp(gamma * wm)
        lam_a = base * math.exp(-gamma * wm)
        ll += (-lam_h + x * math.log(lam_h) - _LOGFACT[min(x, _MAXG)])
        ll += (-lam_a + y * math.log(lam_a) - _LOGFACT[min(y, _MAXG)])
    return ll


def fit():
    we_minus, ga, gb = _load()
    n = len(ga)

    # 1-D golden-ish search over GAMMA (BASE closed-form each time).
    best = None
    for g10 in range(5, 61):  # gamma 0.5 .. 6.0
        gamma = g10 / 10.0
        base = _base_for_gamma(gamma, we_minus, ga, gb)
        ll = _loglik(gamma, base, we_minus, ga, gb)
        if best is None or ll > best[2]:
            best = (gamma, base, ll)
    # Refine around the best gamma.
    g0 = best[0]
    for g100 in range(int((g0 - 0.1) * 100), int((g0 + 0.1) * 100) + 1):
        gamma = g100 / 100.0
        base = _base_for_gamma(gamma, we_minus, ga, gb)
        ll = _loglik(gamma, base, we_minus, ga, gb)
        if ll > best[2]:
            best = (gamma, base, ll)

    return best[0], best[1], n, (we_minus, ga, gb)


# ---- reporting: turn the fit into "how much randomness" in plain terms ----

def _poisson_pmf(lam):
    return [math.exp(-lam + k * math.log(lam) - _LOGFACT[k])
            for k in range(_MAXG + 1)]


def _tau(x, y, lam_h, lam_a, rho):
    """Dixon-Coles low-score correction (rho<0 inflates draws)."""
    if x == 0 and y == 0:
        return 1.0 - lam_h * lam_a * rho
    if x == 0 and y == 1:
        return 1.0 + lam_h * rho
    if x == 1 and y == 0:
        return 1.0 + lam_a * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _wdl(lam_h, lam_a, rho=0.0):
    """Analytic P(home win), P(draw), P(away win), with optional DC correction."""
    ph, pa = _poisson_pmf(lam_h), _poisson_pmf(lam_a)
    win = draw = loss = z = 0.0
    for x in range(_MAXG + 1):
        for y in range(_MAXG + 1):
            p = ph[x] * pa[y]
            if rho:
                p *= _tau(x, y, lam_h, lam_a, rho)
            z += p
            if x > y:
                win += p
            elif x == y:
                draw += p
            else:
                loss += p
    return win / z, draw / z, loss / z


def home_advantage_fit(gamma, base, rho):
    """Estimate the home-field advantage in Elo points by maximum likelihood.

    We rebuild venue-agnostic ratings (home_adv=0) so the home edge isn't already
    baked into them, then on non-neutral matches find the Elo bump H that, added
    to the home side, best predicts the actual scorelines. Returns (H, n_matches).
    This H is what host nations get in the 2026 sim (engine.HOME_ADVANTAGE_ELO)."""
    rows = []
    for r in histelo.replay(home_adv=0):
        if r["settled"] and not r["neutral"] and int(r["date"][:4]) >= MIN_YEAR:
            rows.append((r["diff"], r["ga"], r["gb"]))

    def loglik(h):
        s = 0.0
        for diff, x, y in rows:
            we = 1.0 / (1.0 + 10 ** (-(diff + h) / 400.0))
            wm = we - 0.5
            la, lb = base * math.exp(gamma * wm), base * math.exp(-gamma * wm)
            s += -la + x * math.log(la) - _LOGFACT[min(x, _MAXG)]
            s += -lb + y * math.log(lb) - _LOGFACT[min(y, _MAXG)]
        return s

    best = max(range(0, 161, 5), key=loglik)
    best = max(range(max(0, best - 5), best + 6), key=loglik)
    return best, len(rows)


def fit_rho(gamma, base, data, target_draw):
    """Find the DC rho that makes the model's overall draw rate match reality."""
    we_minus = data[0]
    n = len(we_minus)

    def model_draw(rho):
        s = 0.0
        for wm in we_minus:
            s += _wdl(base * math.exp(gamma * wm),
                      base * math.exp(-gamma * wm), rho)[1]
        return s / n

    lo, hi = -0.20, 0.0  # rho is negative to add draws
    for _ in range(40):
        mid = (lo + hi) / 2
        if model_draw(mid) < target_draw:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def report(gamma, base, rho, data):
    we_minus, ga, gb = data
    n = len(ga)
    print(f"\nFitted on {n:,} international matches ({MIN_YEAR}+).")
    print(f"  BASE  (even-match goals per team) = {base:.3f}")
    print(f"  GAMMA (strength->goals)           = {gamma:.3f}")
    print(f"  RHO   (Dixon-Coles draw boost)    = {rho:.4f}")

    print("\nWhat the fit says about per-match randomness:")
    for label, we in [("Coin-flip   (50/50)", 0.50),
                      ("Slight edge (60%)", 0.60),
                      ("Clear fav   (70%)", 0.70),
                      ("Big fav     (80%)", 0.80),
                      ("Huge fav    (90%)", 0.90)]:
        wm = we - 0.5
        lam_h = base * math.exp(gamma * wm)
        lam_a = base * math.exp(-gamma * wm)
        w, d, l = _wdl(lam_h, lam_a, rho)
        print(f"  {label}: avg score {lam_h:.2f}-{lam_a:.2f}  ->  "
              f"win {100*w:4.1f}% / draw {100*d:4.1f}% / lose {100*l:4.1f}%")

    # Calibration: predicted vs actual outcomes, bucketed by win-expectancy.
    print("\nCalibration check (model vs reality, by rating edge):")
    print(f"  {'edge bucket':<16}{'n':>6}{'pred W/D/L':>22}{'actual W/D/L':>22}")
    buckets = [(0.50, 0.55), (0.55, 0.62), (0.62, 0.70), (0.70, 0.80),
               (0.80, 1.01)]
    for lo, hi in buckets:
        idx = [i for i, wm in enumerate(we_minus) if lo <= wm + 0.5 < hi]
        if not idx:
            continue
        pw = pd = pl = 0.0
        aw = ad = al = 0
        for i in idx:
            wm = we_minus[i]
            w, d, l = _wdl(base * math.exp(gamma * wm),
                           base * math.exp(-gamma * wm), rho)
            pw += w
            pd += d
            pl += l
            if ga[i] > gb[i]:
                aw += 1
            elif ga[i] == gb[i]:
                ad += 1
            else:
                al += 1
        m = len(idx)
        pred = f"{100*pw/m:4.0f}/{100*pd/m:4.0f}/{100*pl/m:4.0f}"
        act = f"{100*aw/m:4.0f}/{100*ad/m:4.0f}/{100*al/m:4.0f}"
        print(f"  {lo:.2f}-{hi:.2f}      {m:>6}{pred:>22}{act:>22}")

    # Overall draw rate, now with the DC correction applied.
    pred_draw = sum(_wdl(base * math.exp(gamma * wm),
                         base * math.exp(-gamma * wm), rho)[1]
                    for wm in we_minus) / n
    act_draw = sum(1 for x, y in zip(ga, gb) if x == y) / n
    print(f"\n  Overall draw rate: model {100*pred_draw:.1f}%  vs  "
          f"actual {100*act_draw:.1f}%  (DC-corrected)")
    print("\nFitted constants are in engine.py (BASE_GOALS / ELO_SUPREMACY / "
          "DRAW_RHO). After changing them, run `python3 run.py --recalibrate`.")


if __name__ == "__main__":
    print("Fitting per-match randomness to historical results...")
    gamma, base, n, data = fit()
    act_draw = sum(1 for x, y in zip(data[1], data[2]) if x == y) / n
    rho = fit_rho(gamma, base, data, act_draw)
    report(gamma, base, rho, data)
