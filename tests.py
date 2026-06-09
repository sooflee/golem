#!/usr/bin/env python3
"""
Sanity tests — guard against silent breakage (e.g. a bad data edit).

Dependency-free; run with `python3 tests.py`. Exits non-zero on failure.
"""

import random

import backtest
import data
import engine
import market
import montecarlo
import tournament

PASS = 0


def check(name, cond):
    global PASS
    if not cond:
        raise AssertionError(f"FAILED: {name}")
    PASS += 1


def test_data():
    data.validate()
    check("12 groups", len(data.GROUPS) == 12)
    check("4 teams per group", all(len(v) == 4 for v in data.GROUPS.values()))
    teams = [t for g in data.GROUPS.values() for t in g]
    check("48 distinct teams", len(set(teams)) == 48)
    check("every team rated", all(t in data.ELO for t in teams))
    check("hosts are real teams", data.HOSTS <= set(teams))


def test_engine():
    check("win_exp in (0,1)", 0 < engine.win_expectancy(2100, 1500) < 1)
    check("win_exp symmetric",
          abs(engine.win_expectancy(1800, 1600)
              + engine.win_expectancy(1600, 1800) - 1) < 1e-9)
    check("win_exp even = .5", abs(engine.win_expectancy(1700, 1700) - .5) < 1e-9)
    rng = random.Random(1)
    gs = [engine.match_goals(1950, 1600, rng) for _ in range(2000)]
    check("goals are non-negative ints",
          all(isinstance(a, int) and isinstance(b, int) and a >= 0 and b >= 0
              for a, b in gs))
    check("favorite outscores underdog on average",
          sum(a for a, b in gs) > sum(b for a, b in gs))


def test_market():
    for src in market.SOURCES:
        p = market.implied_probabilities(source=src)
        check(f"{src}: sums to 1", abs(sum(p.values()) - 1) < 1e-6)
        check(f"{src}: 48 teams", len(p) == 48)
        check(f"{src}: all positive", all(v > 0 for v in p.values()))
        check(f"{src}: keys match data", set(p) == set(data.ELO))
        check(f"{src}: fav-bias raises favorite",
              market.implied_probabilities(1.1, src)[max(p, key=p.get)]
              > p[max(p, key=p.get)])


def test_tournament():
    rng = random.Random(2)
    res = tournament.simulate(rng=rng)
    check("champion is a real team", res["champion"] in data.ELO)
    check("reached covers all 48", len(res["reached"]) == 48)
    check("exactly 32 reach the R32",
          sum(1 for v in res["reached"].values() if v >= 1) == 32)
    check("30 knockout matches recorded", len(res["won"]) == 30)
    check("champion reached the top round", res["reached"][res["champion"]] == 6)
    d1 = tournament.simulate(deterministic=True)["champion"]
    d2 = tournament.simulate(deterministic=True)["champion"]
    check("deterministic run is stable", d1 == d2)


def test_montecarlo():
    counts, n = montecarlo.run(3000, seed=0)
    rows = montecarlo.probabilities(counts, n)
    check("48 teams in output", len(rows) == 48)
    check("champion probs sum to ~1",
          abs(sum(r["Champion"] for r in rows) - 1) < 0.02)
    check("rounds are monotonic (R16>=QF>=SF>=Final>=Champ)",
          all(r["R16"] + 1e-9 >= r["QF"] >= r["SF"] >= r["Final"] >= r["Champion"]
              for r in rows))
    check("all probabilities in [0,1]",
          all(0 <= r[k] <= 1 for r in rows
              for k in ("R16", "QF", "SF", "Final", "Champion")))


def test_backtest():
    m = backtest.wc_matches()
    check("backtest has matches", len(m) > 100)
    ll, brier, acc = backtest.score(m, engine.ELO_SUPREMACY,
                                    engine.BASE_GOALS, engine.DRAW_RHO)
    base = backtest.baseline_logloss(m)
    check("log-loss is sane", 0.5 < ll < 1.3)
    check("model beats baseline", ll < base)
    check("favorite accuracy is sane", 0.5 < acc < 0.9)


if __name__ == "__main__":
    for fn in (test_data, test_engine, test_market, test_tournament,
               test_montecarlo, test_backtest):
        fn()
        print(f"  {fn.__name__} ok")
    print(f"\nALL {PASS} CHECKS PASSED")
