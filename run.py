#!/usr/bin/env python3
"""
2026 World Cup bracket predictor.

Usage:
    python run.py                 # Monte Carlo odds + favorites bracket
    python run.py --sims 100000   # more simulations (smoother odds)
    python run.py --seed 7        # reproducible run with a different seed
    python run.py --top 20        # show more teams in the odds table
    python run.py --bracket-only  # just the single most-likely (chalk) bracket
"""

import argparse

import calibrate
import data
import montecarlo
import tournament


def pct(x):
    return f"{100 * x:5.1f}%"


def print_odds(rows, top):
    print("\n=== Title & deep-run odds (Monte Carlo) ===\n")
    header = f"{'Team':<24}{'R16':>7}{'QF':>7}{'SF':>7}{'Final':>8}{'Champ':>8}"
    print(header)
    print("-" * len(header))
    for r in rows[:top]:
        print(f"{r['team']:<24}{pct(r['R16']):>7}{pct(r['QF']):>7}"
              f"{pct(r['SF']):>7}{pct(r['Final']):>8}{pct(r['Champion']):>8}")


def print_group_winners(rows_by_team):
    print("\n=== Most likely group winner / runner-up ===\n")
    for g, teams in data.GROUPS.items():
        ranked = sorted(teams, key=lambda t: rows_by_team[t]["R16"], reverse=True)
        adv = ", ".join(f"{t} ({pct(rows_by_team[t]['R16'])})" for t in ranked[:2])
        print(f"  Group {g}: {adv}")


def print_chalk_bracket(elo):
    """The single deterministic 'favorites always win' bracket — one concrete
    prediction you could fill in on a printed bracket."""
    res = tournament.simulate(elo=elo, deterministic=True)
    won = res["won"]
    print("\n=== Favorites bracket (higher-rated team wins every game) ===\n")

    print("Round of 16:")
    for m in (89, 90, 91, 92, 93, 94, 95, 96):
        f1, f2 = tournament.R16[m]
        print(f"  {won[f1]} vs {won[f2]}  ->  {won[m]}")

    print("\nQuarterfinals:")
    for m in (97, 98, 99, 100):
        f1, f2 = tournament.QF[m]
        print(f"  {won[f1]} vs {won[f2]}  ->  {won[m]}")

    print("\nSemifinals:")
    for m in (101, 102):
        f1, f2 = tournament.SF[m]
        print(f"  {won[f1]} vs {won[f2]}  ->  {won[m]}")

    print(f"\nFinal:  {res['runner_up']} vs {res['champion']}")
    print(f"\n>>> Predicted champion: {res['champion']} <<<")


def main():
    p = argparse.ArgumentParser(description="2026 World Cup bracket predictor")
    p.add_argument("--sims", type=int, default=50000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top", type=int, default=16)
    p.add_argument("--bracket-only", action="store_true")
    p.add_argument("--pure-elo", action="store_true",
                   help="use raw Elo instead of calibrated ratings")
    p.add_argument("--pure-market", action="store_true",
                   help="calibrate to the raw market only (no ensemble/bias)")
    p.add_argument("--market-weight", type=float, default=0.85,
                   help="ensemble weight on market vs Elo (1.0=market only)")
    p.add_argument("--fav-bias", type=float, default=1.05,
                   help="favorite-longshot correction exponent (1.0=off)")
    p.add_argument("--recalibrate", action="store_true",
                   help="recompute the calibration before running")
    args = p.parse_args()

    data.validate()

    if args.pure_elo:
        elo = data.ELO
        print("Ratings: raw Elo (uncalibrated).")
    else:
        mw, kb = (1.0, 1.0) if args.pure_market else (args.market_weight,
                                                      args.fav_bias)
        if args.recalibrate:
            print("Calibrating...")
        elo = calibrate.load_or_calibrate(mw, kb, force=args.recalibrate)
        if mw >= 0.999 and kb == 1.0:
            print("Ratings: calibrated to pure market.")
        else:
            print(f"Ratings: ensemble-calibrated "
                  f"({int(mw*100)}% market / {int((1-mw)*100)}% Elo, "
                  f"fav-bias {kb}).")

    if args.bracket_only:
        print_chalk_bracket(elo)
        return

    print(f"Simulating {args.sims:,} tournaments (seed={args.seed})...")
    counts, n = montecarlo.run(args.sims, args.seed, elo=elo)
    rows = montecarlo.probabilities(counts, n)
    rows_by_team = {r["team"]: r for r in rows}

    print_odds(rows, args.top)
    print_group_winners(rows_by_team)
    print_chalk_bracket(elo)


if __name__ == "__main__":
    main()
