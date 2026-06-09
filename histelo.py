"""
Reconstruct each team's World Football Elo *before* every historical match.

We run the standard eloratings.net algorithm forward over ~50 years of results
(data_hist/results.csv, 49k internationals). This gives us, for every match, the
pre-match rating of both sides — the input we need to measure how much an Elo gap
actually predicts a result (see fit_variance.py).

Elo update:  R' = R + K * (W - We)
  We = 1 / (1 + 10^(-dr/400)),  dr = (elo_home + HOME_ADV) - elo_away
  W  = 1 win / 0.5 draw / 0 loss
  K  = importance_weight * goal_difference_multiplier
"""

import csv
import os

HOME_ADV = 100.0          # eloratings.net home-advantage, skipped if neutral
INIT_RATING = 1500.0
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "data_hist", "results.csv")


def _importance(tournament):
    """Base K-factor by match importance (eloratings.net scale)."""
    t = tournament.lower()
    if "friendly" in t:
        return 20.0
    if "world cup" in t and "qual" not in t:
        return 60.0
    if "qualif" in t:
        return 40.0
    # Continental finals & major confederation tournaments.
    majors = ("uefa euro", "copa am", "african cup", "africa cup",
              "afc asian cup", "gold cup", "confederations", "nations league",
              "oceania nations", "uefa nations")
    if any(m in t for m in majors):
        return 50.0
    return 30.0  # other competitive matches


def _gd_multiplier(goal_diff):
    """eloratings.net goal-difference weighting."""
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    if goal_diff == 3:
        return 1.75
    return 1.75 + (goal_diff - 3) / 8.0


def _expected(dr):
    return 1.0 / (1.0 + 10 ** (-dr / 400.0))


def replay(min_history=10, home_adv=HOME_ADV):
    """Replay every match in chronological order, updating Elo as we go. Yields
    one record per match *before* that match updates the ratings:

        dict(date, tournament, neutral, home, away, ga, gb, diff, settled)

    where `diff` is the home side's adjusted Elo minus the away side's (exactly
    the rating gap our match model is given), and `settled` is True once both
    teams have played >= min_history prior matches (ratings stabilized).

    home_adv is the Elo bump given to the (non-neutral) home team. Pass 0 to get
    venue-agnostic ratings/diffs (used to *estimate* home advantage itself)."""
    ratings = {}
    played = {}  # team -> count of prior matches

    with open(RESULTS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            home, away = row["home_team"], row["away_team"]
            try:
                ga, gb = int(row["home_score"]), int(row["away_score"])
            except (ValueError, KeyError):
                continue  # unplayed / missing score
            neutral = row["neutral"].strip().upper() == "TRUE"

            rh = ratings.get(home, INIT_RATING)
            ra = ratings.get(away, INIT_RATING)
            dr = (rh + (0.0 if neutral else home_adv)) - ra

            yield {"date": row["date"], "tournament": row["tournament"],
                   "neutral": neutral, "home": home, "away": away,
                   "ga": ga, "gb": gb, "diff": dr,
                   "settled": (played.get(home, 0) >= min_history
                               and played.get(away, 0) >= min_history)}

            # Elo update.
            w = 1.0 if ga > gb else (0.0 if ga < gb else 0.5)
            k = _importance(row["tournament"]) * _gd_multiplier(abs(ga - gb))
            delta = k * (w - _expected(dr))
            ratings[home] = rh + delta
            ratings[away] = ra - delta
            played[home] = played.get(home, 0) + 1
            played[away] = played.get(away, 0) + 1


def reconstruct(min_year=1990, min_history=10):
    """Settled matches on/after min_year as {diff, ga, gb} — input to the
    variance fit (fit_variance.py)."""
    for r in replay(min_history):
        if r["settled"] and int(r["date"][:4]) >= min_year:
            yield {"diff": r["diff"], "ga": r["ga"], "gb": r["gb"]}


def current_ratings():
    """Replay everything and return final ratings (a check vs public tables)."""
    ratings = {}
    played = {}
    with open(RESULTS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            home, away = row["home_team"], row["away_team"]
            try:
                ga, gb = int(row["home_score"]), int(row["away_score"])
            except (ValueError, KeyError):
                continue
            neutral = row["neutral"].strip().upper() == "TRUE"
            rh = ratings.get(home, INIT_RATING)
            ra = ratings.get(away, INIT_RATING)
            dr = (rh + (0.0 if neutral else HOME_ADV)) - ra
            w = 1.0 if ga > gb else (0.0 if ga < gb else 0.5)
            k = _importance(row["tournament"]) * _gd_multiplier(abs(ga - gb))
            delta = k * (w - _expected(dr))
            ratings[home] = rh + delta
            ratings[away] = ra - delta
            played[home] = played.get(home, 0) + 1
            played[away] = played.get(away, 0) + 1
    return ratings


if __name__ == "__main__":
    n = sum(1 for _ in reconstruct())
    print(f"Built {n:,} training matches (1990+, settled teams).")
    r = current_ratings()
    print("\nTop 15 by reconstructed Elo (sanity check vs public tables):")
    for t, v in sorted(r.items(), key=lambda x: -x[1])[:15]:
        print(f"  {t:<22}{v:7.0f}")
