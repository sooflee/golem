# 2026 World Cup bracket predictor

A **market-calibrated, Elo-driven Monte Carlo** model for the 2026 FIFA World Cup
(48 teams, the new 12-group format with a Round of 32). Pure Python standard
library — no install.

By default it uses ratings calibrated to the betting market (the strongest
available forecast), so its title odds match the market while it also produces
the full per-round bracket probabilities the market doesn't publish.

## Run it

```bash
python3 run.py                  # ensemble-calibrated odds + favorites bracket
python3 run.py --sims 100000    # more sims = smoother odds
python3 run.py --pure-market    # calibrate to raw market only (no ensemble)
python3 run.py --pure-elo       # raw Elo, no calibration at all
python3 run.py --market-weight 0.5 --fav-bias 1.1   # tune the ensemble
python3 run.py --recalibrate    # recompute the calibration first
python3 run.py --top 24         # show more teams
python3 run.py --bracket-only   # just the single most-likely bracket
python3 calibrate.py            # (re)calibrate and show model-vs-target table
python3 forecast.py             # show the blended ensemble target odds
python3 market.py               # de-vigged market title odds + the vig
python3 fit_variance.py         # measure per-match randomness from real results
python3 backtest.py             # score the model on real World Cups (validation)
python3 report.py               # generate index.html for GitHub Pages
python3 histelo.py              # reconstruct historical Elo (sanity check)
python3 data.py                 # sanity-check the input data
```

### Publishing the report (GitHub Pages)

`python3 report.py` builds a self-contained `index.html` (inline CSS + JS, no
dependencies) with the live forecast, bracket, backtest results, and a "How it
works" methodology section. It includes an **interactive slider** that reweights
the forecast between the pure market and our pure Elo model: the page precomputes
a real model run at each blend weight and embeds them as JSON, so dragging the
slider shows genuine model output (not interpolation). First run is slower because
it calibrates each weight (cached per weight afterward). To publish:

```bash
python3 report.py                 # writes ./index.html
git add index.html && git commit -m "forecast" && git push
```

Then in the repo: **Settings -> Pages -> Deploy from branch -> main / root**.
(Or output to `docs/`: `python3 report.py --out docs/index.html` and select the
`/docs` folder.) Re-run before kickoff after refreshing odds to update the page.

Default ratings are an **ensemble** (85% market / 15% independent Elo) with a
mild favorite-longshot correction — the recommended "most accurate" config: it
leans on the betting market (the best-calibrated single forecast) and treats the
Elo model as a small hedge. `--market-weight 1.0 --fav-bias 1.0` reduces this to
the pure market; `--market-weight 0.0` is a pure ratings model (Goldman-style,
ignoring the market).

**For best accuracy, refresh the odds in `market.py` right before kickoff and
re-run with `--recalibrate`** — late injury/squad news is the biggest remaining
signal, and the market prices it in.

## How it works

1. **Ratings → match outcomes** (`engine.py`). The Elo gap sets a win
   expectancy (scale-free /400 logistic); that maps to two Poisson goal means,
   with a Dixon-Coles low-score correction so the draw rate is right. The
   per-match randomness here is **fit to real data** (see below), not guessed.
   Host nations (USA/Canada/Mexico) get a home-advantage bump.
2. **One tournament** (`tournament.py`). Round-robin groups → rank top 2 + the 8
   best third-placed teams → slot them into the official Round-of-32 bracket →
   single elimination to the Final.
3. **Many tournaments** (`montecarlo.py`). Repeat tens of thousands of times and
   count how often each team reaches each round. Those frequencies are the
   probabilities.

4. **Market calibration** (`market.py`, `calibrate.py`). Plain Elo gives decent
   match dynamics but its title odds drift from the betting market. We take the
   market's de-vigged outright odds and iteratively nudge each team's rating
   until the simulation's championship probabilities reproduce them. The result
   keeps Elo's match-level behavior but is anchored to market-grade title odds.
   Calibrated ratings are cached in `calibrated_elo.json`.

5. **Per-match randomness, fit to data** (`histelo.py`, `fit_variance.py`).
   Rather than guess how much a rating edge decides a match, we replay ~50k
   historical internationals to reconstruct each team's Elo before every match,
   then fit the goal model (BASE, GAMMA) and a Dixon-Coles draw correction (RHO)
   by maximum likelihood — **on recent football only (matches since 2000, ~25k
   games)** so the variance reflects the modern game. Change `MIN_YEAR` in
   `fit_variance.py` to widen/narrow the window. The result is well-calibrated:
   predicted W/D/L matches reality in every rating-edge bucket and the draw rate
   is exact. The fitted constants live in `engine.py`.

   What the data says about football's irreducible randomness:
   - an even match is a draw ~32% of the time;
   - a "clear favorite" (70% by rating) still fails to win ~40% of single games.
   This is why a model claiming a single team at 26% to win 7 knockout-ish
   rounds is making a strong low-variance assumption.

Two views of the prediction:
- **Probabilities**: each team's odds of reaching R16/QF/SF/Final/winning.
- **Favorites bracket**: one concrete fillable bracket where the higher-rated
  team wins every game.

## Validation (`backtest.py`)

The per-match model is scored against **552 real World Cup matches (1990-2022)**,
using each team's pre-match reconstructed Elo:

- **log-loss 0.977 vs 1.065** for a base-rate baseline — an **8.2% improvement**
  (large, for high-variance W/D/L football).
- **Favorite picks the winner 73%** of decisive games; calibration tracks
  reality across all confidence buckets.
- **The fitted variance generalizes**: sweeping GAMMA, the World-Cup-optimal
  value (~1.8) is statistically indistinguishable from the ~1.98 fit on the
  recent-internationals sample (the log-loss curve is flat across 1.8-2.0). So the
  randomness isn't overfit — it holds on the matches we actually care about.

What is *not* backtested: the market-ensemble weight, which would need
historical betting odds we don't have. The backtest validates the engine and
variance, not `--market-weight`.

## On accuracy (and why a bank can say "Spain 26%")

Before kickoff there is no ground truth, so "accuracy" can only be judged after
the fact by scoring many predictions (Brier / log-loss). A model that says
Spain 26% vs the market's ~15% isn't necessarily "more accurate" — the gap is
mostly **assumed match variance** (less randomness => the favorite's edge
compounds into a bigger title number) plus a model being free to disagree with
the market. Markets are historically the best-calibrated *single* forecast.

What genuinely improves accuracy here, in priority order:

1. **Ensemble** (built in): blend the market with an independent model so errors
   partly cancel. Beats any single source. Tune with `--market-weight`.
2. **Favorite-longshot correction** (built in): markets underprice favorites;
   `--fav-bias` nudges them up. ~1.05-1.10 is typical.
3. **Backtest the variance** (planned): the right `engine.py` variance and blend
   weight should be chosen by scoring on past World Cups, not by eye.
4. **Better inputs**: multiple rating systems (Elo + SPI + Opta power), squad
   value, xG-based attack/defense, manual injury adjustments near kickoff.

To go beyond that you need proprietary data — diminishing returns from here.

## Tuning

- `data.py` — Elo ratings (all 48 from eloratings.net, no estimates), the 12
  groups, host list.
- `engine.py` — the model's knobs: `HOME_ADVANTAGE_ELO` plus the data-fitted
  `BASE_GOALS`, `ELO_SUPREMACY`, `DRAW_RHO` (re-fit via `fit_variance.py`).

## Known simplifications

- Group tiebreakers use points → goal difference → goals scored → Elo. FIFA's
  actual rules add head-to-head and fair-play steps; this rarely changes who
  advances.
- The 8 third-placed teams are slotted by a deterministic bipartite matching
  that honors FIFA's eligibility lists, rather than reproducing the full 495-row
  combination table. Valid brackets, negligible effect on aggregate odds.
- Ratings are static (no in-tournament form/injury updates).

## Data sources (June 2026)

- Groups: 2026 World Cup final draw (5 Dec 2025).
- Elo: eloratings.net (all 48 teams, the canonical World Football Elo).
- Markets (`market.py`): two selectable sources — **ESPN** sportsbook futures
  (~18% vig) and the **Kalshi** exchange (KXMENWORLDCUP-26, ~5% vig, via its API).
  The report has a selector to calibrate to either; both are de-vigged the same way.
- Bracket structure: official FIFA match schedule.
