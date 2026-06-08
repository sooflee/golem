"""
Betting-market data: the strongest publicly available forecast.

Outright "to win the World Cup" odds for all 48 teams, ~early June 2026
(ESPN / major sportsbooks futures board). Stored as decimal odds.

We convert these to implied probabilities and strip the bookmaker margin
("vig") by normalizing so they sum to 1. The result is the market's actual
forecast of each team's title chance — the target the model is calibrated to.
"""

# Decimal odds (stake returned per 1 unit). American +450 -> 5.5; "14-1" -> 15.
MARKET_ODDS_DECIMAL = {
    "Spain": 5.5,
    "France": 5.75,
    "England": 8.0,
    "Portugal": 9.5,
    "Argentina": 10.0,
    "Brazil": 10.5,
    "Germany": 15.0,
    "Netherlands": 21.0,
    "Norway": 36.0,
    "Belgium": 41.0,
    "Colombia": 41.0,
    "Morocco": 51.0,
    "United States": 61.0,
    "Switzerland": 66.0,
    "Uruguay": 66.0,
    "Japan": 66.0,
    "Mexico": 81.0,
    "Ecuador": 81.0,
    "Turkiye": 91.0,
    "Croatia": 91.0,
    "Senegal": 91.0,
    "Sweden": 121.0,
    "Austria": 151.0,
    "Canada": 201.0,
    "Scotland": 201.0,
    "Ivory Coast": 251.0,
    "Czechia": 251.0,
    "Paraguay": 301.0,
    "Egypt": 301.0,
    "Ghana": 301.0,
    "Algeria": 351.0,
    "South Korea": 401.0,
    "Bosnia and Herzegovina": 501.0,
    "Tunisia": 501.0,
    "Australia": 601.0,
    "Iran": 701.0,
    "DR Congo": 1001.0,
    "Saudi Arabia": 1001.0,
    "South Africa": 1001.0,
    "Panama": 1001.0,
    "Cape Verde": 1001.0,
    "Qatar": 1501.0,
    "Uzbekistan": 1501.0,
    "New Zealand": 1501.0,
    "Iraq": 1501.0,
    "Jordan": 2501.0,
    "Curacao": 2501.0,
    "Haiti": 2501.0,
}


def implied_probabilities(bias_k=1.0):
    """De-vigged market title probabilities (sum to 1.0).

    bias_k applies a favorite-longshot correction: betting markets underprice
    strong favorites and overprice longshots. Raising de-vigged probs to a power
    k>1 and renormalizing nudges favorites up / longshots down. k=1.0 is the raw
    de-vigged market; ~1.05-1.10 is a typical correction."""
    raw = {t: 1.0 / d for t, d in MARKET_ODDS_DECIMAL.items()}
    total = sum(raw.values())
    probs = {t: p / total for t, p in raw.items()}
    if bias_k != 1.0:
        adj = {t: p ** bias_k for t, p in probs.items()}
        s = sum(adj.values())
        probs = {t: p / s for t, p in adj.items()}
    return probs


def overround():
    """The bookmaker margin: how far raw implied probs sum above 1.0."""
    return sum(1.0 / d for d in MARKET_ODDS_DECIMAL.values())


if __name__ == "__main__":
    probs = implied_probabilities()
    print(f"Overround (vig): {100 * (overround() - 1):.1f}%\n")
    print("De-vigged market title odds:")
    for t, p in sorted(probs.items(), key=lambda x: -x[1])[:15]:
        print(f"  {t:<24}{100 * p:5.1f}%")
