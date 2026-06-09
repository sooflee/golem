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

# Kalshi prediction-market prices (yes bid/ask midpoint), event KXMENWORLDCUP-26,
# pulled from the Kalshi API, June 2026. These are already ~probabilities (a
# peer-to-peer exchange, so the "vig" is just the small bid/ask overround ~5%,
# vs a sportsbook's ~18%). Stored raw; de-vigged the same way (normalize to 1).
KALSHI_PROBS = {
    "Spain": 0.1645, "France": 0.1625, "England": 0.1015, "Portugal": 0.1005,
    "Argentina": 0.0885, "Brazil": 0.0785, "Germany": 0.0595, "Netherlands": 0.0475,
    "Norway": 0.0235, "Belgium": 0.0225, "Colombia": 0.0185, "Japan": 0.0165,
    "United States": 0.016, "Mexico": 0.0155, "Morocco": 0.0155, "Turkiye": 0.0115,
    "Switzerland": 0.0105, "Uruguay": 0.0095, "Croatia": 0.0095, "Ecuador": 0.0085,
    "Senegal": 0.0075, "Iraq": 0.005, "DR Congo": 0.005, "Bosnia and Herzegovina": 0.005,
    "Czechia": 0.005, "Sweden": 0.0045, "Austria": 0.0045, "Ivory Coast": 0.0035,
    "Canada": 0.0035, "Egypt": 0.0025, "Scotland": 0.0025, "Paraguay": 0.0025,
    "South Korea": 0.0025, "Haiti": 0.0015, "Algeria": 0.0015, "Iran": 0.0015,
    "Ghana": 0.0015, "Australia": 0.0015, "Panama": 0.0005, "Curacao": 0.0005,
    "Qatar": 0.0005, "South Africa": 0.0005, "Cape Verde": 0.0005, "Jordan": 0.0005,
    "Uzbekistan": 0.0005, "New Zealand": 0.0005, "Tunisia": 0.0005, "Saudi Arabia": 0.0005,
}

# Human labels for the report's market selector.
SOURCES = {"kalshi": "Kalshi (exchange)", "espn": "ESPN (sportsbooks)"}


def _raw(source):
    if source == "kalshi":
        return dict(KALSHI_PROBS)
    return {t: 1.0 / d for t, d in MARKET_ODDS_DECIMAL.items()}


def implied_probabilities(bias_k=1.0, source="espn"):
    """De-vigged market title probabilities (sum to 1.0) for the chosen source.

    bias_k applies a favorite-longshot correction: markets underprice strong
    favorites and overprice longshots. Raising de-vigged probs to a power k>1 and
    renormalizing nudges favorites up / longshots down. k=1.0 is the raw de-vigged
    market; ~1.05-1.10 is a typical correction."""
    raw = _raw(source)
    total = sum(raw.values())
    probs = {t: p / total for t, p in raw.items()}
    if bias_k != 1.0:
        adj = {t: p ** bias_k for t, p in probs.items()}
        s = sum(adj.values())
        probs = {t: p / s for t, p in adj.items()}
    return probs


def overround(source="espn"):
    """The market margin: how far raw implied probs sum above 1.0."""
    return sum(_raw(source).values())


if __name__ == "__main__":
    for src, label in SOURCES.items():
        probs = implied_probabilities(source=src)
        print(f"\n{label}  (vig {100 * (overround(src) - 1):.1f}%)")
        for t, p in sorted(probs.items(), key=lambda x: -x[1])[:8]:
            print(f"  {t:<24}{100 * p:5.1f}%")
