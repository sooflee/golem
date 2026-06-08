"""
Input data for the 2026 FIFA World Cup model.

Everything the model needs that comes from the real world lives here, so you can
tweak a rating or fix a group without touching the simulation code.

Sources (fetched June 2026):
  - Groups: 2026 World Cup final draw (5 Dec 2025).
  - Elo: eloratings.net (World Football Elo Ratings) — the canonical source, a
    genuine /400 Elo so its rating gaps map correctly through engine.py's
    win-expectancy. All 48 teams are real values (no estimates).

Note on scale: eloratings.net runs higher than some other Elo sites (top team
~2150 vs ~1870 elsewhere). That's fine — the engine is scale-free (it only uses
rating gaps), and market calibration anchors the headline odds regardless.
"""

# Host nations get a home-advantage bump (see engine.HOME_ADVANTAGE_ELO).
HOSTS = {"Mexico", "Canada", "United States"}

# World Football Elo ratings from eloratings.net, ~June 2026.
ELO = {
    "Spain": 2155,
    "Argentina": 2114,
    "France": 2062,
    "England": 2021,
    "Brazil": 1991,
    "Portugal": 1986,
    "Colombia": 1982,
    "Netherlands": 1944,
    "Ecuador": 1938,
    "Germany": 1932,
    "Norway": 1914,
    "Turkiye": 1911,
    "Croatia": 1911,
    "Japan": 1906,
    "Belgium": 1893,
    "Uruguay": 1892,
    "Switzerland": 1891,
    "Mexico": 1875,
    "Senegal": 1867,
    "Paraguay": 1833,
    "Austria": 1830,
    "Morocco": 1827,
    "Canada": 1788,
    "Scotland": 1782,
    "Australia": 1777,
    "Iran": 1772,
    "Algeria": 1760,
    "South Korea": 1758,
    "Czechia": 1740,
    "Panama": 1730,
    "United States": 1726,
    "Uzbekistan": 1718,
    "Sweden": 1712,
    "Egypt": 1696,
    "Ivory Coast": 1695,
    "Jordan": 1680,
    "DR Congo": 1661,
    "Tunisia": 1628,
    "Iraq": 1618,
    "Bosnia and Herzegovina": 1595,
    "Cape Verde": 1578,
    "Saudi Arabia": 1569,
    "New Zealand": 1562,
    "Haiti": 1548,
    "South Africa": 1518,
    "Ghana": 1510,
    "Curacao": 1434,
    "Qatar": 1421,
}

# The 12 groups from the final draw. 4 teams each, 48 total.
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkiye"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


def validate():
    """Sanity-check the data: 48 teams, every team has an Elo."""
    teams = [t for g in GROUPS.values() for t in g]
    assert len(teams) == 48, f"expected 48 teams, got {len(teams)}"
    assert len(set(teams)) == 48, "duplicate team across groups"
    missing = [t for t in teams if t not in ELO]
    assert not missing, f"missing Elo for: {missing}"
    return True


if __name__ == "__main__":
    validate()
    print("data OK: 48 teams, all rated (eloratings.net).")
