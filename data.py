"""
Input data for the 2026 FIFA World Cup model.

Everything the model needs that comes from the real world lives here, so you can
tweak a rating or fix a group without touching the simulation code.

Sources (fetched June 2026):
  - Groups: 2026 World Cup final draw (5 Dec 2025).
  - Elo: worldfootballrankings.com / eloratings.net public top-50.

Teams outside the public top-50 are marked ESTIMATE below. They are lower-ranked
qualifiers whose exact Elo wasn't in the scraped table; the values are reasonable
placeholders. Edit them if you have better numbers — the model is fully
data-driven, so nothing else needs to change.
"""

# Host nations get a home-advantage bump (see engine.HOME_ADVANTAGE_ELO).
HOSTS = {"Mexico", "Canada", "United States"}

# World Football Elo ratings, ~June 2026.
ELO = {
    # --- from public top-50 table ---
    "Argentina": 1876.12,
    "Spain": 1873.01,
    "France": 1869.43,
    "England": 1827.05,
    "Portugal": 1766.18,
    "Brazil": 1765.86,
    "Morocco": 1757.29,
    "Netherlands": 1751.10,
    "Belgium": 1742.24,
    "Germany": 1735.77,
    "Croatia": 1712.24,
    "Colombia": 1695.99,
    "Mexico": 1687.48,
    "Senegal": 1686.41,
    "Uruguay": 1673.07,
    "United States": 1671.23,
    "Japan": 1661.58,
    "Switzerland": 1650.06,
    "Iran": 1619.58,
    "Turkiye": 1605.73,
    "Austria": 1597.40,
    "Ecuador": 1596.48,
    "South Korea": 1591.63,
    "Australia": 1579.34,
    "Algeria": 1571.03,
    "Egypt": 1562.37,
    "Canada": 1559.48,
    "Norway": 1555.60,
    "Ivory Coast": 1540.87,
    "Panama": 1539.16,
    "Paraguay": 1505.35,
    "Czechia": 1505.74,
    "Scotland": 1503.34,
    "DR Congo": 1479.68,
    "Tunisia": 1476.41,
    "Uzbekistan": 1461.21,
    "Sweden": 1509.79,
    # --- ESTIMATEs (qualifiers outside the scraped top-50) ---
    "Ghana": 1455.0,          # ESTIMATE
    "Bosnia and Herzegovina": 1450.0,  # ESTIMATE
    "Qatar": 1448.0,          # ESTIMATE
    "South Africa": 1445.0,   # ESTIMATE
    "Saudi Arabia": 1440.0,   # ESTIMATE
    "Iraq": 1420.0,           # ESTIMATE
    "Jordan": 1405.0,         # ESTIMATE
    "New Zealand": 1400.0,    # ESTIMATE
    "Cape Verde": 1398.0,     # ESTIMATE
    "Curacao": 1335.0,        # ESTIMATE
    "Haiti": 1330.0,          # ESTIMATE
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
    print("data OK: 48 teams, all rated.")
