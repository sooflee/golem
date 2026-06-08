"""
Simulate one full 2026 World Cup: group stage -> Round of 32 -> ... -> Final.

The 48-team format:
  - 12 groups of 4, round robin (3 games each).
  - Top 2 of every group advance (24 teams) + the 8 best 3rd-placed teams = 32.
  - The 8 third-placed qualifiers are slotted into specific Round-of-32 matches
    via FIFA's combination table; we reproduce its eligibility constraints with a
    deterministic bipartite matching (see THIRD_PLACE_SLOTS).
  - Single elimination from the Round of 32 to the Final.

Bracket structure (match numbers) is the official one from the FIFA schedule.
"""

from itertools import combinations

import data
import engine

# --- Round of 32 ---------------------------------------------------------
# Eight matches pit a group winner against one of the 8 best third-placed teams.
#   match_no: (winner_group, [groups its 3rd-place opponent may come from])
THIRD_PLACE_SLOTS = {
    74: ("E", ["A", "B", "C", "D", "F"]),
    77: ("I", ["C", "D", "F", "G", "H"]),
    79: ("A", ["C", "E", "F", "H", "I"]),
    80: ("L", ["E", "H", "I", "J", "K"]),
    81: ("D", ["B", "E", "F", "I", "J"]),
    82: ("G", ["A", "E", "H", "I", "J"]),
    85: ("B", ["E", "F", "G", "I", "J"]),
    87: ("K", ["D", "E", "I", "J", "L"]),
}

# The other eight R32 matches have fully determined slots.
# Each slot is (position, group) with position "W" (winner) or "R" (runner-up).
R32_FIXED = {
    73: (("R", "A"), ("R", "B")),
    75: (("W", "F"), ("R", "C")),
    76: (("W", "C"), ("R", "F")),
    78: (("R", "E"), ("R", "I")),
    83: (("R", "K"), ("R", "L")),
    84: (("W", "H"), ("R", "J")),
    86: (("W", "J"), ("R", "H")),
    88: (("R", "D"), ("R", "G")),
}

# Bracket tree from the Round of 16 onward: match_no -> (feeder, feeder).
R16 = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
       93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
QF = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SF = {101: (97, 98), 102: (99, 100)}
FINAL_MATCH = 104  # Winner 101 vs Winner 102

ROUND_NAMES = ["Group", "R32", "R16", "QF", "SF", "Final", "Champion"]


class TeamRecord:
    """A team's group-stage tally."""
    __slots__ = ("name", "pts", "gd", "gf")

    def __init__(self, name):
        self.name = name
        self.pts = 0
        self.gd = 0
        self.gf = 0

    def add(self, scored, conceded):
        self.gf += scored
        self.gd += scored - conceded
        if scored > conceded:
            self.pts += 3
        elif scored == conceded:
            self.pts += 1


def _sort_key(rec, elo, hosts):
    # FIFA tiebreakers (head-to-head, etc.) are more elaborate; points -> goal
    # difference -> goals for -> Elo is the standard modeling approximation.
    return (rec.pts, rec.gd, rec.gf, engine.adjusted_elo(rec.name, elo, hosts))


def _play_group(teams, elo, hosts, rng, deterministic):
    """Round-robin a group; return standings (best first)."""
    recs = {t: TeamRecord(t) for t in teams}
    if not deterministic:
        for a, b in combinations(teams, 2):
            ea = engine.adjusted_elo(a, elo, hosts)
            eb = engine.adjusted_elo(b, elo, hosts)
            ga, gb = engine.match_goals(ea, eb, rng)
            recs[a].add(ga, gb)
            recs[b].add(gb, ga)
    return sorted(recs.values(),
                  key=lambda r: _sort_key(r, elo, hosts),
                  reverse=True)


def _assign_thirds(third_groups, rng, deterministic):
    """Match the 8 qualifying third-place groups to the 8 R32 third-place slots,
    honoring each slot's eligibility list. Returns {match_no: group}.

    This is a perfect bipartite matching found by backtracking. FIFA's official
    table is a fixed lookup; any valid matching preserves the same-group
    avoidance the table encodes, with negligible effect on aggregate odds."""
    slots = list(THIRD_PLACE_SLOTS.items())  # [(match_no, (wgrp, eligible)), ...]
    # Order slots by how constrained they are (fewest eligible groups present
    # first) to make backtracking fast and deterministic.
    present = set(third_groups)
    slots.sort(key=lambda s: sum(1 for g in s[1][1] if g in present))

    assignment = {}
    used = set()

    def backtrack(i):
        if i == len(slots):
            return True
        match_no, (_, eligible) = slots[i]
        options = [g for g in eligible if g in present and g not in used]
        if not deterministic:
            rng.shuffle(options)
        for g in options:
            used.add(g)
            assignment[match_no] = g
            if backtrack(i + 1):
                return True
            used.remove(g)
            del assignment[match_no]
        return False

    if not backtrack(0):
        # No valid assignment for this particular set of 8 groups (rare). Fall
        # back to a simple in-order fill so the sim never crashes.
        leftover = list(present)
        for match_no, _ in slots:
            assignment[match_no] = leftover.pop()
    return assignment


def simulate(elo=None, hosts=None, rng=None, deterministic=False):
    """Run one tournament. Returns a dict:
        champion, runner_up, semifinalists, and
        reached: {team: index into ROUND_NAMES of furthest round reached}.

    deterministic=True ignores rng and lets the higher adjusted-Elo team always
    win (groups ranked purely by Elo) — i.e. the chalk / favorites bracket.
    """
    elo = elo or data.ELO
    hosts = hosts or data.HOSTS

    # --- Group stage ---
    winners, runners, thirds = {}, {}, {}
    third_records = []
    for g, teams in data.GROUPS.items():
        standings = _play_group(teams, elo, hosts, rng, deterministic)
        winners[g] = standings[0].name
        runners[g] = standings[1].name
        thirds[g] = standings[2].name
        third_records.append((g, standings[2]))

    reached = {t: 0 for grp in data.GROUPS.values() for t in grp}

    # --- Best 8 third-placed teams ---
    third_records.sort(key=lambda gr: _sort_key(gr[1], elo, hosts), reverse=True)
    qualifying_thirds = third_records[:8]
    third_groups = [g for g, _ in qualifying_thirds]
    third_slot = _assign_thirds(third_groups, rng, deterministic)

    # Everyone who reached the knockouts is at least R32.
    for g in winners:
        reached[winners[g]] = 1
        reached[runners[g]] = 1
    for g in third_groups:
        reached[thirds[g]] = 1

    # --- Resolve Round-of-32 fixtures into concrete teams ---
    def slot_team(pos, grp):
        return winners[grp] if pos == "W" else runners[grp]

    r32 = {}
    for m, (sa, sb) in R32_FIXED.items():
        r32[m] = (slot_team(*sa), slot_team(*sb))
    for m, (wgrp, _) in THIRD_PLACE_SLOTS.items():
        r32[m] = (winners[wgrp], thirds[third_slot[m]])

    # --- Play a knockout match between two named teams ---
    def play(a, b):
        ea = engine.adjusted_elo(a, elo, hosts)
        eb = engine.adjusted_elo(b, elo, hosts)
        if deterministic:
            return a if ea >= eb else b
        return engine.knockout_winner(a, ea, b, eb, rng)

    # Round of 32 -> winners keyed by match number, advance to round index 2.
    won = {}
    for m, (a, b) in r32.items():
        w = play(a, b)
        won[m] = w
        reached[w] = 2

    def play_round(bracket, round_idx):
        for m, (f1, f2) in bracket.items():
            w = play(won[f1], won[f2])
            won[m] = w
            reached[w] = round_idx

    play_round(R16, 3)
    play_round(QF, 4)
    play_round(SF, 5)

    fa, fb = won[101], won[102]
    champion = play(fa, fb)
    runner_up = fb if champion == fa else fa
    reached[champion] = 6
    reached[runner_up] = 5  # finalist; champion gets 6

    return {
        "champion": champion,
        "runner_up": runner_up,
        "semifinalists": [won[97], won[98], won[99], won[100]],
        "reached": reached,
        # Concrete bracket detail, useful for the deterministic chalk bracket.
        "r32": r32,
        "won": won,
        "winners": winners,
        "runners": runners,
        "thirds": {g: thirds[g] for g in third_groups},
    }
