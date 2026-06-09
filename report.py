#!/usr/bin/env python3
"""
Generate an interactive single-page report (index.html) for GitHub Pages.

The forecast is precomputed at several market/model blend weights in Python and
embedded as JSON; a slider in the browser switches between them (each stop is a
real model run, not interpolated). Below the forecast is a "How it works"
methodology section with a worked example.

    python3 report.py                 # index.html, 60k sims per weight
    python3 report.py --sims 100000
    python3 report.py --outdir docs

First run is slow (it calibrates each weight; results are cached per weight).
"""

import argparse
import datetime
import html
import json
import math
import os

import backtest
import calibrate
import data
import engine
import fit_variance as fv
import histelo
import market
import montecarlo
import tournament

# Slider stops, ordered left->right = pure model -> pure market (matches the
# slider's end labels). Value is the weight on the market.
WEIGHTS = [0.0, 0.25, 0.50, 0.70, 0.80, 0.85, 0.90, 1.0]
REC_IDX = 5  # 0.85 (85% market / 15% model) is the recommended blend
DEFAULT_SOURCE = "kalshi"  # market shown on first load

# Host-advantage states the report can toggle between (Elo bump for hosts).
# Ratings are calibrated at the default (60); the others re-simulate those same
# ratings with the bump changed, so "none" / "full" are counterfactuals.
HOST_STATES = [("none", 0.0), ("wc", 60.0), ("full", 96.0)]
HOST_LABELS = {"none": "None", "wc": "WC (60)", "full": "Full (96)"}
DEFAULT_HOST = "wc"
ROUND_KEYS = ("R16", "QF", "SF", "Final", "Champion")

# Hand-drawn-style red ellipse drawn around the favorite's championship odds.
CIRCLE_SVG = ('<svg class=circ viewBox="0 0 120 56" preserveAspectRatio=none>'
              '<path d="M98,9 C71,1 31,4 16,17 C4,28 8,46 41,52 C77,58 114,46 110,25 '
              'C107,11 86,6 67,8" fill=none stroke="#e23b3b" stroke-width=2.6 '
              'stroke-linecap=round/></svg>')


# ---------------------------------------------------------------- compute ----

def _snap(elo, sims):
    counts, n = montecarlo.run(sims, seed=0, elo=elo)
    rows = montecarlo.probabilities(counts, n)
    by_team = {r["team"]: r for r in rows}
    chalk = tournament.simulate(elo=elo, deterministic=True)
    snap = {"champ": chalk["champion"],
            "won": {str(m): chalk["won"][m] for m in chalk["won"]},
            "teams": {t: [round(by_team[t][k], 4) for k in ROUND_KEYS]
                      for t in by_team}}
    return snap, rows, by_team, chalk, n


def gather(sims):
    # snaps[host_state][source] = [per-weight snapshot]. Ratings are calibrated
    # once (at the default host bump); each host state re-simulates them with the
    # bump set to none/60/96, so none & full are counterfactuals around the
    # market-calibrated default.
    cal_ha = engine.HOME_ADVANTAGE_ELO  # calibrate at the default (60)
    snaps = {k: {} for k, _ in HOST_STATES}
    mkt = {}
    rec = None
    for src in market.SOURCES:
        mkt[src] = market.implied_probabilities(source=src)
        for k, _ in HOST_STATES:
            snaps[k][src] = []
        for i, w in enumerate(WEIGHTS):
            print(f"  {src} {int(w*100)}% ...", flush=True)
            engine.HOME_ADVANTAGE_ELO = cal_ha
            elo = calibrate.load_or_calibrate(market_weight=w, bias_k=1.05,
                                              source=src, verbose=False)
            for k, hv in HOST_STATES:
                engine.HOME_ADVANTAGE_ELO = hv
                snap, rows, by_team, chalk, n = _snap(elo, sims)
                snaps[k][src].append(snap)
                if (k == DEFAULT_HOST and src == DEFAULT_SOURCE
                        and i == REC_IDX):
                    rec = {"rows": rows, "by_team": by_team, "chalk": chalk,
                           "elo": elo, "n": n}
    engine.HOME_ADVANTAGE_ELO = cal_ha  # restore
    return {"snaps": snaps, "rec": rec, "mkt": mkt, "sims": rec["n"]}


# ----------------------------------------------------------------- helpers ---

def esc(s):
    return html.escape(str(s))


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def bar(x):
    w = max(0.5, 100 * x)
    return (f'<div class="bar"><span style="width:{w:.1f}%;'
            f'background:var(--ac)"></span></div>')


def odds_table(rows, mkt, top):
    head = ("<tr><th>#</th><th class=l>Team</th><th>Champion</th>"
            "<th>Final</th><th>Semi</th><th>Quarter</th><th>R16</th>"
            "<th>Market</th></tr>")
    body = []
    for i, r in enumerate(rows[:top], 1):
        m = mkt.get(r["team"])
        mtxt = pct(m) if m is not None else "&ndash;"
        if i == 1:
            ch = (f"<td class='prob circled'>{bar(r['Champion'])}"
                  f"<span class=cw>{CIRCLE_SVG}{pct(r['Champion'])}</span></td>")
        else:
            ch = (f"<td class=prob>{bar(r['Champion'])}"
                  f"<span>{pct(r['Champion'])}</span></td>")
        body.append(
            f"<tr><td class=rank>{i}</td><td class=l><b>{esc(r['team'])}</b></td>"
            f"{ch}<td>{pct(r['Final'])}</td><td>{pct(r['SF'])}</td>"
            f"<td>{pct(r['QF'])}</td><td>{pct(r['R16'])}</td>"
            f"<td class=mkt>{mtxt}</td></tr>")
    return f"<table class=odds>{head}{''.join(body)}</table>"


def groups_grid(by_team):
    cards = []
    for g, teams in data.GROUPS.items():
        ranked = sorted(teams, key=lambda t: by_team[t]["R16"], reverse=True)
        rowsh = []
        for j, t in enumerate(ranked):
            cls = "adv" if j < 2 else "out"
            rowsh.append(f"<div class='grow {cls}'><span>{esc(t)}</span>"
                         f"<em>{pct(by_team[t]['R16'],0)}</em></div>")
        cards.append(f"<div class=gcard><h4>Group {g}</h4>{''.join(rowsh)}</div>")
    return f"<div class=groups>{''.join(cards)}</div>"


def bracket(chalk, by_team):
    won = chalk["won"]

    def line(t, w, key):
        cls = "win" if t == w else ""
        return (f"<div class='m {cls}'>{esc(t)}"
                f"<em>{pct(by_team[t][key])}</em></div>")

    def match(a, b, w, key):
        return f"<div class=tie>{line(a, w, key)}{line(b, w, key)}</div>"

    def col(title, mnos, feeders, key):
        ties = "".join(match(won[f1], won[f2], won[m], key)
                       for m in mnos for f1, f2 in [feeders[m]])
        return f"<div class=bcol><h4>{title}</h4>{ties}</div>"

    cols = [col("Round of 16", (89, 90, 91, 92, 93, 94, 95, 96), tournament.R16, "R16"),
            col("Quarter-finals", (97, 98, 99, 100), tournament.QF, "QF"),
            col("Semi-finals", (101, 102), tournament.SF, "SF")]
    fa, fb, champ = won[101], won[102], chalk["champion"]
    cols.append(f"<div class=bcol><h4>Final</h4>{match(fa, fb, champ, 'Final')}"
                f"<div class=trophy>&#127942; {esc(champ)} "
                f"{pct(by_team[champ]['Champion'])}</div></div>")
    return f"<div class=bracket>{''.join(cols)}</div>"


def randomness_table():
    g, b, rho = engine.ELO_SUPREMACY, engine.BASE_GOALS, engine.DRAW_RHO
    rows = []
    for label, we in [("Even (50/50)", .50), ("Slight edge (60%)", .60),
                      ("Clear favorite (70%)", .70), ("Big favorite (80%)", .80),
                      ("Huge favorite (90%)", .90)]:
        wm = we - .5
        w, d, l = fv._wdl(b * math.exp(g * wm), b * math.exp(-g * wm), rho)
        rows.append(f"<tr><td class=l>{label}</td><td>{pct(w,0)}</td>"
                    f"<td>{pct(d,0)}</td><td>{pct(l,0)}</td></tr>")
    return ("<table class=rand><tr><th class=l>Matchup (by rating)</th>"
            "<th>Win</th><th>Draw</th><th>Lose</th></tr>"
            f"{''.join(rows)}</table>")


def ticks_html():
    """Tick marks under the slider, positioned at each stop's actual market %
    (so spacing reflects the real distance between stops, not even steps)."""
    out = []
    for i, w in enumerate(WEIGHTS):
        # Match the slider thumb's travel: its center moves from 11px (half the
        # 22px thumb) to width-11px, not 0%..100%. calc() mirrors that exactly.
        left = f"calc(11px + {w:.4f} * (100% - 22px))"
        cls = "tk rec" if i == REC_IDX else "tk"
        out.append(f'<span class="{cls}" data-i="{i}" '
                   f'style="left:{left}">{int(round(w * 100))}</span>')
    return "".join(out)


def worked_example(elo, a="Spain", b="Croatia"):
    ea, eb = elo[a], elo[b]
    gap = ea - eb
    we = engine.win_expectancy(ea, eb)
    wm = we - 0.5
    g, base, rho = engine.ELO_SUPREMACY, engine.BASE_GOALS, engine.DRAW_RHO
    la, lb = base * math.exp(g * wm), base * math.exp(-g * wm)
    w, d, l = fv._wdl(la, lb, rho)
    ph, pa = fv._poisson_pmf(la), fv._poisson_pmf(lb)
    best, bestp = (0, 0), 0.0
    for x in range(8):
        for y in range(8):
            p = ph[x] * pa[y] * fv._tau(x, y, la, lb, rho)
            if p > bestp:
                best, bestp = (x, y), p
    return f"""
<p><b>{esc(a)}</b> (rating {ea:.0f}) vs <b>{esc(b)}</b> (rating {eb:.0f}),
neutral venue. Exactly what the engine does:</p>
<p class=step>Step A &mdash; rating gap</p>
<div class=formula>{ea:.0f} &minus; {eb:.0f} = {gap:.0f} points in {esc(a)}'s favor</div>
<p class=step>Step B &mdash; turn the gap into a win expectancy</p>
<div class=formula>win_exp = 1 / (1 + 10^(&minus;{gap:.0f}/400)) = {we:.3f}
&rarr; on strength alone, {esc(a)} is about a {100*we:.0f}% favorite</div>
<p class=step>Step C &mdash; turn that into expected goals</p>
<div class=formula>&lambda;({esc(a)})  = {base} &times; e^( {g}&times;({we:.3f}&minus;0.5)) = {la:.2f} goals
&lambda;({esc(b)}) = {base} &times; e^(&minus;{g}&times;({we:.3f}&minus;0.5)) = {lb:.2f} goals</div>
<p class=step>Step D &mdash; roll the dice (Poisson + draw correction)</p>
<div class=stats>
<div class=stat><b>{pct(w,0)}</b><small>{esc(a)} win</small></div>
<div class=stat><b>{pct(d,0)}</b><small>draw</small></div>
<div class=stat><b>{pct(l,0)}</b><small>{esc(b)} win</small></div>
</div>
<p class=mut>Most likely exact score: {esc(a)} {best[0]}&ndash;{best[1]} {esc(b)}.
{esc(b)} still avoids defeat {pct(d+l,0)} of the time &mdash; that's the
randomness from Step&nbsp;4. A tournament is just this, 103 times, repeated many
times over.</p>
"""


# ------------------------------------------------------------------ style ----

STYLE = """
:root{--bg:#fbfaf7;--ink:#1c1b19;--mut:#6f6a62;--ac:#1f6f54;--acbg:#eaf2ed;
--line:#e6e1d8;--tile:#f5f2ec;
--serif:'Iowan Old Style','Palatino Linotype',Palatino,Georgia,'Times New Roman',serif;
--sans:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,system-ui,sans-serif}
*{box-sizing:border-box}
html{scrollbar-gutter:stable}
body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.7 var(--serif);
-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
.wrap{max-width:900px;margin:0 auto;padding:20px 22px 64px}
p{margin:0 0 1em}em{font-style:italic}a{color:var(--ac)}
ul{padding-left:1.1em}li{margin:.35em 0}
.mut{font-family:var(--sans);color:var(--mut);font-size:13.5px;line-height:1.55}
code{background:#f0ede5;padding:2px 6px;border-radius:4px;font-size:14px;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
/* masthead */
.kick{font-family:var(--sans);color:var(--ac);letter-spacing:.14em;text-transform:uppercase;
font-size:12px;font-weight:700}
header{padding:34px 0 22px;margin-bottom:8px;border-bottom:2px solid var(--ink)}
header h1{font-size:clamp(26px,3.6vw,38px);line-height:1.12;letter-spacing:-.015em;
margin:.2em 0 .3em;font-weight:700}
.dek{font-size:18px;line-height:1.5;color:#3a3833;margin:0 0 18px}
.dek b{color:var(--ac)}
.byline{font-family:var(--sans);font-size:13.5px;color:var(--mut)}
.byline a{color:var(--ac)}
/* article flow */
section{margin:36px 0;padding:0}
h2{font-size:23px;line-height:1.25;font-weight:700;letter-spacing:-.01em;margin:0 0 .5em}
h2 .n{font-family:var(--sans);font-size:14px;font-weight:700;color:#fff;background:var(--ac);
border-radius:50%;width:28px;height:28px;display:inline-flex;align-items:center;justify-content:center;
vertical-align:middle;margin-right:10px}
/* collapsible sections */
details>summary{list-style:none;cursor:pointer;display:flex;align-items:center;gap:11px;user-select:none;
background:var(--tile);border:1px solid var(--line);border-radius:9px;padding:12px 16px;transition:border-color .15s}
details>summary:hover{border-color:var(--ac)}
details>summary::-webkit-details-marker{display:none}
details>summary h2{margin:0;font-size:20px}
details>summary::before{content:"\\25B8";color:var(--ac);font-size:14px;transition:transform .15s}
details[open]>summary::before{transform:rotate(90deg)}
details>summary::after{content:"Show more \\25BE";margin-left:auto;font-family:var(--sans);font-size:12px;
font-weight:700;color:#fff;background:var(--ac);border-radius:20px;padding:4px 12px;white-space:nowrap}
details[open]>summary::after{content:"Hide \\25B4";color:var(--ac);background:transparent;border:1px solid var(--ac)}
details[open]>summary{margin-bottom:.7em;border-radius:9px 9px 9px 9px}
details:not([open]){margin-bottom:6px}
/* section break */
.divider{text-align:center;margin:62px 0 12px}
.divider span{font-family:var(--sans);color:var(--ac);letter-spacing:.16em;text-transform:uppercase;
font-size:12px;font-weight:700}
.divider h2{font-size:27px;margin:.12em 0}
.divider:after{content:"";display:block;width:56px;height:2px;background:var(--ac);margin:14px auto 0}
/* tables as figures */
table{width:100%;border-collapse:collapse;font-family:var(--sans);font-size:14px;
font-variant-numeric:tabular-nums;margin:16px 0}
th,td{padding:9px 8px;text-align:right;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.04em;
border-bottom:1.5px solid var(--ink)}
.l{text-align:left}.rank{color:var(--mut)}.mkt{color:var(--mut)}
/* fixed layout so columns don't re-measure (and shift) when the data changes */
.odds{table-layout:fixed}
.odds th:nth-child(1),.odds td:nth-child(1){width:6%}
.odds th:nth-child(2),.odds td:nth-child(2){width:26%}
.odds td:nth-child(2){white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.odds th:nth-child(3),.odds td:nth-child(3){width:17%}
.odds th:nth-child(8),.odds td:nth-child(8){width:9%}
.odds tr:hover td{background:var(--tile)}
.prob{position:relative}
.prob span:last-child{position:relative}
.bar{position:absolute;inset:0;display:flex;align-items:center;padding:0 8px}
.bar span{height:62%;border-radius:2px;display:block;opacity:.18;transition:width .3s}
.prob.circled{overflow:visible}
.prob .cw{position:relative;display:inline-block}
.circ{position:absolute;left:50%;top:50%;width:158%;height:215%;
transform:translate(-50%,-50%) rotate(-3deg);pointer-events:none;z-index:3;overflow:visible}
/* groups */
.groups{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.gcard{background:var(--tile);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.gcard h4{font-family:var(--sans);margin:0 0 8px;font-size:11px;color:var(--mut);
text-transform:uppercase;letter-spacing:.05em}
.grow{display:flex;justify-content:space-between;font-family:var(--sans);font-size:13px;padding:3px 0}
.grow em{font-style:normal;color:var(--mut)}
.grow.adv{font-weight:600}.grow.adv em{color:var(--ac);font-weight:700}
.grow.out{opacity:.42}
/* bracket */
.bracket{display:flex;gap:12px;overflow-x:auto;padding-bottom:6px;font-family:var(--sans)}
.bcol{flex:1;min-width:150px}
.bcol h4{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.05em;margin:0 0 10px}
.tie{background:var(--tile);border:1px solid var(--line);border-radius:7px;margin-bottom:9px;overflow:hidden}
.m{padding:7px 10px;font-size:13px;border-bottom:1px solid var(--line);
display:flex;justify-content:space-between;gap:8px}
.m:last-child{border-bottom:0}
.m em{font-style:normal;color:var(--mut);font-variant-numeric:tabular-nums}
.m.win{background:var(--acbg);font-weight:700;box-shadow:inset 3px 0 0 var(--ac)}
.m.win em{color:var(--ac)}
.trophy{margin-top:10px;text-align:center;font-weight:700;color:var(--ac);font-family:var(--serif);font-size:17px}
/* stats */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:18px;margin:22px 0}
.stat{text-align:center}
.stat b{display:block;font-size:34px;color:var(--ac);font-weight:700;line-height:1}
.stat small{font-family:var(--sans);color:var(--mut);font-size:12.5px;display:block;margin-top:7px;line-height:1.4}
/* aside / formula */
.note{background:var(--acbg);border-left:3px solid var(--ac);border-radius:0 6px 6px 0;
padding:13px 18px;margin:18px 0;font-size:16px;line-height:1.6}
.note b{color:var(--ac)}
.formula{background:#f3f1ea;border:1px solid var(--line);border-radius:6px;padding:12px 16px;margin:12px 0;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13.5px;white-space:pre-wrap;color:#2a4a40;line-height:1.5}
p.step{font-family:var(--sans);color:var(--ac);font-weight:700;margin:1.4em 0 .3em;
font-size:13px;text-transform:uppercase;letter-spacing:.04em}
/* slider figure */
.tuner{background:var(--tile);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin:18px 0}
.tuner .row{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px}
.tuner .wlab{font-family:var(--sans);font-size:15px}.tuner .wlab b{color:var(--ac)}
.tuner .wtag{font-family:var(--sans);font-size:11px;font-weight:700;text-transform:uppercase;
letter-spacing:.06em;color:#fff;background:var(--ac);padding:3px 10px;border-radius:20px}
.tuner .wtag:empty{display:none}
.mkts{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:14px;
padding-bottom:14px;border-bottom:1px solid var(--line)}
.mlbl{font-family:var(--sans);font-size:12px;font-weight:700;text-transform:uppercase;
letter-spacing:.06em;color:var(--mut)}
.mbtn,.hbtn{font-family:var(--sans);font-size:13px;font-weight:600;cursor:pointer;
background:#fff;color:var(--ink);border:1px solid var(--line);border-radius:20px;padding:5px 14px}
.mbtn:hover,.hbtn:hover{border-color:var(--ac)}
.mbtn.on,.hbtn.on{background:var(--ac);color:#fff;border-color:var(--ac)}
.showctl{display:flex;align-items:center;gap:7px;margin-top:12px;flex-wrap:wrap}
.nbtn{font-family:var(--sans);font-size:13px;font-weight:600;cursor:pointer;background:#fff;
color:var(--ink);border:1px solid var(--line);border-radius:18px;padding:4px 13px}
.nbtn:hover{border-color:var(--ac)}
.nbtn.on{background:var(--ac);color:#fff;border-color:var(--ac)}
input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:5px;border-radius:5px;
background:linear-gradient(90deg,#cdc6b8,var(--ac));margin:16px 0 6px;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:22px;height:22px;border-radius:50%;
background:#fff;border:3px solid var(--ac);cursor:pointer;box-shadow:0 1px 3px rgba(0,0,0,.2)}
input[type=range]::-moz-range-thumb{width:20px;height:20px;border-radius:50%;background:#fff;border:3px solid var(--ac)}
.ends{display:flex;justify-content:space-between;font-family:var(--sans);color:var(--mut);font-size:12px}
.ticks{position:relative;height:24px;margin-top:5px}
.ticks .tk{position:absolute;transform:translateX(-50%);font-family:var(--sans);font-size:11px;
color:var(--mut);cursor:pointer;text-align:center;line-height:1}
.ticks .tk::before{content:"";display:block;width:1px;height:6px;background:#cdc6b8;margin:0 auto 3px}
.ticks .tk:hover{color:var(--ink)}
.ticks .tk.rec{color:var(--ac);font-weight:700}
.ticks .tk.rec::before{background:var(--ac);height:8px}
.ticks .tk.on{color:var(--ink);font-weight:700}
.ticks .tk.on::before{background:var(--ink);height:8px}
footer{margin-top:54px;padding-top:22px;border-top:1px solid var(--line);
font-family:var(--sans);color:var(--mut);font-size:13px;line-height:1.6}
@media(max-width:600px){
body{font-size:16px}
.wrap{padding:16px 14px 50px}
.stats{grid-template-columns:1fr 1fr}
/* odds table: drop the mid round columns, keep Team / Champion / Market */
.odds{font-size:13px}
.odds th,.odds td{padding:8px 5px}
.odds th:nth-child(4),.odds td:nth-child(4),.odds th:nth-child(5),.odds td:nth-child(5),
.odds th:nth-child(6),.odds td:nth-child(6),.odds th:nth-child(7),.odds td:nth-child(7){display:none}
.odds th:nth-child(1),.odds td:nth-child(1){width:9%}
.odds th:nth-child(2),.odds td:nth-child(2){width:40%}
.odds th:nth-child(3),.odds td:nth-child(3){width:28%}
.odds th:nth-child(8),.odds td:nth-child(8){width:23%}
/* slider ticks: keep the marks, label only the two ends to avoid overlap */
.ticks .tk{font-size:0}
.ticks .tk:first-child,.ticks .tk:last-child{font-size:10px}
.bcol{min-width:130px}
}
"""


# ------------------------------------------------------------------- body ----

BODY = """
<header>
<div class=kick>Forecast &middot; World Cup 2026</div>
<h1>Who will win the 2026 World Cup?</h1>
<p class=dek>Nobody really knows &mdash; it only happens once, and luck gets a big
vote. So I had a computer play the whole thing {sims:,} times, leaning on the
betting market and 25 years of results, just to see who keeps walking away with
the trophy. Right now, that's <b>&#127942; <span id=champName>{champ}</span></b>.</p>
<div class=byline>Updated {date} &middot; <a href="#how">how it works &darr;</a></div>
</header>

<div class=tuner>
<div class=mkts><span class=mlbl>Calibrate to market:</span>{mktbtns}</div>
<div class=mkts><span class=mlbl>Host advantage (USA/MEX/CAN):</span>{hostbtns}</div>
<div class=row><div class=wlab id=wlab></div><div class=wtag id=wtag></div></div>
<input type=range id=wslider min=0 max=100 step=1 value="{recpct}">
<div class=ticks>{ticks}</div>
<div class=ends><span>&larr; Our model (Elo)</span><span>Betting market &rarr; (% on market)</span></div>
<p class=mut style="margin:.6em 0 0">Drag to reweight the forecast between our
independent Elo model and the betting market. Every stop is a real model run.
Watch teams the market and the model disagree on (e.g. Argentina) move the most.</p>
<p class=mut style="margin:.7em 0 0">The default is <b>85% market / 15% model</b>.
Why 15%, and not 0 or 50? It's a judgment, not a fitted number: blending an
independent model with the market usually helps a little (so the weight should be
above zero), but the market knows more than our ratings (so it should stay small).
We can't tune it precisely &mdash; that would need historical betting odds, which
don't exist for past tournaments &mdash; so we chose a conservative value and built
this slider, rather than ask you to take one number on faith.</p>
</div>

<section><h2>Title &amp; deep-run odds</h2><div id=oddsWrap>{odds}</div>
<div class=showctl><span class=mlbl>Show top:</span><button class="nbtn on" data-n=8>8</button><button class=nbtn data-n=16>16</button><button class=nbtn data-n=48>48</button></div>
<p class=mut>"Market" is the de-vigged bookmaker title odds. Other columns are the
simulated probability of reaching each round at the current slider setting. The
<span style="color:#e23b3b;font-weight:700">red circle</span> marks the model's
most likely champion. At blends below 100% the model can read above or below the
Market column &mdash; that's its independent (Elo) view disagreeing with the
bookmakers.</p></section>

<section><details><summary><h2>Group stage &mdash; predicted top two</h2></summary>
<div id=groupsWrap>{groups}</div>
<p class=mut>Highlighted = advance. The 8 best third-placed teams also qualify.</p>
</details></section>

<section><details><summary><h2>Knockout bracket (favorites)</h2></summary>
<p class=mut>One concrete bracket where the higher-rated side wins every game. The
% next to each team is its chance of reaching that round (across all simulations).</p>
<div id=bracketWrap>{bracket}</div>
</details></section>

<div class=divider id=how><span>The method</span><h2>How it works</h2></div>

<section><h2><span class=n>0</span>The big idea: play the tournament 100,000 times</h2>
<p>Here's the problem: the World Cup happens exactly once, and on any given day the
better team loses all the time. You can't out-predict that. So I stopped trying to,
and asked a question you actually <em>can</em> answer:
<b>if this exact tournament were replayed thousands of times, how often would each
team win?</b></p>
<p>That's a <b>Monte Carlo simulation</b>. Simulate one match realistically &mdash;
with the right randomness &mdash; and you can play all 103 matches (we skip the
dead-rubber 3rd-place playoff), repeat {sims:,}
times, and just <em>count</em>. Win 16,400 of 100,000 &rarr; a 16.4% title chance.</p>
<div class=note><b>The whole challenge</b> is three things: (1) how good is each
team, (2) how "how good" becomes a result <em>including luck</em>, and (3) the
tournament structure. The rest is counting.</div></section>

<section><h2><span class=n>1</span>How good is each team? (Elo ratings)</h2>
<p>Every team gets an <b>Elo rating</b> &mdash; the system invented for chess. You
gain points for winning (more for beating a strong team), and the <em>gap</em>
between two ratings predicts the result, built so a <b>400-point gap &asymp; a
10-to-1 favorite</b>:</p>
<div class=formula>win_expectancy = 1 / (1 + 10^(&minus;(rating_A &minus; rating_B)/400))</div>
<p>Equal teams &rarr; 0.5; a 200-point edge &rarr; ~76%. Hosts get a small home bump.</p>
<div class=note><b>Why win-expectancy, not raw points?</b> Different sites publish
Elo on different scales. This formula always maps a <em>gap</em> to the same
probability, so the model works with any ratings.</div></section>

<section><h2><span class=n>2</span>Simulating one match</h2>
<p>I simulate the actual <b>scoreline</b>, not just who won &mdash; the group stage
comes down to goal difference, so goals matter. Each side's win-expectancy becomes
an expected number of goals, and the score is a random draw from a <b>Poisson
distribution</b> (the go-to model for counting rare, scattered events &mdash;
goals, raindrops, typos on a page):</p>
<div class=formula>goals_A ~ Poisson( {base} &times; e^( {gamma}&times;(win_exp&minus;0.5)) )
goals_B ~ Poisson( {base} &times; e^(&minus;{gamma}&times;(win_exp&minus;0.5)) )</div>
<p>Two even teams average {base} each; a favorite's average rises. The
<b>Dixon-Coles correction</b> nudges low scores so the draw rate matches reality.</p></section>

<section><h2><span class=n>3</span>A worked example: Spain vs Croatia</h2>{worked}</section>

<section><h2><span class=n>4</span>The key step: measuring how random football is</h2>
<p>The two numbers above ({base} and {gamma}) control <b>how much a rating edge
really decides a match versus how much is luck</b>. Guess wrong and the model gets
over-confident (a bank once put one team at 26% to win it all) or under-confident.
So I didn't eyeball it &mdash; I <b>measured it against {fit_n:,} real
internationals since {fit_year}</b>, letting the data pick the settings that make
what <em>actually happened</em> most likely (<b>maximum likelihood</b>). The verdict on
football's built-in randomness:</p>
{rand}
<div class=note><b>This is why upsets are common.</b> Even a clear favorite loses
or draws ~40% of single games. A high title number is only justified if it comes
from a genuine <em>rating gap</em> &mdash; not from pretending matches are more
predictable than {fit_n:,} games show. (A model that reaches, say, 26% by
under-estimating this randomness is over-confident; here the randomness is fit to
the data, so any concentration comes from the gaps themselves &mdash; which is why
the slider's pure-model end can be high yet still honest.)</div></section>

<section><h2><span class=n>5</span>Trusting the betting market (calibration &amp; the slider)</h2>
<p>The <b>betting market</b> is the best-calibrated forecast there is &mdash;
millions of people with real money, reacting to injuries and news. You can
calibrate to either market (the buttons above the slider) &mdash; a
<b>sportsbook</b> futures board (ESPN) or the <b>Kalshi exchange</b>. Their prices
sum to over 100% &mdash; that overage (the "vig",
~{vig:.0f}% for the sportsbook but only ~{vig_k:.0f}% on the peer-to-peer
exchange) we strip out. Then we
<b>calibrate</b>: nudge each team's rating until <em>our</em> simulated title odds
match that blended target (market + our model, at the slider's mix).</p>
<p>The slider at the top blends the two: <b>0% = our pure Elo model</b> (free to
disagree with the market, Goldman-style), <b>100% = the pure market</b>. The
recommended default is <b>85% market / 15% model</b> &mdash; lean on the market
(historically the best single forecast) while keeping the model as a hedge.</p></section>

<section><h2>What it can't do (being honest)</h2>
<ul>
<li>Ratings are fixed at kickoff &mdash; no reaction to a mid-tournament injury.</li>
<li>Ratings are a single pre-tournament snapshot from one provider
(eloratings.net) — a different rating system would shift the edges somewhat.</li>
<li>Group tiebreakers are simplified versus FIFA's full head-to-head rules.</li>
<li>We validated the engine and randomness, but not the exact blend weight &mdash;
that needs historical betting odds we don't have. (Hence the slider: judge for yourself.)</li>
</ul>
<p class=mut>Reproduce: <code>python3 run.py</code>,
<code>python3 fit_variance.py</code>, <code>python3 backtest.py</code>.</p></section>

<footer>Generated {date}. Sources: bookmaker futures, World Football Elo, FIFA
fixtures, the martj42 international-results dataset.<br>
A model, not a guarantee &mdash; the point is that upsets happen.</footer>
"""


# JS is kept out of .format templates (its braces would clash); injected as a
# substituted value in the final f-string instead.
SCRIPT = """
<script>
let SRC=SRC0, CUR=REC, NSHOW=8, HOST="wc";
function pct(x,d=1){return (100*x).toFixed(d)+"%";}
function bar(x){let w=Math.max(0.5,100*x);return `<div class="bar"><span style="width:${w.toFixed(1)}%;background:var(--ac)"></span></div>`;}
const CIRC='<svg class=circ viewBox="0 0 120 56" preserveAspectRatio=none><path d="M98,9 C71,1 31,4 16,17 C4,28 8,46 41,52 C77,58 114,46 110,25 C107,11 86,6 67,8" fill=none stroke="#e23b3b" stroke-width=2.6 stroke-linecap=round/></svg>';
function renderOdds(s){
  const ts=Object.keys(s.teams).map(t=>({t,v:s.teams[t]}));
  ts.sort((a,b)=>b.v[4]-a.v[4]||b.v[3]-a.v[3]);
  let h="<table class=odds><tr><th>#</th><th class=l>Team</th><th>Champion</th><th>Final</th><th>Semi</th><th>Quarter</th><th>R16</th><th>Market</th></tr>";
  ts.slice(0, NSHOW).forEach((r,i)=>{
    const m=MKT[SRC][r.t]; const mt=(m!=null)?pct(m):"&ndash;";
    const ch=(i===0)?`<td class='prob circled'>${bar(r.v[4])}<span class=cw>${CIRC}${pct(r.v[4])}</span></td>`:`<td class=prob>${bar(r.v[4])}<span>${pct(r.v[4])}</span></td>`;
    h+=`<tr><td class=rank>${i+1}</td><td class=l><b>${r.t}</b></td>${ch}<td>${pct(r.v[3])}</td><td>${pct(r.v[2])}</td><td>${pct(r.v[1])}</td><td>${pct(r.v[0])}</td><td class=mkt>${mt}</td></tr>`;
  });
  document.getElementById("oddsWrap").innerHTML=h+"</table>";
}
function renderGroups(s){
  let h="<div class=groups>";
  for(const g in GROUPS){
    const teams=GROUPS[g].slice().sort((a,b)=>s.teams[b][0]-s.teams[a][0]);
    h+=`<div class=gcard><h4>Group ${g}</h4>`;
    teams.forEach((t,j)=>{h+=`<div class='grow ${j<2?"adv":"out"}'><span>${t}</span><em>${pct(s.teams[t][0],0)}</em></div>`;});
    h+="</div>";
  }
  document.getElementById("groupsWrap").innerHTML=h+"</div>";
}
function renderBracket(s){
  const won=s.won, T=s.teams;
  const line=(t,w,r)=>`<div class='m ${t===w?"win":""}'>${t}<em>${pct(T[t][r])}</em></div>`;
  const tie=(a,b,w,r)=>`<div class=tie>${line(a,w,r)}${line(b,w,r)}</div>`;
  const col=(title,mnos,feed,r)=>{let t="";mnos.forEach(m=>{const f=feed[m];t+=tie(won[f[0]],won[f[1]],won[m],r);});return `<div class=bcol><h4>${title}</h4>${t}</div>`;};
  let h=col("Round of 16",[89,90,91,92,93,94,95,96],R16,0);
  h+=col("Quarter-finals",[97,98,99,100],QF,1);
  h+=col("Semi-finals",[101,102],SF,2);
  h+=`<div class=bcol><h4>Final</h4>${tie(won[101],won[102],s.champ,3)}<div class=trophy>&#127942; ${s.champ} ${pct(T[s.champ][4])}</div></div>`;
  document.getElementById("bracketWrap").innerHTML=`<div class=bracket>${h}</div>`;
}
function update(i){
  CUR=i;
  const s=SNAP[HOST][SRC][i], w=WEIGHTS[i];
  const lead=Object.keys(s.teams).reduce((a,b)=>s.teams[b][4]>s.teams[a][4]?b:a);
  document.getElementById("champName").textContent=lead;
  renderOdds(s);renderGroups(s);renderBracket(s);
  document.getElementById("wlab").innerHTML=`Market <b>${w}%</b> &nbsp;/&nbsp; Our model <b>${100-w}%</b>`;
  const tag=(w===100)?"Pure market":(w===0)?"Pure model":(i===REC)?"Recommended":"";
  document.getElementById("wtag").textContent=tag;
  TKS.forEach((t,j)=>t.classList.toggle("on",j===i));
  SL.value=w;  // snap the thumb to this stop's market %
}
function snapIdx(v){let b=0,bd=1e9;WEIGHTS.forEach((w,j)=>{const d=Math.abs(w-v);if(d<bd){bd=d;b=j;}});return b;}
const TKS=[].slice.call(document.querySelectorAll(".ticks .tk"));
const SL=document.getElementById("wslider");
TKS.forEach(t=>t.addEventListener("click",()=>update(+t.getAttribute("data-i"))));
SL.addEventListener("input",()=>update(snapIdx(+SL.value)));
const MB=[].slice.call(document.querySelectorAll(".mbtn"));
MB.forEach(b=>b.addEventListener("click",function(){
  SRC=b.getAttribute("data-src");
  MB.forEach(x=>x.classList.toggle("on",x===b));
  update(CUR);}));
const NB=[].slice.call(document.querySelectorAll(".nbtn"));
NB.forEach(b=>b.addEventListener("click",function(){
  NSHOW=+b.getAttribute("data-n");
  NB.forEach(x=>x.classList.toggle("on",x===b));
  renderOdds(SNAP[HOST][SRC][CUR]);}));
const HB=[].slice.call(document.querySelectorAll(".hbtn"));
HB.forEach(b=>b.addEventListener("click",function(){
  HOST=b.getAttribute("data-h");
  HB.forEach(x=>x.classList.toggle("on",x===b));
  update(CUR);}));
update(REC);
</script>
"""


def build_page(d):
    rec = d["rec"]
    fit_n = sum(1 for _ in histelo.reconstruct(min_year=fv.MIN_YEAR))
    worked = worked_example(rec["elo"]).format(sims=f"{d['sims']:,}")
    mktbtns = "".join(
        f'<button class="mbtn{" on" if s == DEFAULT_SOURCE else ""}" '
        f'data-src="{s}">{esc(lbl)}</button>'
        for s, lbl in market.SOURCES.items())
    hostbtns = "".join(
        f'<button class="hbtn{" on" if k == DEFAULT_HOST else ""}" '
        f'data-h="{k}">{HOST_LABELS[k]}</button>'
        for k, _ in HOST_STATES)
    body = BODY.format(
        sims=d["sims"], champ=esc(rec["rows"][0]["team"]),
        recpct=int(round(WEIGHTS[REC_IDX] * 100)), ticks=ticks_html(),
        mktbtns=mktbtns, hostbtns=hostbtns, fit_n=fit_n, fit_year=fv.MIN_YEAR,
        odds=odds_table(rec["rows"], d["mkt"][DEFAULT_SOURCE], 8),
        groups=groups_grid(rec["by_team"]),
        bracket=bracket(rec["chalk"], rec["by_team"]),
        worked=worked, rand=randomness_table(),
        base=engine.BASE_GOALS, gamma=engine.ELO_SUPREMACY,
        vig=100 * (market.overround("espn") - 1),
        vig_k=100 * (market.overround("kalshi") - 1),
        date=datetime.date.today().strftime("%B %-d, %Y"))

    datajs = ("<script>"
              f"const SNAP={json.dumps(d['snaps'])};"
              f"const WEIGHTS={json.dumps([int(w*100) for w in WEIGHTS])};"
              f"const REC={REC_IDX};const SRC0={json.dumps(DEFAULT_SOURCE)};"
              f"const MKT={json.dumps(d['mkt'])};"
              f"const GROUPS={json.dumps(data.GROUPS)};"
              f"const R16={json.dumps(tournament.R16)};"
              f"const QF={json.dumps(tournament.QF)};"
              f"const SF={json.dumps(tournament.SF)};"
              "</script>")

    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content=\"width=device-width,initial-scale=1\">"
            f"<title>2026 World Cup &mdash; Model Forecast</title>"
            f"<style>{STYLE}</style></head><body><div class=wrap>{body}"
            f"</div>{datajs}{SCRIPT}</body></html>")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sims", type=int, default=100000)
    p.add_argument("--outdir", default=".")
    args = p.parse_args()
    data.validate()
    print(f"Running {args.sims:,} sims at each of {len(WEIGHTS)} blend weights...")
    d = gather(args.sims)
    path = os.path.join(args.outdir, "index.html")
    with open(path, "w") as f:
        f.write(build_page(d))
    print(f"Wrote {path}  (recommended-setting champion: "
          f"{d['rec']['chalk']['champion']})")


if __name__ == "__main__":
    main()
