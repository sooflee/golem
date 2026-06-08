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

# Slider stops: % weight on the market (rest on our independent Elo model).
WEIGHTS = [1.0, 0.85, 0.70, 0.50, 0.25, 0.0]
REC_IDX = 1  # 0.85 is the recommended blend
ROUND_KEYS = ("R16", "QF", "SF", "Final", "Champion")


# ---------------------------------------------------------------- compute ----

def gather(sims):
    mkt = market.implied_probabilities()
    snaps = []
    rec = None
    for i, w in enumerate(WEIGHTS):
        print(f"  weight {int(w*100)}% market ...", flush=True)
        elo = calibrate.load_or_calibrate(market_weight=w, bias_k=1.05,
                                          verbose=False)
        counts, n = montecarlo.run(sims, seed=0, elo=elo)
        rows = montecarlo.probabilities(counts, n)
        by_team = {r["team"]: r for r in rows}
        chalk = tournament.simulate(elo=elo, deterministic=True)
        snaps.append({
            "champ": chalk["champion"],
            "won": {str(m): chalk["won"][m] for m in chalk["won"]},
            "teams": {t: [round(by_team[t][k], 4) for k in ROUND_KEYS]
                      for t in by_team},
        })
        if i == REC_IDX:
            rec = {"rows": rows, "by_team": by_team, "chalk": chalk,
                   "elo": elo, "n": n}

    bt_matches = backtest.wc_matches()
    g, b, rho = engine.ELO_SUPREMACY, engine.BASE_GOALS, engine.DRAW_RHO
    ll, brier, acc = backtest.score(bt_matches, g, b, rho)
    base_ll = backtest.baseline_logloss(bt_matches)
    bt_years = [int(m["date"][:4]) for m in bt_matches]

    return {
        "snaps": snaps, "rec": rec, "mkt": mkt, "sims": rec["n"],
        "bt": {"n": len(bt_matches), "ll": ll, "acc": acc, "base_ll": base_ll,
               "impr": 100 * (base_ll - ll) / base_ll,
               "lo": min(bt_years), "hi": max(bt_years)},
    }


# ----------------------------------------------------------------- helpers ---

def esc(s):
    return html.escape(str(s))


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def bar(x):
    w = max(0.5, 100 * x)
    return (f'<div class="bar"><span style="width:{w:.1f}%;'
            f'background:#2dd4bf"></span></div>')


def odds_table(rows, mkt, top):
    head = ("<tr><th>#</th><th class=l>Team</th><th>Champion</th>"
            "<th>Final</th><th>Semi</th><th>Quarter</th><th>R16</th>"
            "<th>Market</th></tr>")
    body = []
    for i, r in enumerate(rows[:top], 1):
        m = mkt.get(r["team"])
        mtxt = pct(m) if m is not None else "&ndash;"
        body.append(
            f"<tr><td class=rank>{i}</td><td class=l><b>{esc(r['team'])}</b></td>"
            f"<td class=prob>{bar(r['Champion'])}<span>{pct(r['Champion'])}</span></td>"
            f"<td>{pct(r['Final'])}</td><td>{pct(r['SF'])}</td>"
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


def bracket(chalk):
    won = chalk["won"]

    def match(a, b, w):
        def line(t):
            return f"<div class='m {'win' if t == w else ''}'>{esc(t)}</div>"
        return f"<div class=tie>{line(a)}{line(b)}</div>"

    def col(title, mnos, feeders):
        ties = "".join(match(won[f1], won[f2], won[m])
                       for m in mnos for f1, f2 in [feeders[m]])
        return f"<div class=bcol><h4>{title}</h4>{ties}</div>"

    cols = [col("Round of 16", (89, 90, 91, 92, 93, 94, 95, 96), tournament.R16),
            col("Quarter-finals", (97, 98, 99, 100), tournament.QF),
            col("Semi-finals", (101, 102), tournament.SF)]
    fa, fb, champ = won[101], won[102], chalk["champion"]
    cols.append(f"<div class=bcol><h4>Final</h4>{match(fa, fb, champ)}"
                f"<div class=trophy>&#127942; {esc(champ)}</div></div>")
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
:root{--bg:#0b1120;--card:#131c30;--ink:#e6edf6;--mut:#8aa0bd;--ac:#2dd4bf;--line:#22304d}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:24px}
header{text-align:center;padding:40px 16px 18px}
header .kick{color:var(--ac);letter-spacing:.18em;text-transform:uppercase;font-size:13px;font-weight:700}
header h1{font-size:clamp(28px,5vw,46px);margin:.2em 0}
header p{color:var(--mut);max-width:680px;margin:0 auto}
.pick{margin:22px auto;display:inline-block;background:var(--card);border:1px solid var(--line);
border-radius:14px;padding:14px 26px}
.pick b{font-size:26px}.pick span{color:var(--ac)}
.tuner{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px 24px;margin:20px 0}
.tuner .row{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px}
.tuner .wlab{font-size:16px}.tuner .wlab b{color:var(--ac)}
.tuner .wtag{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;
color:#06231f;background:var(--ac);padding:3px 10px;border-radius:20px}
.tuner .wtag:empty{display:none}
input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:6px;border-radius:5px;
background:linear-gradient(90deg,#3b82f6,#2dd4bf);margin:16px 0 6px;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:22px;height:22px;border-radius:50%;
background:#fff;border:3px solid var(--ac);cursor:pointer}
input[type=range]::-moz-range-thumb{width:20px;height:20px;border-radius:50%;background:#fff;border:3px solid var(--ac)}
.ends{display:flex;justify-content:space-between;color:var(--mut);font-size:12px}
.divider{text-align:center;margin:46px 0 6px}
.divider span{color:var(--ac);letter-spacing:.16em;text-transform:uppercase;font-size:13px;font-weight:700}
.divider h2{font-size:30px;border:0;margin:.2em 0}
section{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px 24px;margin:20px 0}
h2{font-size:22px;margin:.1em 0 .6em;border-bottom:1px solid var(--line);padding-bottom:.4em}
h2 .n{color:var(--ac);margin-right:8px}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{padding:7px 8px;text-align:right;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.l{text-align:left}.rank{color:var(--mut)}.mkt{color:var(--mut)}
.prob{position:relative;min-width:120px}
.prob span:last-child{position:relative;font-variant-numeric:tabular-nums}
.bar{position:absolute;inset:0;display:flex;align-items:center;padding:0 8px;opacity:.22}
.bar span{height:60%;border-radius:3px;display:block;transition:width .3s}
.groups{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px}
.gcard{background:#0e1626;border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.gcard h4{margin:0 0 8px;font-size:13px;color:var(--mut)}
.grow{display:flex;justify-content:space-between;font-size:13px;padding:3px 0}
.grow em{font-style:normal;color:var(--mut)}
.grow.adv span{font-weight:600}.grow.adv em{color:var(--ac)}
.grow.out{opacity:.45}
.bracket{display:flex;gap:14px;overflow-x:auto;padding-bottom:6px}
.bcol{flex:1;min-width:160px}
.bcol h4{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.05em;margin:0 0 10px}
.tie{background:#0e1626;border:1px solid var(--line);border-radius:8px;margin-bottom:10px;overflow:hidden}
.m{padding:7px 10px;font-size:13px;border-bottom:1px solid var(--line)}
.m:last-child{border-bottom:0}
.m.win{background:rgba(45,212,191,.14);font-weight:600}
.trophy{margin-top:10px;text-align:center;font-weight:700;color:var(--ac)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:14px 0}
.stat{background:#0e1626;border:1px solid var(--line);border-radius:10px;padding:14px;text-align:center}
.stat b{display:block;font-size:26px;color:var(--ac)}
.stat small{color:var(--mut)}
p,li{color:#cdd9ea}.mut{color:var(--mut);font-size:13px}
a{color:var(--ac)}
code{background:#0e1626;padding:2px 6px;border-radius:5px;font-size:13px}
.note{background:#0e1626;border-left:3px solid var(--ac);border-radius:0 8px 8px 0;
padding:12px 16px;margin:14px 0;font-size:14.5px}
.note b{color:var(--ac)}
.formula{background:#0a0f1c;border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:10px 0;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13.5px;white-space:pre-wrap;color:#bfe9e2}
p.step{color:var(--ac);font-weight:700;margin:1.3em 0 .3em;font-size:14px}
footer{text-align:center;color:var(--mut);font-size:13px;padding:30px 16px}
@media(max-width:640px){.stats{grid-template-columns:1fr 1fr}}
"""


# ------------------------------------------------------------------- body ----

BODY = """
<header>
<div class=kick>Monte Carlo &bull; Market-calibrated &bull; Backtested</div>
<h1>2026 FIFA World Cup &mdash; Model Forecast</h1>
<p>An Elo-driven Monte Carlo model anchored to the betting market, with per-match
randomness fit to 50 years of results. {sims:,} simulated tournaments per setting.
<a href="#how">Jump to how it works &darr;</a></p>
<div class=pick>Most likely champion<br><b>&#127942; <span id=champName>{champ}</span></b></div>
</header>

<div class=tuner>
<div class=row><div class=wlab id=wlab></div><div class=wtag id=wtag></div></div>
<input type=range id=wslider min=0 max="{wmax}" step=1 value="{rec}">
<div class=ends><span>&larr; Our model (Elo)</span><span>Betting market &rarr;</span></div>
<p class=mut style="margin:.6em 0 0">Drag to reweight the forecast between our
independent Elo model and the betting market. Every stop is a real model run.
Watch teams the market and the model disagree on (e.g. Argentina) move the most.</p>
</div>

<section><h2>Title &amp; deep-run odds</h2><div id=oddsWrap>{odds}</div>
<p class=mut>"Market" is the de-vigged bookmaker title odds. Other columns are the
simulated probability of reaching each round at the current slider setting. At
blends below 100% the model can read above or below the Market column &mdash;
that's its independent (Elo) view disagreeing with the bookmakers.</p></section>

<section><h2>Group stage &mdash; predicted top two</h2><div id=groupsWrap>{groups}</div>
<p class=mut>Highlighted = advance. The 8 best third-placed teams also qualify.</p></section>

<section><h2>Knockout bracket (favorites)</h2>
<p class=mut>One concrete bracket where the higher-rated side wins every game.</p>
<div id=bracketWrap>{bracket}</div></section>

<div class=divider id=how><span>The method</span><h2>How it works</h2></div>

<section><h2><span class=n>0</span>The big idea: play the tournament 100,000 times</h2>
<p>You can't know what <em>will</em> happen at one World Cup &mdash; it's played
once, and luck matters. So we ask an answerable question:
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
<p>We simulate the <b>score</b>, not just win/lose &mdash; the group stage ranks by
goal difference. We turn each side's win-expectancy into expected goals, then draw
a random scoreline from a <b>Poisson distribution</b> (the standard model for
counting rare scattered events &mdash; goals, raindrops, typos per page):</p>
<div class=formula>goals_A ~ Poisson( {base} &times; e^( {gamma}&times;(win_exp&minus;0.5)) )
goals_B ~ Poisson( {base} &times; e^(&minus;{gamma}&times;(win_exp&minus;0.5)) )</div>
<p>Two even teams average {base} each; a favorite's average rises. The
<b>Dixon-Coles correction</b> nudges low scores so the draw rate matches reality.</p></section>

<section><h2><span class=n>3</span>A worked example: Spain vs Croatia</h2>{worked}</section>

<section><h2><span class=n>4</span>The key step: measuring how random football is</h2>
<p>The two numbers above ({base} and {gamma}) control <b>how much a rating edge
really decides a match versus how much is luck</b>. Guess wrong and the model gets
over-confident (a bank once put one team at 26% to win it all) or under-confident.
So we <b>measure them from {fit_n:,} real internationals since {fit_year}</b>
(recent football only), finding the settings that make the results that
<em>actually happened</em> most likely (<b>maximum likelihood</b>). The verdict on
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
millions of people with real money, reacting to injuries and news. Bookmaker odds
sum to over 100%; that overage (~{vig:.0f}% here) is their margin, the "vig," which
we strip out. Then we <b>calibrate</b>: nudge each team's rating until <em>our</em>
simulated title odds match that blended target (market + our model, at the
slider's mix).</p>
<p>The slider at the top blends the two: <b>0% = our pure Elo model</b> (free to
disagree with the market, Goldman-style), <b>100% = the pure market</b>. The
recommended default is <b>85% market / 15% model</b> &mdash; lean on the market
(historically the best single forecast) while keeping the model as a hedge.</p></section>

<section><h2><span class=n>6</span>Does it actually work? (Backtesting)</h2>
<p>A forecast you never check is just an opinion. We tested the engine on
<b>{bt_n:,} real World Cup matches ({bt_lo}&ndash;{bt_hi})</b>, predicting each from the
teams' ratings beforehand:</p>
<div class=stats>
<div class=stat><b>{bt_impr:.1f}%</b><small>better than a naive baseline (log-loss)</small></div>
<div class=stat><b>{bt_acc:.0f}%</b><small>favorite correctly picked (decisive games)</small></div>
<div class=stat><b>{bt_ll:.3f}</b><small>log-loss vs baseline {bt_base:.3f}</small></div>
</div>
<p>We score with <b>log-loss</b>, which rewards being confident <em>and</em> right
and punishes confident-but-wrong, so you can't game it by hedging. The randomness
we measured on ordinary games was also near-optimal for World Cup games &mdash; so
we didn't tune the model to its own data.</p></section>

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
const RKEY=["r16","qf","sf","f","ch"];
const fav=(s)=>s.champ;
function pct(x,d=1){return (100*x).toFixed(d)+"%";}
function bar(x){let w=Math.max(0.5,100*x);return `<div class="bar"><span style="width:${w.toFixed(1)}%;background:#2dd4bf"></span></div>`;}
function renderOdds(s){
  const ts=Object.keys(s.teams).map(t=>({t,v:s.teams[t]}));
  ts.sort((a,b)=>b.v[4]-a.v[4]||b.v[3]-a.v[3]);
  let h="<table class=odds><tr><th>#</th><th class=l>Team</th><th>Champion</th><th>Final</th><th>Semi</th><th>Quarter</th><th>R16</th><th>Market</th></tr>";
  ts.slice(0,16).forEach((r,i)=>{
    const m=MKT[r.t]; const mt=(m!=null)?pct(m):"&ndash;";
    h+=`<tr><td class=rank>${i+1}</td><td class=l><b>${r.t}</b></td><td class=prob>${bar(r.v[4])}<span>${pct(r.v[4])}</span></td><td>${pct(r.v[3])}</td><td>${pct(r.v[2])}</td><td>${pct(r.v[1])}</td><td>${pct(r.v[0])}</td><td class=mkt>${mt}</td></tr>`;
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
  const won=s.won;
  const tie=(a,b,w)=>`<div class=tie><div class='m ${a===w?"win":""}'>${a}</div><div class='m ${b===w?"win":""}'>${b}</div></div>`;
  const col=(title,mnos,feed)=>{let t="";mnos.forEach(m=>{const f=feed[m];t+=tie(won[f[0]],won[f[1]],won[m]);});return `<div class=bcol><h4>${title}</h4>${t}</div>`;};
  let h=col("Round of 16",[89,90,91,92,93,94,95,96],R16);
  h+=col("Quarter-finals",[97,98,99,100],QF);
  h+=col("Semi-finals",[101,102],SF);
  h+=`<div class=bcol><h4>Final</h4>${tie(won[101],won[102],s.champ)}<div class=trophy>&#127942; ${s.champ}</div></div>`;
  document.getElementById("bracketWrap").innerHTML=`<div class=bracket>${h}</div>`;
}
function update(i){
  const s=SNAP[i], w=WEIGHTS[i];
  document.getElementById("champName").textContent=s.champ;
  renderOdds(s);renderGroups(s);renderBracket(s);
  document.getElementById("wlab").innerHTML=`Market <b>${w}%</b> &nbsp;/&nbsp; Our model <b>${100-w}%</b>`;
  const tag=(w===100)?"Pure market":(w===0)?"Pure model":(i===REC)?"Recommended":"";
  document.getElementById("wtag").textContent=tag;
}
document.getElementById("wslider").addEventListener("input",e=>update(+e.target.value));
update(REC);
</script>
"""


def build_page(d):
    rec = d["rec"]
    fit_n = sum(1 for _ in histelo.reconstruct(min_year=fv.MIN_YEAR))
    worked = worked_example(rec["elo"]).format(sims=f"{d['sims']:,}")
    body = BODY.format(
        sims=d["sims"], champ=esc(rec["chalk"]["champion"]),
        wmax=len(WEIGHTS) - 1, rec=REC_IDX,
        fit_n=fit_n, fit_year=fv.MIN_YEAR,
        odds=odds_table(rec["rows"], d["mkt"], 16),
        groups=groups_grid(rec["by_team"]), bracket=bracket(rec["chalk"]),
        worked=worked, rand=randomness_table(),
        base=engine.BASE_GOALS, gamma=engine.ELO_SUPREMACY,
        vig=100 * (market.overround() - 1),
        bt_n=d["bt"]["n"], bt_impr=d["bt"]["impr"], bt_acc=100 * d["bt"]["acc"],
        bt_ll=d["bt"]["ll"], bt_base=d["bt"]["base_ll"],
        bt_lo=d["bt"]["lo"], bt_hi=d["bt"]["hi"],
        date=datetime.date.today().strftime("%B %-d, %Y"))

    datajs = ("<script>"
              f"const SNAP={json.dumps(d['snaps'])};"
              f"const WEIGHTS={json.dumps([int(w*100) for w in WEIGHTS])};"
              f"const REC={REC_IDX};"
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
