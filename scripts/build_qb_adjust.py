"""Builds qb_adjust.csv: starting-QB point adjustments for every game of the current NFL season.

Method (backtested Nov 2026, fit on 2000-2015, validated on 2016-2025):
- QB quality entering a game = career (passing + rushing EPA) / plays, using only games
  played before that week, shrunk toward replacement level (-0.15 EPA/play) by 150 plays.
- Rated QB level = average quality of the team's starters in its earlier games this season,
  plus last season's starters weighted as 12 games (matches the rating engine's shrink K=12).
- delta = starter quality - rated QB level.
- Spread adjustment (home margin) = 15.38 x (delta_home - delta_away)
- Total adjustment               =  6.81 x (delta_home + delta_away)
Upcoming games with no listed starter assume the team's most recent starter.
Data: nflverse (player stats, games). Run: python scripts/build_qb_adjust.py [out.csv]
"""
import bisect, io, sys, urllib.request
import numpy as np, pandas as pd

PRIOR, REPL, K = 150.0, -0.15, 12
C_SPREAD, C_TOTAL = 15.38, 6.81
REL = "https://github.com/nflverse/nflverse-data/releases/download"
GAMES = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
FR = {"STL": "LA", "SD": "LAC", "OAK": "LV", "LAR": "LA"}
COLS = ["player_id", "team", "season", "week", "attempts", "sk", "passing_epa", "carries", "rushing_epa"]

def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()

def main(out="qb_adjust.csv"):
    g = pd.read_csv(io.BytesIO(get(GAMES)))
    season = int(g[g.home_score.notna()].season.max())
    legacy = pd.read_parquet(io.BytesIO(get(f"{REL}/player_stats/player_stats.parquet")))
    legacy = legacy.rename(columns={"recent_team": "team", "sacks": "sk"})
    frames = [legacy[COLS]]
    for y in range(int(legacy.season.max()) + 1, season + 1):
        w = pd.read_parquet(io.BytesIO(get(f"{REL}/stats_player/stats_player_week_{y}.parquet")))
        frames.append(w.rename(columns={"sacks_suffered": "sk"})[COLS])
    q = pd.concat(frames)
    for c in ["attempts", "sk", "passing_epa", "carries", "rushing_epa"]:
        q[c] = q[c].fillna(0)
    q = q[(q.attempts + q.sk) > 0].copy()
    q["plays"] = q.attempts + q.sk + q.carries
    q["epa"] = q.passing_epa + q.rushing_epa
    q["ord"] = (q.season * 100 + q.week).astype(int)
    q = q.groupby(["player_id", "ord"], as_index=False)[["plays", "epa"]].sum().sort_values(["player_id", "ord"])
    q["pl"] = q.groupby("player_id").plays.cumsum()
    q["ep"] = q.groupby("player_id").epa.cumsum()
    H = {p: (d.ord.tolist(), d.pl.tolist(), d.ep.tolist()) for p, d in q.groupby("player_id")}

    def qual(pid, o):
        h = H.get(pid)
        if h is None:
            return REPL
        j = bisect.bisect_left(h[0], o) - 1
        return REPL if j < 0 else (h[2][j] + PRIOR * REPL) / (h[1][j] + PRIOR)

    for c in ["home_team", "away_team"]:
        g[c] = g[c].replace(FR)
    g["ord"] = g.season * 100 + g.week
    g = g[g.season.isin([season - 1, season])].sort_values(["ord", "gameday", "gametime"])
    names = {}
    for s in ["home", "away"]:
        names.update(dict(zip(g[f"{s}_qb_id"], g[f"{s}_qb_name"])))
    starters, last = {}, {}
    rows = []
    for r in g.itertuples():
        rec = {"game_id": r.game_id, "week": r.week}
        for s in ["home", "away"]:
            team, qb = getattr(r, f"{s}_team"), getattr(r, f"{s}_qb_id")
            assumed = not isinstance(qb, str)
            if assumed:
                qb = last.get(team)
            cur = starters.get((team, r.season), [])
            prev = starters.get((team, r.season - 1), [])
            if r.season == season:
                a = sum(qual(x, r.ord) for x in cur)
                b = np.mean([qual(x, r.ord) for x in prev]) if prev else REPL
                rated = (a + K * b) / (len(cur) + K)
                qs = qual(qb, r.ord) if qb else rated
                rec[f"{s}_qb"] = names.get(qb, "") if qb else ""
                rec[f"{s}_assumed"] = int(assumed)
                rec[f"{s}_delta"] = round(qs - rated, 4)
            if not assumed:
                starters.setdefault((team, r.season), []).append(qb)
                last[team] = qb
        if r.season == season:
            rows.append(rec)
    o = pd.DataFrame(rows)
    o["adj_spread"] = (C_SPREAD * (o.home_delta - o.away_delta)).round(2)
    o["adj_total"] = (C_TOTAL * (o.home_delta + o.away_delta)).round(2)
    o.to_csv(out, index=False)
    print(f"{season}: wrote {len(o)} games to {out}")

if __name__ == "__main__":
    main(*sys.argv[1:])
