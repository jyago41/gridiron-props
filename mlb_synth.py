"""Synthetic MLB seasons in the same schema as data_mlb, used only for offline testing."""
import numpy as np
import pandas as pd

NAMES = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India"]
TEAMS = ["New York Yankees", "Boston Red Sox", "Los Angeles Dodgers", "Houston Astros", "Atlanta Braves", "Chicago Cubs"]


def synth(seasons=(2025, 2026), games_per_team=60, seed=1):
    rng = np.random.default_rng(seed)
    rows, pid = [], 1000
    hitters = {t: [(pid := pid + 1, f"{t.split()[-1]} Hitter {NAMES[i]}", rng.uniform(0.2, 0.32)) for i in range(9)] for t in TEAMS}
    sps = {t: [(pid := pid + 1, f"{t.split()[-1]} Ace {NAMES[i]}", rng.uniform(0.18, 0.32)) for i in range(5)] for t in TEAMS}
    hand = {p[0]: rng.choice(["L", "R", "S"], p=[.35, .55, .10]) for t in TEAMS for p in hitters[t]}
    throw = {p[0]: rng.choice(["L", "R"], p=[.3, .7]) for t in TEAMS for p in sps[t]}
    umps = {u: rng.normal(1.0, 0.08) for u in range(40)}
    gpk = 1
    for season in seasons:
        for day in range(games_per_team):
            order = rng.permutation(TEAMS)
            for h, a in zip(order[::2], order[1::2]):
                gpk += 1
                date = pd.Timestamp(f"{season}-04-01") + pd.Timedelta(days=day)
                go = int(date.strftime("%Y%m%d")) * 10 + 1
                post = int(day >= games_per_team - 12)
                ump = int(rng.integers(0, 40))
                for team, opp, home in ((h, a, 1), (a, h, 0)):
                    opp_sp = sps[opp][day % 5]
                    for spot, (bid, name, skill) in enumerate(hitters[team], 1):
                        pa = int(rng.choice([4, 4, 5, 5, 3])) if spot < 7 else int(rng.choice([3, 4, 4]))
                        same = hand[bid] != "S" and hand[bid] == throw[opp_sp[0]]
                        p_hit = np.clip(skill - (opp_sp[2] - 0.25) * 0.5 - (0.03 if same else 0), 0.1, 0.4)
                        hits = rng.binomial(pa, p_hit)
                        hr = rng.binomial(hits, 0.15)
                        runs, rbi = rng.binomial(hits + 1, 0.35), rng.binomial(hits + 1, 0.35)
                        rows.append(dict(league="mlb", team=team, opponent=opp, season=season, week=0, game_id=str(gpk),
                                         game_order=go, game_date=str(date.date()), is_home=home, postseason=post, hp_ump=ump, bat_side=hand[bid], player_id=f"B{bid}",
                                         player_name=name, position="OF", pos_group="B", lineup_spot=spot, started=1,
                                         pa=pa, hits=hits, total_bases=hits + hr * 3, hr=hr, runs=runs, rbi=rbi,
                                         hrr=hits + runs + rbi, bb=rng.binomial(1, .08), so=rng.binomial(pa, .22), sb=0))
                    sid, sname, krate = sps[team][day % 5]
                    bf = int(rng.integers(14, 21)) if post else int(rng.integers(18, 28))
                    rows.append(dict(league="mlb", team=team, opponent=opp, season=season, week=0, game_id=str(gpk),
                                     game_order=go, game_date=str(date.date()), is_home=home, postseason=post, hp_ump=ump, pitch_hand=throw[sid], player_id=f"P{sid}",
                                     player_name=sname, position="P", pos_group="SP", p_bf=bf,
                                     p_outs=int(bf * 0.7), p_strikeouts=rng.binomial(bf, min(krate * umps[ump], .6)), p_hits=rng.binomial(bf, .22),
                                     p_bb=rng.binomial(bf, .08), p_er=rng.binomial(4, .4), p_pitches=bf * 4))
    return pd.DataFrame(rows), hitters, sps
