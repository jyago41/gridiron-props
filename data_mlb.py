"""MLB player-game box scores from the official MLB Stats API (statsapi.mlb.com, free, no key).

Each season is ~2,500 box scores. They're fetched in parallel and cached to disk; for the current
season only new games are downloaded on later runs.
Batters get player_id "B<id>", pitchers "P<id>" (two-way players like Ohtani get one of each).
"""
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import pandas as pd
import requests

from config import current_mlb_season

BASE = "https://statsapi.mlb.com/api/v1"
GAME_TYPES = "R,F,D,L,W"  # regular season + all playoff rounds
POSTSEASON = {"F", "D", "L", "W"}   # wild card, division series, LCS, World Series
CACHE_DIR = "cache"
BOX_FIELDS = ("teams,away,home,team,id,name,players,person,fullName,position,abbreviation,battingOrder,"
              "stats,batting,pitching,plateAppearances,atBats,hits,doubles,triples,homeRuns,runs,rbi,"
              "baseOnBalls,strikeOuts,stolenBases,totalBases,outs,inningsPitched,earnedRuns,battersFaced,"
              "numberOfPitches,pitchesThrown,batters,pitchers,officials,official,officialType")
_session = requests.Session()


def _get(path, **params):
    r = _session.get(f"{BASE}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def schedule(start: str, end: str, probables: bool = False, game_pks=None) -> pd.DataFrame:
    params = {"sportId": 1, "gameType": GAME_TYPES}
    if game_pks:
        params["gamePks"] = ",".join(str(g) for g in game_pks)
    else:
        params.update(startDate=start, endDate=end)
    if probables:
        params["hydrate"] = "probablePitcher,seriesStatus,officials"
    rows = []
    for d in _get("/schedule", **params).get("dates", []):
        for g in d.get("games", []):
            h, a = g["teams"]["home"], g["teams"]["away"]
            rows.append({
                "game_pk": g["gamePk"], "game_date": d["date"], "start": g.get("gameDate"),
                "game_number": g.get("gameNumber", 1), "state": g["status"]["abstractGameState"],
                "game_type": g.get("gameType", "R"),
                "home_team": h["team"]["name"], "away_team": a["team"]["name"],
                "home_pp": h.get("probablePitcher", {}).get("id"), "away_pp": a.get("probablePitcher", {}).get("id"),
                "home_pp_name": h.get("probablePitcher", {}).get("fullName"),
                "away_pp_name": a.get("probablePitcher", {}).get("fullName"),
                "series": _series_label(g), "hp_ump": _hp_ump(g.get("officials")),
            })
    return pd.DataFrame(rows)


def _hp_ump(officials):
    for o in officials or []:
        if o.get("officialType") == "Home Plate":
            return o.get("official", {}).get("id")
    return None


def handedness(ids, cache_path=os.path.join(CACHE_DIR, "mlb_hands.pkl")) -> dict:
    """{player id: (bats, throws)}, e.g. (L, R). Fetched 150 players per call and cached."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    known = pd.read_pickle(cache_path) if os.path.exists(cache_path) else {}
    todo = [i for i in {int(x) for x in ids} if i not in known]
    for s in range(0, len(todo), 150):
        try:
            data = _get("/people", personIds=",".join(map(str, todo[s:s + 150])),
                        fields="people,id,batSide,pitchHand,code")
            for p in data.get("people", []):
                known[p["id"]] = (p.get("batSide", {}).get("code"), p.get("pitchHand", {}).get("code"))
        except Exception:
            pass
    pd.to_pickle(known, cache_path)
    return known


def _series_label(g) -> str:
    """e.g. 'Division Series, Game 3 · LAD leads 2-0' (blank in the regular season)."""
    if g.get("gameType", "R") not in POSTSEASON:
        return ""
    parts = [g.get("seriesDescription", "")]
    if g.get("seriesGameNumber"):
        parts[0] += f", Game {g['seriesGameNumber']}"
    status = g.get("seriesStatus", {}) or {}
    if status.get("result"):
        parts.append(status["result"])
    return " · ".join(p for p in parts if p)


def lineups(game_pk) -> dict:
    """Posted batting orders: {'home': [player ids], 'away': [...]}; empty lists until posted."""
    try:
        box = fetch_box(game_pk)
        return {s: [int(x) for x in box["teams"][s].get("battingOrder", [])] for s in ("home", "away")}
    except Exception:
        return {"home": [], "away": []}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _ip_to_outs(ip):
    try:
        whole, _, frac = str(ip).partition(".")
        return int(whole) * 3 + (int(frac) if frac else 0)
    except ValueError:
        return float("nan")


def fetch_box(game_pk):
    box = _get(f"/game/{game_pk}/boxscore", fields=BOX_FIELDS)
    if not box.get("teams", {}).get("home", {}).get("players"):   # field filter failed: get it all
        box = _get(f"/game/{game_pk}/boxscore")
    return box


def parse_box(box: dict, game_pk, game_date: str, game_number: int = 1, game_type: str = "R") -> list[dict]:
    rows = []
    season = int(game_date[:4])
    order = int(game_date.replace("-", "")) * 10 + int(game_number or 1)
    teams = box["teams"]
    for side, other in (("home", "away"), ("away", "home")):
        t = teams[side]
        team, opp = t["team"]["name"], teams[other]["team"]["name"]
        base = {"league": "mlb", "team": team, "opponent": opp, "season": season, "week": 0,
                "game_id": str(game_pk), "game_order": order, "game_date": game_date,
                "is_home": int(side == "home"), "postseason": int(game_type in POSTSEASON),
                "hp_ump": _hp_ump(box.get("officials"))}
        players = t.get("players", {})
        for pid in t.get("batters", []):
            p = players.get(f"ID{pid}", {})
            b = p.get("stats", {}).get("batting", {})
            bo = str(p.get("battingOrder", ""))
            if not b or not bo.isdigit():
                continue
            hits, d2, d3, hr = _f(b.get("hits")), _f(b.get("doubles")), _f(b.get("triples")), _f(b.get("homeRuns"))
            tb = _f(b.get("totalBases"))
            if pd.isna(tb):
                tb = hits + d2 + 2 * d3 + 3 * hr
            runs, rbi = _f(b.get("runs")), _f(b.get("rbi"))
            rows.append({**base, "player_id": f"B{pid}", "player_name": p.get("person", {}).get("fullName"),
                         "position": p.get("position", {}).get("abbreviation"), "pos_group": "B",
                         "lineup_spot": int(bo[0]), "started": int(bo.endswith("00")),
                         "pa": _f(b.get("plateAppearances")), "hits": hits, "total_bases": tb, "hr": hr,
                         "runs": runs, "rbi": rbi, "hrr": hits + runs + rbi, "bb": _f(b.get("baseOnBalls")),
                         "so": _f(b.get("strikeOuts")), "sb": _f(b.get("stolenBases"))})
        for i, pid in enumerate(t.get("pitchers", [])):
            p = players.get(f"ID{pid}", {})
            s = p.get("stats", {}).get("pitching", {})
            if not s:
                continue
            outs = _f(s.get("outs"))
            if pd.isna(outs):
                outs = _ip_to_outs(s.get("inningsPitched"))
            rows.append({**base, "player_id": f"P{pid}", "player_name": p.get("person", {}).get("fullName"),
                         "position": "P", "pos_group": "SP" if i == 0 else "RP",
                         "p_outs": outs, "p_strikeouts": _f(s.get("strikeOuts")), "p_hits": _f(s.get("hits")),
                         "p_bb": _f(s.get("baseOnBalls")), "p_er": _f(s.get("earnedRuns")),
                         "p_bf": _f(s.get("battersFaced")),
                         "p_pitches": _f(s.get("numberOfPitches", s.get("pitchesThrown")))})
    return rows


def load_mlb_season(season: int, workers: int = 16, log=print) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"mlb_v3_{season}.pkl")
    cached = pd.read_pickle(path) if os.path.exists(path) else pd.DataFrame()
    if not cached.empty and season < current_mlb_season():
        return cached
    sched = schedule(f"{season}-02-15", f"{season}-11-30")
    if sched.empty:
        return cached
    done = set(cached["game_id"]) if not cached.empty else set()
    todo = sched[(sched.state == "Final") & ~sched.game_pk.astype(str).isin(done)].drop_duplicates("game_pk")
    if len(todo):
        log(f"MLB {season}: downloading {len(todo)} box scores…")

        def work(g):
            try:
                return parse_box(fetch_box(g.game_pk), g.game_pk, g.game_date, g.game_number, g.game_type)
            except Exception:
                return []
        with ThreadPoolExecutor(workers) as ex:
            rows = [r for rs in ex.map(work, todo.itertuples()) for r in rs]
        new = pd.DataFrame(rows)
        cached = pd.concat([cached, new], ignore_index=True) if not cached.empty else new
        cached.to_pickle(path)
    return cached


def add_handedness(df: pd.DataFrame) -> pd.DataFrame:
    hands = handedness(df["player_id"].str[1:].astype(int).unique())
    pid = df["player_id"].str[1:].astype(int)
    df["bat_side"] = pid.map(lambda i: hands.get(i, (None, None))[0])
    df["pitch_hand"] = pid.map(lambda i: hands.get(i, (None, None))[1])
    return df


def load_mlb(seasons: list[int], log=print) -> pd.DataFrame:
    frames = [load_mlb_season(s, log=log) for s in seasons]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    df = add_handedness(df)
    return df.sort_values(["game_order", "game_id"]).reset_index(drop=True)
