"""College football player-game history from the CollegeFootballData API (free key: collegefootballdata.com).

Box scores come from /games/players, positions from /roster, and closing lines from /lines.
Parsed seasons are cached locally so you don't burn API calls re-downloading.
"""
import os
import pandas as pd
import requests

from config import STAT_COLS, current_season

BASE = "https://api.collegefootballdata.com"
POS_GROUP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}
CACHE_DIR = "cache"


def _get(path, params, key):
    r = requests.get(BASE + path, params=params, timeout=60,
                     headers={"Authorization": f"Bearer {key}", "accept": "application/json"})
    r.raise_for_status()
    return r.json()


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _parse_games(games, season, week):
    rows = []
    for g in games:
        teams = g.get("teams", [])
        if len(teams) != 2:
            continue
        for i, t in enumerate(teams):
            opp = teams[1 - i]["team"]
            players = {}
            for cat in t.get("categories", []):
                cname = cat["name"]
                for typ in cat.get("types", []):
                    tname = typ["name"]
                    for a in typ.get("athletes", []):
                        if str(a.get("name", "")).strip().lower() in ("team", ""):
                            continue
                        p = players.setdefault(str(a["id"]), {"player_name": a["name"]})
                        stat = a.get("stat")
                        if cname == "passing" and tname == "C/ATT" and stat and "/" in str(stat):
                            c, att = str(stat).split("/")[:2]
                            p["completions"], p["attempts"] = _num(c), _num(att)
                        elif cname == "passing" and tname == "YDS":
                            p["passing_yards"] = _num(stat)
                        elif cname == "rushing" and tname == "CAR":
                            p["carries"] = _num(stat)
                        elif cname == "rushing" and tname == "YDS":
                            p["rushing_yards"] = _num(stat)
                        elif cname == "receiving" and tname == "REC":
                            p["receptions"] = _num(stat)
                        elif cname == "receiving" and tname == "YDS":
                            p["receiving_yards"] = _num(stat)
            for pid, p in players.items():
                rows.append({"player_id": pid, "team": t["team"], "opponent": opp,
                             "is_home": int(t.get("homeAway") == "home"),
                             "season": season, "week": week, "game_id": str(g["id"]), **p})
    return rows


def load_cfb_season(season: int, key: str, max_week: int = 16, use_cache: bool = True) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"cfb_{season}.pkl")
    # Past seasons never change, so cache them. Always refresh the current season.
    if use_cache and season < current_season() and os.path.exists(path):
        return pd.read_pickle(path)

    rows = []
    for week in range(1, max_week + 1):
        games = _get("/games/players", {"year": season, "week": week, "seasonType": "regular"}, key)
        if not games:
            continue
        rows += _parse_games(games, season, week)
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    roster = pd.DataFrame(_get("/roster", {"year": season}, key))
    if not roster.empty:
        roster = roster.assign(player_id=roster["id"].astype(str))[["player_id", "position"]]
        df = df.merge(roster.drop_duplicates("player_id"), on="player_id", how="left")
    else:
        df["position"] = None

    lines = _get("/lines", {"year": season, "seasonType": "regular"}, key)
    lrows = []
    for g in lines:
        ls = [l for l in g.get("lines", []) if l.get("spread") is not None]
        if ls:
            # CFBD spread is from the home perspective: negative = home favored.
            lrows.append({"game_id": str(g["id"]),
                          "home_spread": -_num(ls[0]["spread"]),
                          "game_total": _num(ls[0].get("overUnder"))})
    lines_df = pd.DataFrame(lrows, columns=["game_id", "home_spread", "game_total"])
    df = df.merge(lines_df, on="game_id", how="left")
    df["team_spread"] = df["home_spread"].where(df["is_home"] == 1, -df["home_spread"])

    df.to_pickle(path)
    return df


def load_cfb(seasons: list[int], key: str) -> pd.DataFrame:
    df = pd.concat([load_cfb_season(s, key) for s in seasons], ignore_index=True)
    df["pos_group"] = df["position"].map(POS_GROUP)
    df = df[df["pos_group"].notna()].copy()
    df["league"] = "cfb"
    for c in STAT_COLS:
        df[c] = pd.to_numeric(df.get(c), errors="coerce").astype(float)
    # Box scores list only players who recorded a stat; blank = 0 in that category.
    for c in ["attempts", "completions", "passing_yards", "carries", "rushing_yards", "receptions", "receiving_yards"]:
        df[c] = df[c].fillna(0.0)
    df["targets"] = float("nan")  # not available in CFB box scores
    cols = ["league", "player_id", "player_name", "position", "pos_group", "team", "opponent",
            "season", "week", "game_id", "is_home", "team_spread", "game_total"] + STAT_COLS
    return df[cols].reset_index(drop=True)
