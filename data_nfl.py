"""NFL player-game history from nflverse (via nflreadpy), normalized to the shared schema."""
import pandas as pd
import nflreadpy as nfl

from config import STAT_COLS

# opportunity stats: rolled as features (never used as targets)
OPP_COLS = ["target_share", "air_yards_share", "wopr", "receiving_air_yards"]

POS_GROUP = {"QB": "QB", "RB": "RB", "FB": "RB", "HB": "RB", "WR": "WR", "TE": "TE"}


def load_nfl(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:  # load one season at a time so a not-yet-published season doesn't break everything
        try:
            ps = nfl.load_player_stats([s])
            keep = ["player_id", "player_display_name", "position", "season", "week", "season_type",
                    "game_id", "team", "opponent_team"] + STAT_COLS + OPP_COLS
            frames.append(ps.select([c for c in keep if c in ps.columns]).to_pandas())  # 17 of 150 columns
        except Exception as e:
            print(f"[nfl] skipping {s}: {e}")
    ps = pd.concat(frames, ignore_index=True)
    sc = nfl.load_schedules(sorted(ps["season"].unique().tolist()))
    sc = sc.select(["game_id", "home_team", "away_team", "spread_line", "total_line"]).to_pandas()

    ps = ps[ps["season_type"].isin(["REG", "POST"])]
    ps = ps[ps["position"].isin(POS_GROUP)].copy()

    sc = sc[["game_id", "home_team", "away_team", "spread_line", "total_line"]]
    df = ps.merge(sc, on="game_id", how="left")

    df["is_home"] = (df["team"] == df["home_team"]).astype(int)
    # nflverse spread_line is positive when the HOME team is favored.
    # team_spread is from the player's team perspective: positive = favored.
    df["team_spread"] = df["spread_line"].where(df["is_home"] == 1, -df["spread_line"])
    df["game_total"] = df["total_line"]

    out = pd.DataFrame({
        "league": "nfl",
        "player_id": df["player_id"].astype(str),
        "player_name": df["player_display_name"],
        "position": df["position"],
        "pos_group": df["position"].map(POS_GROUP),
        "team": df["team"],
        "opponent": df["opponent_team"],
        "season": df["season"].astype(int),
        "week": df["week"].astype(int),
        "game_id": df["game_id"],
        "is_home": df["is_home"],
        "team_spread": df["team_spread"],
        "game_total": df["game_total"],
    })
    for c in STAT_COLS + OPP_COLS:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype(float) if c in df else float("nan")
    return out.reset_index(drop=True)


AWAY_STATUSES = {"Out": "Out", "Doubtful": "Doubtful"}
ROSTER_AWAY = {"RES": "Injured reserve", "INA": "Inactive", "CUT": "Released", "RET": "Retired",
               "EXE": "Exempt", "TRD": "Traded"}


def load_availability(season: int, week: int) -> pd.DataFrame:
    """Who is unavailable for an upcoming week: player_id, team, status.
    Injury reports (Out / Doubtful / Questionable) come out Wednesday-Friday; weekly rosters catch
    injured reserve and other inactive lists. Questionable players are listed but still count as playing."""
    rows = []
    try:
        inj = nfl.load_injuries([season]).to_pandas()
        inj = inj[inj["week"] == week]
        for r in inj.itertuples():
            if r.report_status in ("Out", "Doubtful", "Questionable"):
                rows.append({"player_id": r.gsis_id, "team": r.team, "status": r.report_status})
    except Exception as e:
        print(f"[nfl] injury report unavailable: {e}")
    try:
        ro = nfl.load_rosters_weekly([season]).to_pandas()
        ro = ro[ro["week"] <= week].sort_values("week").drop_duplicates("gsis_id", keep="last")
        for r in ro[ro["status"].isin(ROSTER_AWAY)].itertuples():
            rows.append({"player_id": r.gsis_id, "team": r.team, "status": ROSTER_AWAY[r.status]})
    except Exception as e:
        print(f"[nfl] weekly rosters unavailable: {e}")
    out = pd.DataFrame(rows, columns=["player_id", "team", "status"])
    # injury report wins over roster status if both exist (it's more recent)
    order = {"Out": 0, "Doubtful": 1, "Questionable": 2}
    out["rank"] = out["status"].map(order).fillna(3)
    return out.sort_values("rank").drop_duplicates("player_id").drop(columns="rank").reset_index(drop=True)
