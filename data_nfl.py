"""NFL player-game history from nflverse (via nflreadpy), normalized to the shared schema."""
import pandas as pd
import nflreadpy as nfl

from config import STAT_COLS

POS_GROUP = {"QB": "QB", "RB": "RB", "FB": "RB", "HB": "RB", "WR": "WR", "TE": "TE"}


def load_nfl(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:  # load one season at a time so a not-yet-published season doesn't break everything
        try:
            ps = nfl.load_player_stats([s])
            keep = ["player_id", "player_display_name", "position", "season", "week", "season_type",
                    "game_id", "team", "opponent_team"] + STAT_COLS
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
    for c in STAT_COLS:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return out.reset_index(drop=True)
