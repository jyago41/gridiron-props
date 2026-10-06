"""Feature engineering. Every feature uses only games BEFORE the row's game (shift(1)), so
the same code builds training rows and upcoming-game rows without leakage."""
import numpy as np
import pandas as pd

from config import STAT_COLS

WINDOWS = (3, 5)
POS_CODE = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
DEF_STATS = ["passing_yards", "completions", "rushing_yards", "receiving_yards", "receptions"]
TEAM_STATS = ["attempts", "carries", "targets"]


def _rolling_prior(df, keys, col, w):
    shifted = df.groupby(keys, sort=False)[col].shift(1)
    return (shifted.groupby([df[k] for k in keys], sort=False)
            .rolling(w, min_periods=1).mean()
            .reset_index(level=list(range(len(keys))), drop=True))


def build_features(df: pd.DataFrame, extra_games=None):
    if len(df) and df["league"].iloc[0] == "mlb":
        from features_mlb import build_features_mlb
        return build_features_mlb(df, extra_games)
    df = df.copy()
    df["game_order"] = df["season"] * 100 + df["week"]
    df = df.sort_values(["player_id", "game_order"]).reset_index(drop=True)
    feats = []

    # --- Player form: recent averages, volatility, season-to-date, recency-weighted ---
    g = df.groupby("player_id", sort=False)
    for col in STAT_COLS:
        prev = g[col].shift(1)
        for w in WINDOWS:
            name = f"{col}_r{w}"
            df[name] = _rolling_prior(df, ["player_id"], col, w)
            feats.append(name)
        df[f"{col}_ewm"] = prev.groupby(df["player_id"]).transform(lambda s: s.ewm(halflife=3, ignore_na=True).mean())
        df[f"{col}_std5"] = prev.groupby(df["player_id"]).rolling(5, min_periods=2).std().reset_index(level=0, drop=True)
        prev_szn = df.groupby(["player_id", "season"], sort=False)[col].shift(1)  # resets each season
        df[f"{col}_szn"] = prev_szn.groupby([df["player_id"], df["season"]]).transform(lambda s: s.expanding().mean())
        feats += [f"{col}_ewm", f"{col}_std5", f"{col}_szn"]
    df["games_played"] = g.cumcount()
    feats.append("games_played")

    # --- Team volume (pace / play-calling): team totals per game, rolled ---
    team = df.groupby(["team", "game_order"], as_index=False)[TEAM_STATS].sum(min_count=1).sort_values("game_order")
    for col in TEAM_STATS:
        team[f"team_{col}_r5"] = _rolling_prior(team, ["team"], col, 5)
    tcols = [f"team_{c}_r5" for c in TEAM_STATS]
    df = df.merge(team[["team", "game_order"] + tcols], on=["team", "game_order"], how="left")
    feats += tcols
    # Player share of team volume (e.g. target share) over last 5
    for col in TEAM_STATS:
        df[f"{col}_share5"] = df[f"{col}_r5"] / df[f"team_{col}_r5"].replace(0, np.nan)
        feats.append(f"{col}_share5")

    # --- Opponent defense: what this opponent allowed to this position group, last 5 games ---
    d = (df.groupby(["opponent", "pos_group", "game_order"], as_index=False)[DEF_STATS]
           .sum(min_count=1).sort_values("game_order"))
    for col in DEF_STATS:
        d[f"opp_allow_{col}_r5"] = _rolling_prior(d, ["opponent", "pos_group"], col, 5)
    dcols = [f"opp_allow_{c}_r5" for c in DEF_STATS]
    df = df.merge(d[["opponent", "pos_group", "game_order"] + dcols],
                  on=["opponent", "pos_group", "game_order"], how="left")
    feats += dcols

    # --- Game environment from the betting market ---
    df["implied_team_total"] = df["game_total"] / 2 + df["team_spread"] / 2
    df["pos_code"] = df["pos_group"].map(POS_CODE)
    feats += ["is_home", "team_spread", "game_total", "implied_team_total", "pos_code", "week"]

    return df, feats
