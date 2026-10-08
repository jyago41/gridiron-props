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


REGULAR = {"targets": 2.0, "carries": 3.0}     # post-game 5-game averages that make a player a "regular"


def vacated_opportunity(df: pd.DataFrame, absences: pd.DataFrame | None = None) -> pd.DataFrame:
    """For each team-game: how many targets / carries per game belong to regulars who are NOT playing.
    History: a regular who played in one of the team's previous 3 games but has no stats this game.
    Upcoming games: also anyone listed Out / Doubtful / on IR (absences: player_id, team, game_order)."""
    tg = df[["team", "game_order", "season"]].drop_duplicates(["team", "game_order"]).sort_values(["team", "game_order"])
    tg["k"] = tg.groupby("team").cumcount()
    # upcoming games have no box score yet: there, only the injury report / roster lists say who's out
    has_box = (df[["targets", "carries", "attempts"]].notna().any(axis=1)
               .groupby([df["team"], df["game_order"]]).any().rename("has_box").reset_index())
    tg = tg.merge(has_box, on=["team", "game_order"])
    tg["has_box"] = tg["has_box"].astype(bool)
    d = df[["player_id", "team", "game_order", "pos_group", "targets", "carries"]].merge(
        tg[["team", "game_order", "k", "season"]], on=["team", "game_order"])
    d = d.sort_values(["player_id", "game_order"])
    for c in ("targets", "carries"):
        d[f"{c}_post5"] = (d.groupby("player_id")[c].rolling(5, min_periods=1).mean()
                           .reset_index(level=0, drop=True))
    reg = d[(d.targets_post5 >= REGULAR["targets"]) | (d.carries_post5 >= REGULAR["carries"])]
    usage = ["targets_post5", "carries_post5", "pos_group"]
    cand = pd.concat([reg.assign(k_t=reg.k + s, gap=s) for s in (1, 2, 3)], ignore_index=True)
    # only within a season: offseason departures aren't injuries
    cand = cand.merge(tg[["team", "k", "season"]].rename(columns={"k": "k_t", "season": "season_t"}), on=["team", "k_t"])
    cand = cand[cand.season_t == cand.season].drop(columns="season_t")
    cand = cand.merge(tg[["team", "k", "has_box"]].rename(columns={"k": "k_t"}), on=["team", "k_t"])
    cand = cand[cand.has_box].drop(columns="has_box")
    cand = cand.sort_values("gap").drop_duplicates(["player_id", "team", "k_t"])
    if absences is not None and len(absences):
        # same rule as history: only counts if he played for this team within its last 3 games this season
        last = (reg.sort_values("game_order").drop_duplicates(["player_id", "team"], keep="last")
                [["player_id", "team", "k", "season"] + usage].rename(columns={"k": "k_last"}))
        a = absences.merge(tg[~tg.has_box], on=["team", "game_order"]).rename(columns={"k": "k_t"})
        a = a.merge(last, on=["player_id", "team", "season"])
        a = a[(a.k_t - a.k_last).between(1, 3)].drop(columns="k_last")
        cand = pd.concat([cand, a.assign(gap=0)], ignore_index=True).drop_duplicates(["player_id", "team", "k_t"])
    played = d[["player_id", "team", "k"]].rename(columns={"k": "k_t"}).assign(played=1)
    cand = cand.merge(played, on=["player_id", "team", "k_t"], how="left")
    cand = cand[cand.played.isna()]
    out = cand.assign(
        vac_targets=cand.targets_post5, vac_carries=cand.carries_post5,
        vac_targets_wrte=np.where(cand.pos_group.isin(["WR", "TE"]), cand.targets_post5, 0.0),
        vac_carries_rb=np.where(cand.pos_group == "RB", cand.carries_post5, 0.0),
        vac_qb=(cand.pos_group == "QB").astype(float),
    ).groupby(["team", "k_t"], as_index=False)[["vac_targets", "vac_carries", "vac_targets_wrte", "vac_carries_rb", "vac_qb"]].sum()
    out = out.merge(tg[["team", "k", "game_order"]].rename(columns={"k": "k_t"}), on=["team", "k_t"]).drop(columns="k_t")
    return out


def build_features(df: pd.DataFrame, extra_games=None, absences=None):
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

    # --- Opportunity (NFL): target share, air yards share, WOPR, over the last 3 / 5 games ---
    for col in ["target_share", "air_yards_share", "wopr", "receiving_air_yards"]:
        if col in df and df[col].notna().any():
            for w in (3, 5):
                df[f"{col}_r{w}"] = _rolling_prior(df, ["player_id"], col, w)
                feats.append(f"{col}_r{w}")

    # --- Teammates who are out: opportunity up for grabs ---
    vac = vacated_opportunity(df, absences)
    df = df.merge(vac, on=["team", "game_order"], how="left")
    vcols = ["vac_targets", "vac_carries", "vac_targets_wrte", "vac_carries_rb", "vac_qb"]
    df[vcols] = df[vcols].fillna(0.0)
    feats += vcols

    # --- Game environment from the betting market ---
    df["implied_team_total"] = df["game_total"] / 2 + df["team_spread"] / 2
    df["pos_code"] = df["pos_group"].map(POS_CODE)
    feats += ["is_home", "team_spread", "game_total", "implied_team_total", "pos_code", "week"]

    return df, feats
