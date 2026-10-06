"""MLB features. Like football, every feature uses only games before the row's game."""
import numpy as np
import pandas as pd

from features import _rolling_prior

B_STATS = ["pa", "hits", "total_bases", "hrr", "hr", "bb", "so", "runs", "rbi", "lineup_spot", "started"]
P_STATS = ["p_outs", "p_strikeouts", "p_hits", "p_bf", "p_er", "p_bb", "p_pitches"]


def _prior_sum(df, key, col, w):
    s = df.groupby(key, sort=False)[col].shift(1)
    return s.groupby(df[key], sort=False).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)


def _player_form(d, stats, windows):
    d = d.sort_values(["player_id", "game_order"]).reset_index(drop=True)
    feats = []
    for col in stats:
        for w in windows:
            d[f"{col}_r{w}"] = _rolling_prior(d, ["player_id"], col, w)
            feats.append(f"{col}_r{w}")
        prev = d.groupby(["player_id", "season"], sort=False)[col].shift(1)
        d[f"{col}_szn"] = prev.groupby([d["player_id"], d["season"]]).transform(lambda s: s.expanding().mean())
        feats.append(f"{col}_szn")
    d["games_played"] = d.groupby("player_id").cumcount()
    return d, feats + ["games_played"]


def build_features_mlb(df: pd.DataFrame, extra_games: pd.DataFrame | None = None):
    df = df.copy()
    B, P = df[df.pos_group == "B"].copy(), df[df.pos_group.isin(["SP", "RP"])].copy()

    # ---- batter form: counts plus per-plate-appearance rates over 15 / 40 games ----
    B, bfe = _player_form(B, B_STATS, (5, 15))
    for col in ["hits", "total_bases", "hrr", "so", "bb", "hr"]:
        for w in (15, 40):
            B[f"{col}_per_pa{w}"] = _prior_sum(B, "player_id", col, w) / _prior_sum(B, "player_id", "pa", w).replace(0, np.nan)
            bfe.append(f"{col}_per_pa{w}")

    # ---- pitcher form (starts only for SP features) ----
    P, pfe = _player_form(P, P_STATS, (3, 10))
    for col in ["p_strikeouts", "p_hits", "p_bb", "p_er"]:
        P[f"{col}_per_bf10"] = _prior_sum(P, "player_id", col, 10) / _prior_sum(P, "player_id", "p_bf", 10).replace(0, np.nan)
        pfe.append(f"{col}_per_bf10")
    P["p_bf_r5"] = _rolling_prior(P, ["player_id"], "p_bf", 5)        # usage for eligibility
    # days since this pitcher last pitched: short rest in the playoffs means a shorter outing
    gd = pd.to_datetime(P["game_date"])
    P["rest_days"] = (gd - gd.groupby(P["player_id"]).shift(1)).dt.days.clip(upper=30)
    pfe.append("rest_days")
    for col in P_STATS:
        P[f"{col}_r5"] = _rolling_prior(P, ["player_id"], col, 5)       # baseline for validation / SGP

    # ---- team-game table: offense, defense, ballpark ----
    tg = (B.groupby(["team", "opponent", "game_id", "game_order", "is_home"], as_index=False)
            [["runs", "pa", "so", "hits", "total_bases"]].sum(min_count=1))
    if extra_games is not None and len(extra_games):
        tg = pd.concat([tg, extra_games[["team", "opponent", "game_id", "game_order", "is_home"]]], ignore_index=True)
        tg = tg.drop_duplicates(["team", "game_id"])
    tg = tg.sort_values("game_order").reset_index(drop=True)
    for col in ["runs", "pa", "so", "hits", "total_bases"]:
        tg[f"off_{col}_s15"] = _prior_sum(tg, "team", col, 15)
    tg["off_runs_pg"] = tg["off_runs_s15"] / 15
    tg["off_k_rate"] = tg["off_so_s15"] / tg["off_pa_s15"].replace(0, np.nan)
    tg["off_hit_rate"] = tg["off_hits_s15"] / tg["off_pa_s15"].replace(0, np.nan)
    tg["off_tb_rate"] = tg["off_total_bases_s15"] / tg["off_pa_s15"].replace(0, np.nan)
    # runs allowed by each team = runs its opponents scored against it
    allowed = tg[["opponent", "game_id", "game_order", "runs"]].rename(columns={"opponent": "def_team", "runs": "ra"})
    allowed = allowed.sort_values("game_order").reset_index(drop=True)
    allowed["def_ra_pg"] = _prior_sum(allowed, "def_team", "ra", 15) / 15
    # ballpark: average total runs in the home team's park, last 40 home games
    home = tg[tg.is_home == 1][["team", "game_id", "game_order", "runs"]].merge(
        tg[tg.is_home == 0][["game_id", "runs"]].rename(columns={"runs": "runs_away"}), on="game_id", how="left")
    home["total"] = home["runs"] + home["runs_away"]
    home = home.sort_values("game_order").reset_index(drop=True)
    home["park_runs"] = _rolling_prior(home, ["team"], "total", 40)
    park = home[["game_id", "park_runs"]]
    team_cols = ["off_runs_pg", "off_k_rate", "off_hit_rate", "off_tb_rate"]
    T = tg[["team", "game_id"] + team_cols]

    # ---- batters: own offense, opponent defense, opposing starting pitcher, park ----
    B = B.merge(T, on=["team", "game_id"], how="left")
    B = B.merge(allowed[["def_team", "game_id", "def_ra_pg"]].rename(columns={"def_team": "opponent"}),
                on=["opponent", "game_id"], how="left")
    sp = P[P.pos_group == "SP"][["team", "game_id", "p_strikeouts_per_bf10", "p_hits_per_bf10",
                                  "p_er_per_bf10", "p_bb_per_bf10", "p_outs_r10"]]
    sp = sp.rename(columns={"team": "opponent", "p_strikeouts_per_bf10": "opp_sp_k_rate",
                            "p_hits_per_bf10": "opp_sp_hit_rate", "p_er_per_bf10": "opp_sp_er_rate",
                            "p_bb_per_bf10": "opp_sp_bb_rate", "p_outs_r10": "opp_sp_outs"})
    B = B.merge(sp.drop_duplicates(["opponent", "game_id"]), on=["opponent", "game_id"], how="left")
    B = B.merge(park, on="game_id", how="left")
    b_ctx = team_cols + ["def_ra_pg", "opp_sp_k_rate", "opp_sp_hit_rate", "opp_sp_er_rate",
                         "opp_sp_bb_rate", "opp_sp_outs", "park_runs", "is_home", "postseason"]

    # ---- pitchers: opposing lineup's recent strikeout / contact rates, park ----
    opp = T.rename(columns={"team": "opponent", **{c: f"opp_{c}" for c in team_cols}})
    P = P.merge(opp, on=["opponent", "game_id"], how="left").merge(park, on="game_id", how="left")
    p_ctx = [f"opp_{c}" for c in team_cols] + ["park_runs", "is_home", "postseason"]

    out = pd.concat([B, P], ignore_index=True)
    return out, {"B": bfe + b_ctx, "SP": pfe + ["p_bf_r5"] + p_ctx}
