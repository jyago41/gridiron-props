"""Turn this week's games + props into scored betting legs using the trained models."""
import re
import unicodedata
import numpy as np
import pandas as pd

from config import MARKETS
from features import build_features

NFL_TEAMS = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Los Angeles Rams": "LA", "Los Angeles Chargers": "LAC", "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "Seattle Seahawks": "SEA",
    "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


# statuses that mean a player won't play (their props are usually voided)
UNAVAILABLE = {"Out", "Doubtful", "Injured reserve", "Inactive", "Released", "Retired", "Exempt"}


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()   # José -> Jose
    s = re.sub(r"[^a-z ]", "", s.lower().replace("-", " "))
    return " ".join(re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s).split())


def resolve_team(odds_name: str, league: str, known: set[str]) -> str | None:
    if league == "nfl" or odds_name in NFL_TEAMS:
        return NFL_TEAMS.get(odds_name, odds_name if odds_name in known else None)
    # CFB: "Ohio State Buckeyes" -> "Ohio State" (longest known school name that prefixes it)
    hits = [t for t in known if odds_name == t or odds_name.startswith(t + " ")]
    return max(hits, key=len) if hits else None


def next_game_week(history: pd.DataFrame, games: pd.DataFrame) -> tuple[int, int]:
    """Season/week label for the upcoming slate. If history ends in a prior season, this is
    week 1 of the new season (season-to-date features then correctly start empty)."""
    last_season = int(history["season"].max())
    kickoff = pd.to_datetime(games["commence_time"]).min()
    slate_season = kickoff.year if kickoff.month >= 8 else kickoff.year - 1
    if slate_season > last_season:
        return slate_season, 1
    return last_season, int(history.loc[history.season == last_season, "week"].max()) + 1


def build_slate(history: pd.DataFrame, games: pd.DataFrame, props: pd.DataFrame, models: dict,
                league: str, model_weight: float = 0.5) -> pd.DataFrame:
    """model_weight blends model probability with the sharp no-vig market probability.
    Books are efficient; trusting the model 100% overstates edges. 0.3-0.5 is sane."""
    if games.empty or props.empty:
        return pd.DataFrame()
    if league == "mlb":
        from slate_mlb import build_slate_mlb
        return build_slate_mlb(history, games, props, models, model_weight)
    known = set(history["team"].unique())
    games = games.assign(home=games["home_team"].map(lambda t: resolve_team(t, league, known)),
                         away=games["away_team"].map(lambda t: resolve_team(t, league, known)))
    games = games.dropna(subset=["home", "away"])

    latest = history.sort_values(["season", "week"]).drop_duplicates("player_id", keep="last")
    latest = latest.assign(key=latest["player_name"].map(norm_name))
    season, next_week = next_game_week(history, games)

    # Map each prop's player name to a player on one of the two teams in that game
    future, links = [], []
    for ev in games.itertuples():
        ev_props = props[props.event_id == ev.event_id]
        for name in ev_props["player"].unique():
            cand = latest[(latest.key == norm_name(name)) & latest.team.isin([ev.home, ev.away])]
            if cand.empty:
                continue
            p = cand.iloc[0]
            is_home = int(p.team == ev.home)
            spread = ev.home_spread if is_home else -ev.home_spread
            future.append({**{c: p[c] for c in ["league", "player_id", "player_name", "position", "pos_group", "team"]},
                           "opponent": ev.away if is_home else ev.home, "season": season, "week": next_week,
                           "game_id": ev.event_id, "is_home": is_home, "team_spread": spread, "game_total": ev.game_total})
            links.append({"event_id": ev.event_id, "player": name, "player_id": p.player_id,
                          "game": f"{ev.away_team} @ {ev.home_team}", "commence_time": ev.commence_time})
    if not future:
        return pd.DataFrame()

    fut = pd.DataFrame(future).drop_duplicates("player_id")
    # this week's injury report + IR/inactive lists: who's out, and whose opportunity is up for grabs
    status, absences = {}, None
    if league == "nfl":
        try:
            from data_nfl import load_availability
            avail = load_availability(season, next_week)
            status = dict(zip(avail.player_id, avail.status))
            away = avail[avail.status.isin(UNAVAILABLE)]
            absences = away.assign(game_order=season * 100 + next_week)[["player_id", "team", "game_order"]]
        except Exception as e:
            print(f"[nfl] availability skipped: {e}")
    feat, _ = build_features(pd.concat([history, fut], ignore_index=True), absences=absences)
    fut_feat = feat[(feat.season == season) & (feat.week == next_week) & feat.player_id.isin(fut.player_id)]
    fut_feat = fut_feat.drop_duplicates("player_id").set_index("player_id")

    legs = props.merge(pd.DataFrame(links), on=["event_id", "player"])
    rows = []
    for r in legs.itertuples():
        m = models.get(r.market)
        if m is None or r.player_id not in fut_feat.index:
            continue
        x = fut_feat.loc[[r.player_id]]
        pred = float(m.predict(x)[0])
        p_model = float(m.prob_over(pred, r.line)[0])
        p_over = model_weight * p_model + (1 - model_weight) * r.p_market_over
        for side, p, dec, book, pm in (("Over", p_over, r.over_dec, r.over_book, p_model),
                                       ("Under", 1 - p_over, r.under_dec, r.under_book, 1 - p_model)):
            if pd.isna(dec):          # milestone lines are usually Over-only
                continue
            rows.append({"league": league.upper(), "event_id": r.event_id, "player_id": r.player_id,
                         "season": season, "week": next_week, "game": r.game, "commence_time": r.commence_time,
                         "player": r.player, "team": x["team"].iloc[0], "pos_group": x["pos_group"].iloc[0], "market": r.market,
                         "market_label": MARKETS[r.market]["label"], "side": side, "line": r.line,
                         "projection": round(pred, 1), "p_model": pm,
                         "p_market": r.p_market_over if side == "Over" else 1 - r.p_market_over,
                         "p_final": p, "decimal": dec, "book": book, "ev": p * dec - 1,
                         "status": status.get(r.player_id, "Active") if league == "nfl" else "",
                         "alt": bool(getattr(r, "alt", False))})
    out = pd.DataFrame(rows)
    # Sharp books rarely miss by 25+ points of probability. When the model disagrees that much,
    # it's usually missing news (injury, benching, role change). Flag it; parlays skip these.
    out["disagreement"] = (out["p_model"] - out["p_market"]).abs()
    # keep only the better side of each prop
    return out.sort_values("ev", ascending=False).reset_index(drop=True)


PROJ_MARKETS = ["pass_yds", "pass_cmp", "rush_yds", "rec_yds", "receptions"]


def project_players(history: pd.DataFrame, games: pd.DataFrame, models: dict, league: str,
                    qs=(0.25, 0.75)) -> pd.DataFrame:
    """Projections for every offensive regular in the given games (no sportsbook data needed).
    Regulars: QB/RB/WR/TE who played for the team in one of its last 3 games and had real usage;
    players listed Out / Doubtful / IR are left off."""
    known = set(history["team"].unique())
    games = games.assign(home=games["home_team"].map(lambda t: resolve_team(t, league, known)),
                         away=games["away_team"].map(lambda t: resolve_team(t, league, known))).dropna(subset=["home", "away"])
    if games.empty:
        return pd.DataFrame()
    season, next_week = next_game_week(history, games)
    status, absences = {}, None
    if league == "nfl":
        try:
            from data_nfl import load_availability
            avail = load_availability(season, next_week)
            status = dict(zip(avail.player_id, avail.status))
            absences = (avail[avail.status.isin(UNAVAILABLE)].assign(game_order=season * 100 + next_week)
                        [["player_id", "team", "game_order"]])
        except Exception as e:
            print(f"[nfl] availability skipped: {e}")

    h = history.assign(game_order=history["season"] * 100 + history["week"])
    current_team = h.sort_values("game_order").drop_duplicates("player_id", keep="last").set_index("player_id")["team"]
    future, meta = [], {}
    for ev in games.itertuples():
        for team, opp, home in ((ev.home, ev.away, 1), (ev.away, ev.home, 0)):
            t = h[h.team == team]
            last3 = sorted(t["game_order"].unique())[-3:]
            r = t[t.game_order.isin(last3)]
            use = r.groupby("player_id").agg(targets=("targets", "mean"), carries=("carries", "mean"),
                                             attempts=("attempts", "mean"))
            latest = r.sort_values("game_order").drop_duplicates("player_id", keep="last").set_index("player_id")
            for pid, u in use.iterrows():
                p = latest.loc[pid]
                if current_team.get(pid) != team or status.get(pid) in UNAVAILABLE:
                    continue
                if p["pos_group"] == "QB" and not (u.attempts >= 10):
                    continue
                if p["pos_group"] != "QB" and not (u.targets >= 1 or u.carries >= 2):
                    continue
                spread = ev.home_spread if home else -ev.home_spread
                future.append({"league": league, "player_id": pid, "player_name": p["player_name"],
                               "position": p["position"], "pos_group": p["pos_group"], "team": team,
                               "opponent": opp, "season": season, "week": next_week, "game_id": ev.event_id,
                               "is_home": home, "team_spread": spread, "game_total": ev.game_total})
                meta[pid] = f"{ev.away_team} @ {ev.home_team}"
    if not future:
        return pd.DataFrame()
    fut = pd.DataFrame(future).drop_duplicates("player_id")
    feat, _ = build_features(pd.concat([history, fut], ignore_index=True), absences=absences)
    ff = feat[(feat.season == season) & (feat.week == next_week) & feat.player_id.isin(fut.player_id)]
    ff = ff.drop_duplicates("player_id").reset_index(drop=True)

    out = ff[["player_id", "player_name", "team", "opponent", "pos_group"]].copy()
    out["game"] = out.player_id.map(meta)
    out["status"] = out.player_id.map(lambda p: status.get(p, "Active")) if league == "nfl" else ""
    for mk in PROJ_MARKETS:
        m = models.get(mk)
        if m is None:
            continue
        pred = m.predict(ff)
        lo_hi = m.quantiles(pred, qs)
        show = ff["pos_group"].isin(MARKETS[mk]["positions"]).to_numpy().copy()
        if mk == "rush_yds":       # receivers only if they actually get carries
            show &= ~(ff["pos_group"].isin(["WR", "TE"]) & (ff["carries_r5"].fillna(0) < 1)).values
        out[mk] = np.where(show, pred, np.nan)
        out[f"{mk}_lo"] = np.where(show, lo_hi[:, 0], np.nan)
        out[f"{mk}_hi"] = np.where(show, lo_hi[:, 1], np.nan)
    order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
    return out.assign(_o=out.pos_group.map(order)).sort_values(["game", "team", "_o", "rec_yds", "rush_yds"],
                                                                ascending=[True, True, True, False, False]).drop(columns="_o")
