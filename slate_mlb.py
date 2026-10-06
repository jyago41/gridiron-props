"""MLB version of the slate builder: props -> players -> projections -> scored legs."""
import numpy as np
import pandas as pd

from config import MARKETS
from features import build_features
from slate import norm_name


def _same_team(a: str, b: str) -> bool:
    a, b = str(a).lower(), str(b).lower()
    return a == b or a.split()[-1] == b.split()[-1]   # "Athletics" vs "Oakland Athletics"


def match_schedule(games: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """Attach MLB gamePk + probable pitchers to each sportsbook event (handles doubleheaders)."""
    out = []
    for ev in games.itertuples():
        c = sched[sched.home_team.map(lambda t: _same_team(t, ev.home_team))
                  & sched.away_team.map(lambda t: _same_team(t, ev.away_team))]
        if c.empty:
            continue
        gap = (pd.to_datetime(c["start"], utc=True) - pd.Timestamp(ev.commence_time)).abs()
        g = c.loc[gap.idxmin()]
        out.append({**ev._asdict(), "game_pk": g.game_pk, "game_date": g.game_date,
                    "postseason": int(g.get("game_type", "R") in {"F", "D", "L", "W"}),
                    "game_number": g.game_number, "home_pp": g.home_pp, "away_pp": g.away_pp,
                    "home": g.home_team, "away": g.away_team})
    return pd.DataFrame(out)


def lineup_status(player_id: str, side: str, lu: dict, probable) -> str:
    """Batters: their posted lineup spot. Pitchers: whether they're the listed starter."""
    pid = int(player_id[1:])
    if player_id.startswith("P"):
        if pd.isna(probable):
            return "Starter not announced"
        return "Probable starter" if int(probable) == pid else "Not the listed starter"
    order = lu.get(side, [])
    if not order:
        return "Lineup not posted yet"
    return f"Batting {order.index(pid) + 1}" if pid in order else "Not in lineup"


BLOCKED = {"Not in lineup", "Not the listed starter"}


def build_slate_mlb(history, games, props, models, model_weight=0.5, sched=None, get_lineups=None) -> pd.DataFrame:
    if sched is None:
        from data_mlb import schedule
        d = pd.to_datetime(games["commence_time"]).dt.tz_convert("America/New_York").dt.date
        sched = schedule(str(d.min() - pd.Timedelta(days=1)), str(d.max() + pd.Timedelta(days=1)), probables=True)
    ev = match_schedule(games, sched)
    if ev.empty:
        return pd.DataFrame()
    if get_lineups is None:
        from data_mlb import lineups as get_lineups
    posted = {str(g.game_pk): get_lineups(g.game_pk) for g in ev.itertuples()}
    side_pp = {str(g.game_pk): {"home": g.home_pp, "away": g.away_pp, "home_team": g.home} for g in ev.itertuples()}

    latest = history.sort_values("game_order").drop_duplicates("player_id", keep="last")
    latest = latest.assign(key=latest["player_name"].map(norm_name))
    future, links, extra = [], [], []
    for g in ev.itertuples():
        order = int(str(g.game_date).replace("-", "")) * 10 + int(g.game_number or 1)
        base = {"league": "mlb", "season": int(str(g.game_date)[:4]), "week": 0, "game_id": str(g.game_pk),
                "game_order": order, "game_date": str(g.game_date), "postseason": g.postseason}
        for team, opp, home in ((g.home, g.away, 1), (g.away, g.home, 0)):
            extra.append({"team": team, "opponent": opp, "game_id": str(g.game_pk), "game_order": order, "is_home": home})

        def add(row, team):
            opp = g.away if team == g.home else g.home
            future.append({**{c: row[c] for c in ["player_id", "player_name", "position", "pos_group"]},
                           **base, "team": team, "opponent": opp, "is_home": int(team == g.home)})

        # probable starters (even without props) so batters see who they're facing
        for team, pp in ((g.home, g.home_pp), (g.away, g.away_pp)):
            if pd.notna(pp):
                r = latest[latest.player_id == f"P{int(pp)}"]
                if len(r):
                    add({**r.iloc[0].to_dict(), "pos_group": "SP"}, team)
        for name, mkts in props[props.event_id == g.event_id].groupby("player")["market"]:
            roles = {MARKETS[m]["role"] for m in mkts}
            for role in roles:
                prefix = "B" if role == "B" else "P"
                cand = latest[(latest.key == norm_name(name)) & latest.player_id.str.startswith(prefix)
                              & latest.team.map(lambda t: _same_team(t, g.home) or _same_team(t, g.away))]
                if cand.empty:
                    continue
                r = cand.iloc[0].to_dict()
                team = g.home if _same_team(r["team"], g.home) else g.away
                add({**r, "pos_group": "SP" if role == "SP" else "B"}, team)
                links.append({"event_id": g.event_id, "player": name, "role": role, "player_id": r["player_id"],
                              "game": f"{g.away_team} @ {g.home_team}", "commence_time": g.commence_time,
                              "game_ref": str(g.game_pk), "season": base["season"]})
    if not links:
        return pd.DataFrame()

    fut = pd.DataFrame(future).drop_duplicates("player_id")
    feat, feats = build_features(pd.concat([history, fut], ignore_index=True), pd.DataFrame(extra))
    fut_ids = set(fut.player_id)
    ff = feat[feat.game_id.isin(set(fut.game_id)) & feat.player_id.isin(fut_ids) & feat["pa"].isna() & feat["p_bf"].isna()]
    ff = ff.drop_duplicates("player_id").set_index("player_id")

    legs = props.merge(pd.DataFrame(links), on=["event_id", "player"])
    legs = legs[legs.apply(lambda r: MARKETS[r.market]["role"] == r.role, axis=1)]
    rows = []
    for r in legs.itertuples():
        m = models.get(r.market)
        if m is None or r.player_id not in ff.index:
            continue
        x = ff.loc[[r.player_id]]
        pred = float(m.predict(x)[0])
        p_model = float(m.prob_over(pred, r.line)[0])
        p_over = model_weight * p_model + (1 - model_weight) * r.p_market_over
        info = side_pp[r.game_ref]
        side_ = "home" if x["team"].iloc[0] == info["home_team"] else "away"
        lu = lineup_status(r.player_id, side_, posted[r.game_ref], info[side_])
        for side, p, dec, book, pm in (("Over", p_over, r.over_dec, r.over_book, p_model),
                                       ("Under", 1 - p_over, r.under_dec, r.under_book, 1 - p_model)):
            rows.append({"league": "MLB", "event_id": r.event_id, "player_id": r.player_id, "season": r.season,
                         "week": 0, "game_ref": r.game_ref, "game": r.game, "commence_time": r.commence_time,
                         "player": r.player, "team": x["team"].iloc[0], "pos_group": x["pos_group"].iloc[0],
                         "market": r.market, "market_label": MARKETS[r.market]["label"], "side": side,
                         "line": r.line, "projection": round(pred, 2), "p_model": pm,
                         "p_market": r.p_market_over if side == "Over" else 1 - r.p_market_over,
                         "p_final": p, "decimal": dec, "book": book, "ev": p * dec - 1, "lineup": lu})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["disagreement"] = (out["p_model"] - out["p_market"]).abs()
    return out.sort_values("ev", ascending=False).drop_duplicates(["event_id", "player", "market"]).reset_index(drop=True)
