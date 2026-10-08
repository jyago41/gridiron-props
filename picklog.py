"""Pick tracking. Every run with --log saves picks; `python grade.py` fills in results after
the games and prints your record. This is how you find out if the model actually has an edge."""
import os
from datetime import datetime
import pandas as pd

from config import MARKETS

DRIVE = "/content/drive/MyDrive/gridiron_picks"
LOG_PATH = os.path.join(DRIVE, "picks.csv") if os.path.isdir("/content/drive/MyDrive") else "picks.csv"
COLS = ["logged_at", "kind", "parlay_id", "league", "season", "week", "game_ref", "event_id", "commence_time",
        "game", "player", "player_id", "market", "side", "line", "decimal", "book", "p_final",
        "p_close", "clv_ev", "close_at", "actual", "result"]


def save_picks(singles: pd.DataFrame, parlays: list[dict], path=LOG_PATH) -> int:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = []
    for _, l in singles.iterrows():
        rows.append({**l.to_dict(), "kind": "single", "parlay_id": ""})
    for i, pl in enumerate(parlays, 1):
        pid = f"{now}-P{i}"
        for l in pl["legs"]:
            rows.append({**l, "kind": "parlay_leg", "parlay_id": pid})
    if not rows:
        return 0
    df = pd.DataFrame(rows).assign(logged_at=now, actual=None, result=None).reindex(columns=COLS)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    print(f"\nSaved {len(df)} picks to {path}")
    return len(df)


def load_log(path=LOG_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLS)
    log = pd.read_csv(path, dtype={"result": object, "player_id": str, "parlay_id": str, "game_ref": str,
                                   "event_id": str, "close_at": str})
    for c in COLS:                      # older logs: add any new columns
        if c not in log:
            log[c] = None
    for c in ("p_close", "clv_ev"):
        log[c] = pd.to_numeric(log[c], errors="coerce")
    log["actual"] = log["actual"].astype(float)
    return log


def capture_closing(odds_key, bookmakers=None, path=LOG_PATH, hours_ahead=36):
    """Closing line value: re-price every saved pick whose game starts in the next `hours_ahead` hours.
    clv_ev = (the market's fair chance now) x (the price you got) - 1. Positive means you beat the
    closing line, the best early sign that picks have a real edge. Run it shortly before games start;
    running it again overwrites with the newer (closer to kickoff) number."""
    from odds import OddsClient, consolidate_props
    log = load_log(path)
    if log.empty:
        return 0, 0, None
    ct = pd.to_datetime(log["commence_time"], utc=True, errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    todo = log["result"].isna() & log["event_id"].notna() & (ct > now) & (ct < now + pd.Timedelta(hours=hours_ahead))
    client = OddsClient(odds_key, bookmakers=bookmakers)
    found, missing = 0, 0
    for (lg, eid), rows in log[todo].groupby(["league", "event_id"]):
        try:
            props = consolidate_props(client.player_props(str(lg).lower(), eid, sorted(rows.market.unique()), True))
        except Exception:
            continue
        lookup = {} if props.empty else {(r.player, r.market, float(r.line)): r.p_market_over for r in props.itertuples()}
        for i in rows.index:
            p_over = lookup.get((log.at[i, "player"], log.at[i, "market"], float(log.at[i, "line"])))
            if p_over is None:
                missing += 1     # that exact line is no longer offered
                continue
            p = p_over if log.at[i, "side"] == "Over" else 1 - p_over
            log.at[i, "p_close"] = p
            log.at[i, "clv_ev"] = p * float(log.at[i, "decimal"]) - 1
            log.at[i, "close_at"] = now.strftime("%Y-%m-%d %H:%M UTC")
            found += 1
    log.to_csv(path, index=False)
    return found, missing, client.remaining


def grade_log(path=LOG_PATH) -> tuple[pd.DataFrame, dict]:
    """Fill in results for finished games. Returns (log, summary)."""
    log = load_log(path)
    summary = {"singles_w": 0, "singles_l": 0, "singles_profit": 0.0,
               "parlays_w": 0, "parlays_l": 0, "parlays_profit": 0.0, "pending": 0}
    if log.empty:
        return log, summary
    todo = log["result"].isna() & (log["league"] == "NFL")
    if todo.any():
        import nflreadpy as nfl
        stats = nfl.load_player_stats(sorted(log.loc[todo, "season"].astype(int).unique().tolist())).to_pandas()
        stats["player_id"] = stats["player_id"].astype(str)
        stats = stats.set_index(["player_id", "season", "week"])
        for i in log.index[todo]:
            key = (str(log.at[i, "player_id"]), int(log.at[i, "season"]), int(log.at[i, "week"]))
            if key not in stats.index:
                continue  # game not played yet / stats not published yet (usually next morning)
            vals = stats.loc[[key], MARKETS[log.at[i, "market"]]["stat"]]
            actual = float(vals.iloc[0])
            line, side = float(log.at[i, "line"]), log.at[i, "side"]
            log.at[i, "actual"] = actual
            log.at[i, "result"] = "push" if actual == line else (
                "win" if (actual > line) == (side == "Over") else "loss")
        log.to_csv(path, index=False)

    todo = log["result"].isna() & (log["league"] == "MLB") & log["game_ref"].notna()
    if todo.any():
        from data_mlb import fetch_box, parse_box, schedule
        pks = sorted(log.loc[todo, "game_ref"].astype(str).unique())
        final = schedule(None, None, game_pks=pks)
        final = final[final.state == "Final"]
        for g in final.itertuples():
            box = {r["player_id"]: r for r in parse_box(fetch_box(g.game_pk), g.game_pk, g.game_date, g.game_number, g.game_type)}
            for i in log.index[todo & (log["game_ref"].astype(str) == str(g.game_pk))]:
                r = box.get(str(log.at[i, "player_id"]))
                stat = MARKETS[log.at[i, "market"]]["stat"]
                # a batter who didn't play (or a scratched pitcher) is a void bet at most books
                if r is None or pd.isna(r.get(stat)):
                    log.at[i, "result"] = "void"
                    continue
                actual, line, side = float(r[stat]), float(log.at[i, "line"]), log.at[i, "side"]
                log.at[i, "actual"] = actual
                log.at[i, "result"] = "push" if actual == line else (
                    "win" if (actual > line) == (side == "Over") else "loss")
        log.to_csv(path, index=False)

    done = log[log["result"].notna()]
    singles = done[done.kind == "single"]
    if len(singles):
        summary["singles_w"] = int((singles.result == "win").sum())
        summary["singles_l"] = int((singles.result == "loss").sum())
        summary["singles_profit"] = float(((singles.result == "win") * (singles.decimal - 1)
                                           - (singles.result == "loss") * 1.0).sum())
    legs = done[done.kind == "parlay_leg"]
    if len(legs):
        g = legs.groupby("parlay_id")
        total = log[log.kind == "parlay_leg"].groupby("parlay_id").size()
        lost = g.apply(lambda d: (d.result == "loss").any())
        finished = lost | (g.size() == total.reindex(g.size().index))  # one loss settles a parlay
        lost, dec = lost[finished], g["decimal"].prod()[finished]
        summary["parlays_w"], summary["parlays_l"] = int((~lost).sum()), int(lost.sum())
        summary["parlays_profit"] = float(((~lost) * (dec - 1) - lost).sum())
    summary["pending"] = int(log["result"].isna().sum())
    return log, summary


def grade(path=LOG_PATH):
    if not os.path.exists(path):
        print("No picks logged yet. Run with --log first.")
        return
    _, s = grade_log(path)
    w, l = s["singles_w"], s["singles_l"]
    if w + l:
        print(f"Single bets: {w}-{l}  hit rate {w / (w + l):.1%}  profit {s['singles_profit']:+.2f} units "
              f"(1 unit per bet; break-even hit rate at -110 is 52.4%)")
    if s["parlays_w"] + s["parlays_l"]:
        print(f"Parlays: {s['parlays_w']}-{s['parlays_l']}  profit {s['parlays_profit']:+.2f} units")
    if s["pending"]:
        print(f"{s['pending']} picks still waiting on results (stats post the morning after games).")
