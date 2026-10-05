"""Pick tracking. Every run with --log saves picks; `python grade.py` fills in results after
the games and prints your record. This is how you find out if the model actually has an edge."""
import os
from datetime import datetime
import pandas as pd

from config import MARKETS

DRIVE = "/content/drive/MyDrive/gridiron_picks"
LOG_PATH = os.path.join(DRIVE, "picks.csv") if os.path.isdir("/content/drive/MyDrive") else "picks.csv"
COLS = ["logged_at", "kind", "parlay_id", "league", "season", "week", "game", "player", "player_id",
        "market", "side", "line", "decimal", "book", "p_final", "actual", "result"]


def save_picks(singles: pd.DataFrame, parlays: list[dict]):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = []
    for _, l in singles.iterrows():
        rows.append({**l.to_dict(), "kind": "single", "parlay_id": ""})
    for i, pl in enumerate(parlays, 1):
        pid = f"{now}-P{i}"
        for l in pl["legs"]:
            rows.append({**l, "kind": "parlay_leg", "parlay_id": pid})
    if not rows:
        return
    df = pd.DataFrame(rows).assign(logged_at=now, actual=None, result=None).reindex(columns=COLS)
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    df.to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)
    print(f"\nSaved {len(df)} picks to {LOG_PATH}")


def grade():
    if not os.path.exists(LOG_PATH):
        print("No picks logged yet. Run with --log first.")
        return
    log = pd.read_csv(LOG_PATH, dtype={"result": object, "player_id": str, "parlay_id": str})
    log["actual"] = log["actual"].astype(float)
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
        log.to_csv(LOG_PATH, index=False)

    done = log[log["result"].notna()]
    singles = done[done.kind == "single"]
    if len(singles):
        w, l = (singles.result == "win").sum(), (singles.result == "loss").sum()
        profit = (singles.result == "win") * (singles.decimal - 1) - (singles.result == "loss") * 1.0
        print(f"Single bets: {w}-{l}  hit rate {w / max(w + l, 1):.1%}  profit {profit.sum():+.2f} units "
              f"(1 unit per bet; break-even hit rate at -110 is 52.4%)")
    legs = done[done.kind == "parlay_leg"]
    if len(legs):
        g = legs.groupby("parlay_id")
        full = g.size() == log[log.kind == "parlay_leg"].groupby("parlay_id").size().reindex(g.size().index)
        res = g.apply(lambda d: "loss" if (d.result == "loss").any() else "win")[full]
        dec = g["decimal"].prod()[full]
        profit = ((res == "win") * (dec - 1) - (res == "loss")).sum()
        print(f"Parlays: {(res == 'win').sum()}-{(res == 'loss').sum()}  profit {profit:+.2f} units")
    pending = log["result"].isna().sum()
    if pending:
        print(f"{pending} picks still waiting on results (stats post the morning after games).")
