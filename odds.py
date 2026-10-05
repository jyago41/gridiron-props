"""Live game lines + player props from The Odds API (the-odds-api.com).

Cost note: each /events/{id}/odds call costs (number of markets x regions) credits, so a full
NFL Sunday with 5 prop markets is ~16 games x 5 = ~80 credits. The free tier is 500/month.
"""
from datetime import datetime, timedelta, timezone
import time
import numpy as np
import pandas as pd
import requests

from config import MARKETS, ODDS_KEY_TO_MARKET, SPORT_KEYS

BASE = "https://api.the-odds-api.com/v4"


def american_to_decimal(a):
    a = np.asarray(a, dtype=float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / np.abs(a))


def decimal_to_american(d):
    d = float(d)
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


class OddsClient:
    def __init__(self, api_key: str, regions="us", bookmakers: list[str] | None = None):
        self.key, self.regions, self.bookmakers = api_key, regions, bookmakers
        self.remaining = None

    def _get(self, path, **params):
        params = {"apiKey": self.key, "oddsFormat": "american", **params}
        if self.bookmakers:
            params["bookmakers"] = ",".join(self.bookmakers)
        else:
            params["regions"] = self.regions
        for attempt in range(5):
            r = requests.get(f"{BASE}{path}", params=params, timeout=30)
            if r.status_code != 429:
                break
            time.sleep(2 * (attempt + 1))  # rate-limited: wait, then retry
        r.raise_for_status()
        time.sleep(1)  # pause between calls so the free plan doesn't rate-limit us
        self.remaining = r.headers.get("x-requests-remaining")
        return r.json()

    def game_lines(self, league: str, days_ahead: int = 7) -> pd.DataFrame:
        """Upcoming games with consensus spread (home perspective, + = home favored) and total."""
        data = self._get(f"/sports/{SPORT_KEYS[league]}/odds", markets="spreads,totals")
        now = datetime.now(timezone.utc)
        rows = []
        for ev in data:
            start = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
            if not (now < start < now + timedelta(days=days_ahead)):
                continue
            spreads, totals = [], []
            for bk in ev.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    for o in mk.get("outcomes", []):
                        if mk["key"] == "spreads" and o["name"] == ev["home_team"]:
                            spreads.append(-o["point"])
                        elif mk["key"] == "totals" and o["name"] == "Over":
                            totals.append(o["point"])
            rows.append({"event_id": ev["id"], "commence_time": start, "home_team": ev["home_team"],
                         "away_team": ev["away_team"],
                         "home_spread": float(np.median(spreads)) if spreads else np.nan,
                         "game_total": float(np.median(totals)) if totals else np.nan})
        return pd.DataFrame(rows)

    def player_props(self, league: str, event_id: str) -> pd.DataFrame:
        markets = ",".join(m["odds_key"] for m in MARKETS.values())
        data = self._get(f"/sports/{SPORT_KEYS[league]}/events/{event_id}/odds", markets=markets)
        rows = []
        for bk in data.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk["key"] not in ODDS_KEY_TO_MARKET:
                    continue
                for o in mk.get("outcomes", []):
                    if o.get("point") is None or o["name"] not in ("Over", "Under"):
                        continue
                    rows.append({"event_id": event_id, "book": bk["title"],
                                 "market": ODDS_KEY_TO_MARKET[mk["key"]], "player": o.get("description"),
                                 "side": o["name"], "point": float(o["point"]), "price": float(o["price"])})
        return pd.DataFrame(rows)


def consolidate_props(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (event, player, market): consensus line, no-vig market probability,
    and the best available price on each side across books (line shopping)."""
    if raw.empty:
        return raw
    raw = raw.assign(dec=american_to_decimal(raw["price"]))
    key = ["event_id", "player", "market"]
    # consensus line = most commonly posted point
    mode_pt = (raw.groupby(key + ["point"]).size().reset_index(name="n")
                  .sort_values("n", ascending=False).drop_duplicates(key)[key + ["point"]])
    at = raw.merge(mode_pt, on=key + ["point"])

    wide = at.pivot_table(index=key + ["point", "book"], columns="side", values="dec", aggfunc="max").reset_index()
    wide = wide.dropna(subset=["Over", "Under"])
    wide["p_over_novig"] = (1 / wide["Over"]) / (1 / wide["Over"] + 1 / wide["Under"])

    best_over = at[at.side == "Over"].sort_values("dec", ascending=False).drop_duplicates(key)
    best_under = at[at.side == "Under"].sort_values("dec", ascending=False).drop_duplicates(key)
    out = (wide.groupby(key + ["point"], as_index=False)
               .agg(p_market_over=("p_over_novig", "mean"), n_books=("book", "nunique")))
    out = out.merge(best_over[key + ["dec", "book"]].rename(columns={"dec": "over_dec", "book": "over_book"}), on=key)
    out = out.merge(best_under[key + ["dec", "book"]].rename(columns={"dec": "under_dec", "book": "under_book"}), on=key)
    return out.rename(columns={"point": "line"})
