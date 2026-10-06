"""Live game lines + player props from The Odds API (the-odds-api.com).

Cost note: each /events/{id}/odds call costs (number of markets x regions) credits, so a full
NFL Sunday with 5 prop markets is ~16 games x 5 = ~80 credits. The free tier is 500/month.
"""
from datetime import datetime, timedelta, timezone
import time
import numpy as np
import pandas as pd
import requests

from config import MARKETS, ODDS_KEY_TO_MARKET, SPORT_KEYS, markets_for

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
        """Upcoming games with moneyline, spread and total: consensus line, no-vig win chance,
        and the best price on each side across books. (3 credits per call.)"""
        data = self._get(f"/sports/{SPORT_KEYS[league]}/odds", markets="h2h,spreads,totals")
        now = datetime.now(timezone.utc)
        rows = []
        for ev in data:
            start = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
            if not (now < start < now + timedelta(days=days_ahead)):
                continue
            home, away = ev["home_team"], ev["away_team"]
            ml, sp, tot = [], [], []
            for bk in ev.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    for o in mk.get("outcomes", []):
                        rec = {"book": bk["title"], "name": o["name"], "price": o["price"], "point": o.get("point")}
                        {"h2h": ml, "spreads": sp, "totals": tot}.get(mk["key"], []).append(rec)
            row = {"event_id": ev["id"], "commence_time": start, "home_team": home, "away_team": away}
            row.update(_moneyline(ml, home, away))
            row.update(_spread(sp, home, away))
            row.update(_total(tot))
            rows.append(row)
        return pd.DataFrame(rows)

    def player_props(self, league: str, event_id: str, markets: list[str] | None = None) -> pd.DataFrame:
        """markets: our market keys to pull (default: all for the league). Each costs ~1 credit."""
        chosen = markets or list(markets_for(league))
        markets = ",".join(MARKETS[m]["odds_key"] for m in chosen)
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


# ---------- game-line helpers ----------
def _best(recs):
    """Best (highest-paying) price and its book from a list of outcome records."""
    if not recs:
        return np.nan, None
    b = max(recs, key=lambda r: american_to_decimal(r["price"]))
    return float(b["price"]), b["book"]


def _moneyline(ml, home, away):
    df = pd.DataFrame(ml)
    out = {"home_ml": np.nan, "home_ml_book": None, "away_ml": np.nan, "away_ml_book": None, "home_win_prob": np.nan}
    if df.empty:
        return out
    out["home_ml"], out["home_ml_book"] = _best(df[df.name == home].to_dict("records"))
    out["away_ml"], out["away_ml_book"] = _best(df[df.name == away].to_dict("records"))
    w = df.pivot_table(index="book", columns="name", values="price", aggfunc="first").dropna()
    if home in w and away in w and len(w):
        ph, pa = 1 / american_to_decimal(w[home]), 1 / american_to_decimal(w[away])
        out["home_win_prob"] = float(np.mean(ph / (ph + pa)))  # vig removed, averaged across books
    return out


def _spread(sp, home, away):
    df = pd.DataFrame(sp)
    out = {"home_spread": np.nan, "spread_home_price": np.nan, "spread_home_book": None,
           "spread_away_price": np.nan, "spread_away_book": None}
    if df.empty:
        return out
    hp = df[df.name == home]
    if hp.empty:
        return out
    pt = float(hp["point"].mode().iloc[0])            # consensus home line, e.g. -2.5
    out["home_spread"] = -pt                          # + = home favored (model convention)
    out["spread_home_price"], out["spread_home_book"] = _best(hp[hp.point == pt].to_dict("records"))
    ap = df[(df.name == away) & (df.point == -pt)]
    out["spread_away_price"], out["spread_away_book"] = _best(ap.to_dict("records"))
    return out


def _total(tot):
    df = pd.DataFrame(tot)
    out = {"game_total": np.nan, "over_price": np.nan, "over_book": None, "under_price": np.nan, "under_book": None}
    if df.empty:
        return out
    pt = float(df[df.name == "Over"]["point"].mode().iloc[0])
    out["game_total"] = pt
    out["over_price"], out["over_book"] = _best(df[(df.name == "Over") & (df.point == pt)].to_dict("records"))
    out["under_price"], out["under_book"] = _best(df[(df.name == "Under") & (df.point == pt)].to_dict("records"))
    return out
