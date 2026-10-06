"""Live game lines + player props from The Odds API (the-odds-api.com).

Cost note: each /events/{id}/odds call costs (number of markets x regions) credits, so a full
NFL Sunday with 5 prop markets is ~16 games x 5 = ~80 credits. The free tier is 500/month.
"""
from datetime import datetime, timedelta, timezone
import time
import numpy as np
import pandas as pd
import requests

from config import ALT_KEYS, MARKETS, ODDS_KEY_TO_MARKET, SPORT_KEYS, markets_for

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

    def player_props(self, league: str, event_id: str, markets: list[str] | None = None,
                     alternates: bool = True) -> pd.DataFrame:
        """markets: our market keys to pull (default: all for the league). Each costs ~1 credit;
        alternates (milestone X+ lines) cost 1 more per market."""
        chosen = markets or list(markets_for(league))
        keys = [MARKETS[m]["odds_key"] for m in chosen] + ([MARKETS[m]["alt_key"] for m in chosen] if alternates else [])
        markets = ",".join(keys)
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
                                 "side": o["name"], "point": float(o["point"]), "price": float(o["price"]),
                                 "alt": mk["key"] in ALT_KEYS})
        return pd.DataFrame(rows)


DEFAULT_ALT_HOLD = 0.08   # typical sportsbook margin on one-sided milestone lines


def consolidate_props(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (event, player, market, line), for EVERY line the books post (main + milestones):
    the book's fair (no-vig) chance of the Over, and the best price on each side at that line.
    Milestone lines are often Over-only; their fair chance is estimated by removing a typical margin."""
    if raw.empty:
        return raw
    raw = raw.assign(dec=american_to_decimal(raw["price"]))
    if "alt" not in raw:
        raw["alt"] = False
    key = ["event_id", "player", "market"]
    line = key + ["point"]

    # two-way prices at the same line & book -> exact no-vig probability + that book's margin
    wide = raw.pivot_table(index=line + ["book"], columns="side", values="dec", aggfunc="max").reset_index()
    for s in ("Over", "Under"):
        if s not in wide:
            wide[s] = np.nan
    two = wide.dropna(subset=["Over", "Under"]).copy()
    two["p_nv"] = (1 / two["Over"]) / (1 / two["Over"] + 1 / two["Under"])
    two["hold"] = 1 / two["Over"] + 1 / two["Under"] - 1
    p_two = two.groupby(line, as_index=False)["p_nv"].mean()
    hold = two.groupby(key, as_index=False)["hold"].median().rename(columns={"hold": "hold_pm"})

    out = raw.groupby(line, as_index=False).agg(n_books=("book", "nunique"), alt=("alt", "min"))
    best = lambda s: (raw[raw.side == s].sort_values("dec", ascending=False).drop_duplicates(line)
                      [line + ["dec", "book"]].rename(columns={"dec": f"{s.lower()}_dec", "book": f"{s.lower()}_book"}))
    out = out.merge(best("Over"), on=line, how="left").merge(best("Under"), on=line, how="left")
    mean_imp = lambda s: (raw[raw.side == s].assign(imp=lambda d: 1 / d.dec).groupby(line, as_index=False)["imp"]
                          .mean().rename(columns={"imp": f"imp_{s.lower()}"}))
    out = (out.merge(p_two, on=line, how="left").merge(hold, on=key, how="left")
              .merge(mean_imp("Over"), on=line, how="left").merge(mean_imp("Under"), on=line, how="left"))
    h = np.maximum(out["hold_pm"].fillna(DEFAULT_ALT_HOLD), 0.04) + np.where(out["alt"], 0.02, 0.0)
    est = np.where(out["imp_over"].notna(), out["imp_over"] / (1 + h), 1 - out["imp_under"] / (1 + h))
    out["p_market_over"] = out["p_nv"].fillna(pd.Series(est, index=out.index)).clip(0.01, 0.99)
    return out.drop(columns=["p_nv", "hold_pm", "imp_over", "imp_under"]).rename(columns={"point": "line"})


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
