"""Shared configuration: prop markets, stat columns, and sportsbook settings."""
from datetime import date

# Unified per-game stat columns (both NFL and CFB are normalized to these).
STAT_COLS = [
    "attempts", "completions", "passing_yards",
    "carries", "rushing_yards",
    "targets", "receptions", "receiving_yards",
]

# One XGBoost model is trained per (league, market).
# usage/min_usage define which players are "prop-worthy" based on PRIOR usage only (no leakage).
MARKETS = {
    "pass_yds":   {"stat": "passing_yards",   "positions": ["QB"],             "usage": "attempts", "min_usage": 15, "odds_key": "player_pass_yds",          "label": "Passing yards", "sports": ("nfl", "cfb")},
    "pass_cmp":   {"stat": "completions",     "positions": ["QB"],             "usage": "attempts", "min_usage": 15, "odds_key": "player_pass_completions",  "label": "Completions", "sports": ("nfl", "cfb")},
    "rush_yds":   {"stat": "rushing_yards",   "positions": ["QB", "RB", "WR"], "usage": "carries",  "min_usage": 3,  "odds_key": "player_rush_yds",          "label": "Rushing yards", "sports": ("nfl", "cfb")},
    "rec_yds":    {"stat": "receiving_yards", "positions": ["WR", "TE", "RB"], "usage": "targets",  "min_usage": 3,  "odds_key": "player_reception_yds",     "label": "Receiving yards", "sports": ("nfl", "cfb")},
    "receptions": {"stat": "receptions",      "positions": ["WR", "TE", "RB"], "usage": "targets",  "min_usage": 3,  "odds_key": "player_receptions",        "label": "Receptions", "sports": ("nfl", "cfb")},

    # ---- MLB: batters (role B) and starting pitchers (role SP); counts use a Poisson objective ----
    "mlb_hits":   {"stat": "hits",         "positions": ["B"],  "usage": "pa",   "min_usage": 3,  "odds_key": "batter_hits",            "label": "Hits",               "sports": ("mlb",), "role": "B",  "objective": "count:poisson"},
    "mlb_tb":     {"stat": "total_bases",  "positions": ["B"],  "usage": "pa",   "min_usage": 3,  "odds_key": "batter_total_bases",     "label": "Total bases",        "sports": ("mlb",), "role": "B",  "objective": "count:poisson"},
    "mlb_hrr":    {"stat": "hrr",          "positions": ["B"],  "usage": "pa",   "min_usage": 3,  "odds_key": "batter_hits_runs_rbis",  "label": "Hits + runs + RBIs", "sports": ("mlb",), "role": "B",  "objective": "count:poisson"},
    "mlb_k":      {"stat": "p_strikeouts", "positions": ["SP"], "usage": "p_bf", "min_usage": 15, "odds_key": "pitcher_strikeouts",     "label": "Pitcher strikeouts", "sports": ("mlb",), "role": "SP", "objective": "count:poisson"},
    "mlb_outs":   {"stat": "p_outs",       "positions": ["SP"], "usage": "p_bf", "min_usage": 15, "odds_key": "pitcher_outs",           "label": "Pitcher outs",       "sports": ("mlb",), "role": "SP"},
    "mlb_hits_a": {"stat": "p_hits",       "positions": ["SP"], "usage": "p_bf", "min_usage": 15, "odds_key": "pitcher_hits_allowed",   "label": "Hits allowed",       "sports": ("mlb",), "role": "SP", "objective": "count:poisson"},
}
ODDS_KEY_TO_MARKET = {v["odds_key"]: k for k, v in MARKETS.items()}

SPORT_KEYS = {"nfl": "americanfootball_nfl", "cfb": "americanfootball_ncaaf", "mlb": "baseball_mlb"}


def markets_for(league: str) -> dict:
    return {k: v for k, v in MARKETS.items() if league in v["sports"]}

XGB_PARAMS = dict(
    n_estimators=350, learning_rate=0.04, max_depth=4, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
    objective="reg:squarederror", tree_method="hist", n_jobs=2,  # free hosting has ~2 CPUs
)


def current_mlb_season() -> int:
    today = date.today()
    return today.year if today.month >= 3 else today.year - 1


def current_season() -> int:
    today = date.today()
    return today.year if today.month >= 8 else today.year - 1
