"""Glue used by both the CLI and the app: load history, train/load models, build a slate."""
import os
import pandas as pd

from config import MARKETS, current_season
from features import build_features
from model import PropModel, train_all


def seasons_back(n=4):
    cur = current_season()
    return list(range(cur - n + 1, cur + 1))


def load_history(league: str, seasons: list[int], cfbd_key: str | None = None) -> pd.DataFrame:
    if league == "nfl":
        from data_nfl import load_nfl
        return load_nfl(seasons)
    from data_cfb import load_cfb
    if not cfbd_key:
        raise ValueError("College football needs a free CollegeFootballData API key (CFBD_API_KEY).")
    return load_cfb(seasons, cfbd_key)


def get_models(league: str, history: pd.DataFrame, retrain: bool = False) -> dict:
    paths = [PropModel(league, m).path() for m in MARKETS]
    if not retrain and all(os.path.exists(p) for p in paths):
        return {m: PropModel.load(league, m) for m in MARKETS}
    feat, feats = build_features(history)
    return train_all(feat, feats, league)


def live_slate(league, history, models, odds_key, model_weight, days_ahead=7, bookmakers=None, log=print):
    from odds import OddsClient, consolidate_props
    from slate import build_slate
    client = OddsClient(odds_key, bookmakers=bookmakers)
    games = client.game_lines(league, days_ahead)
    log(f"{league.upper()}: {len(games)} games in the next {days_ahead} days")
    raw = [client.player_props(league, eid) for eid in games["event_id"]]
    raw = pd.concat([r for r in raw if not r.empty], ignore_index=True) if raw else pd.DataFrame()
    props = consolidate_props(raw)
    log(f"{league.upper()}: {len(props)} props pulled; {client.remaining} Odds API credits left")
    return build_slate(history, games, props, models, league, model_weight)
