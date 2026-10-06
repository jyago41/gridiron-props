"""Glue used by both the CLI and the app: load history, train/load models, build a slate."""
import os
import pandas as pd

from config import current_mlb_season, current_season, markets_for
from features import build_features
from model import PropModel, train_all


def seasons_back(n=4):
    cur = current_season()
    return list(range(cur - n + 1, cur + 1))


def seasons_for(league: str) -> list[int]:
    if league == "mlb":           # baseball plays ~2,400 games a year, so two seasons is plenty
        y = current_mlb_season()
        return [y - 1, y]
    return seasons_back(4)


def load_history(league: str, seasons: list[int], cfbd_key: str | None = None, log=print) -> pd.DataFrame:
    if league == "mlb":
        from data_mlb import load_mlb
        return load_mlb(seasons, log=log)
    if league == "nfl":
        from data_nfl import load_nfl
        return load_nfl(seasons)
    from data_cfb import load_cfb
    if not cfbd_key:
        raise ValueError("College football needs a free CollegeFootballData API key (CFBD_API_KEY).")
    return load_cfb(seasons, cfbd_key)


def get_models(league: str, history: pd.DataFrame, retrain: bool = False) -> dict:
    mk = markets_for(league)
    paths = [PropModel(league, m).path() for m in mk]
    if not retrain and all(os.path.exists(p) for p in paths):
        return {m: PropModel.load(league, m) for m in mk}
    feat, feats = build_features(history)
    return train_all(feat, feats, league)


def live_slate(league, history, models, odds_key, model_weight, days_ahead=7, bookmakers=None, log=print, game=None):
    from odds import OddsClient, consolidate_props
    from slate import build_slate
    client = OddsClient(odds_key, bookmakers=bookmakers)
    games = client.game_lines(league, days_ahead)
    if game:  # only spend credits on the game(s) asked for
        games = games[games.home_team.str.contains(game, case=False) | games.away_team.str.contains(game, case=False)]
    log(f"{league.upper()}: {len(games)} games in the next {days_ahead} days")
    raw = []
    for eid in games["event_id"]:
        try:
            raw.append(client.player_props(league, eid))
        except Exception as e:  # one bad game shouldn't kill the whole slate
            log(f"skipped one game ({e})")
    raw = pd.concat([r for r in raw if not r.empty], ignore_index=True) if raw else pd.DataFrame()
    props = consolidate_props(raw)
    log(f"{league.upper()}: {len(props)} props pulled; {client.remaining} Odds API credits left")
    return build_slate(history, games, props, models, league, model_weight)


def live_games(league, odds_key, days_ahead=7, bookmakers=None):
    from odds import OddsClient
    return OddsClient(odds_key, bookmakers=bookmakers).game_lines(league, days_ahead)


def live_props(league, odds_key, event_ids, log=print, markets=None, bookmakers=None, alternates=True):
    """Fetch + consolidate props for just the chosen games (each game costs ~5 credits)."""
    from odds import OddsClient, consolidate_props
    client = OddsClient(odds_key, bookmakers=bookmakers)
    raw = []
    for eid in event_ids:
        try:
            raw.append(client.player_props(league, eid, markets, alternates))
        except Exception as e:
            log(f"Skipped one game ({e})")
    raw = [r for r in raw if not r.empty]
    props = consolidate_props(pd.concat(raw, ignore_index=True)) if raw else pd.DataFrame()
    return props, client.remaining
