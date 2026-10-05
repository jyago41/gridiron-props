"""Demo slate: real NFL history + a mock sportsbook, so the whole app runs with no API keys.
Lines are set like a book would (recent average +/- noise, ~4.5% vig). Not real betting advice."""
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

from config import MARKETS
from slate import NFL_TEAMS

ABBR_TO_NAME = {v: k for k, v in NFL_TEAMS.items()}


def mock_slate(history: pd.DataFrame, n_games=10, seed=7):
    rng = np.random.default_rng(seed)
    last_season = history[history.season == history.season.max()]
    teams = [t for t in last_season.team.unique() if t in ABBR_TO_NAME]
    rng.shuffle(teams)
    kickoff = datetime.now(timezone.utc) + timedelta(days=2)

    games, props = [], []
    for g in range(min(n_games, len(teams) // 2)):
        home, away = teams[2 * g], teams[2 * g + 1]
        eid = f"demo{g}"
        games.append({"event_id": eid, "commence_time": kickoff + timedelta(hours=3 * (g % 3)),
                      "home_team": ABBR_TO_NAME[home], "away_team": ABBR_TO_NAME[away],
                      "home_spread": float(rng.choice(np.arange(-7, 7.5, 0.5))),
                      "game_total": float(rng.choice(np.arange(38, 52.5, 0.5)))})
        recent = last_season[last_season.team.isin([home, away])]
        recent = recent[recent.week >= recent.week.max() - 4]
        # books only post lines for players with an established role: must have played the team's
        # latest game AND at least 60% of its recent games
        last_game = recent.groupby("team").week.transform("max")
        active = set(recent.loc[recent.week == last_game, "player_name"])
        team_games = recent.groupby("team").week.nunique()
        played = recent.groupby(["team", "player_name"]).week.nunique().reset_index(name="n")
        played = played[played.n >= 0.6 * played.team.map(team_games)]
        recent = recent[recent.player_name.isin(active) & recent.player_name.isin(played.player_name)]
        for market, m in MARKETS.items():
            avg = recent[recent.pos_group.isin(m["positions"])].groupby("player_name")[m["stat"]].mean()
            top = avg.sort_values(ascending=False).head(2 if m["positions"] == ["QB"] else 4)
            for player, mu in top.items():
                if mu < 1.5:
                    continue
                line = np.floor(mu * rng.normal(1.0, 0.08)) + 0.5
                skew = rng.normal(0, 0.03)
                p_over = np.clip(0.5 + skew, 0.4, 0.6)
                vig = 1.045
                over_dec, under_dec = 1 / (p_over * vig), 1 / ((1 - p_over) * vig)
                props.append({"event_id": eid, "player": player, "market": market, "line": line,
                              "p_market_over": p_over, "n_books": 1,
                              "over_dec": round(over_dec * rng.uniform(1.0, 1.02), 3), "over_book": "DemoBook",
                              "under_dec": round(under_dec * rng.uniform(1.0, 1.02), 3), "under_book": "DemoBook"})
    return pd.DataFrame(games), pd.DataFrame(props)
