"""CLI.
  python train.py --league nfl                      # train + print validation report
  python train.py --league nfl --parlays            # demo slate (no keys)
  python train.py --league nfl cfb --parlays --live # real props (needs ODDS_API_KEY, CFBD_API_KEY)
"""
import argparse
import os
import pandas as pd

from config import MARKETS
from parlay import build_parlays
from pipeline import get_models, live_slate, load_history, seasons_back


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", nargs="+", default=["nfl"], choices=["nfl", "cfb"])
    ap.add_argument("--seasons", type=int, default=4)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--parlays", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--legs", type=int, default=3)
    ap.add_argument("--mode", default="balanced", choices=["safest", "balanced", "value"])
    ap.add_argument("--model-weight", type=float, default=0.5)
    ap.add_argument("--game", help='only one game, e.g. --game Saints (matches any part of the matchup)')
    ap.add_argument("--singles", type=int, default=8, help="how many top single bets to show")
    ap.add_argument("--log", action="store_true", help="save picks so grade.py can track results")
    a = ap.parse_args()

    all_legs = []
    for lg in a.league:
        hist = load_history(lg, seasons_back(a.seasons), os.getenv("CFBD_API_KEY"))
        models = get_models(lg, hist, retrain=a.retrain or not a.parlays)
        print(f"\n== {lg.upper()} walk-forward validation ==")
        print(pd.DataFrame({k: m.metrics for k, m in models.items()}).T.round(3).to_string())
        if not a.parlays:
            continue
        if a.live:
            all_legs.append(live_slate(lg, hist, models, os.environ["ODDS_API_KEY"], a.model_weight, game=a.game))
        else:
            from demo import mock_slate
            from slate import build_slate
            g, p = mock_slate(hist)
            all_legs.append(build_slate(hist, g, p, models, lg, a.model_weight))

    if a.parlays:
        legs = pd.concat(all_legs, ignore_index=True)
        if a.game:
            legs = legs[legs["game"].str.contains(a.game, case=False)]
            print(f"\nFiltered to games matching '{a.game}': {legs['game'].unique().tolist()}")

        # Single bets: the conservative play. Must be +EV, >=55% to hit, and not contradicted by the book.
        ok = legs[(legs.ev > 0) & (legs.p_final >= 0.55) & (legs.disagreement <= 0.25)]
        singles = ok.sort_values("p_final", ascending=False).head(a.singles)
        flagged = legs[(legs.ev > 0) & (legs.disagreement > 0.25)]
        print(f"\n== Top single bets ({len(singles)}) ==")
        for l in singles.itertuples():
            price = (l.decimal - 1) * 100 if l.decimal >= 2 else -100 / (l.decimal - 1)
            print(f"   {l.player:<24} {l.side} {l.line} {l.market_label.lower():<16} proj {l.projection:<6} "
                  f"hit {l.p_final:.0%}  (book says {l.p_market:.0%})  {price:+.0f} @ {l.book}")
        if len(flagged):
            print("\n== Skipped: model and book disagree a lot (check injury/role news) ==")
            for l in flagged.itertuples():
                print(f"   {l.player:<24} {l.side} {l.line} {l.market_label.lower():<16} proj {l.projection}")

        parlays = build_parlays(legs, a.n, a.legs, a.mode)
        if a.game and not parlays:
            print("\n(No parlays for a single game: legs must come from different games. "
                  "Same-game parlays are correlated and carry extra vig, so singles are the safer play.)")
        if a.log:
            from picklog import save_picks
            save_picks(singles, parlays)
        for i, pl in enumerate(parlays, 1):
            print(f"\nParlay {i}: {pl['american']:+d}  hit {pl['hit_prob']:.1%}  EV {pl['ev']:+.1%}  stake {pl['kelly_stake_pct']:.1f}% of bankroll")
            for l in pl["legs"]:
                print(f"   {l['player']:<24} {l['side']} {l['line']} {MARKETS[l['market']]['label'].lower():<16} "
                      f"proj {l['projection']:<6} p={l['p_final']:.0%} @ {l['book']}")


if __name__ == "__main__":
    main()
