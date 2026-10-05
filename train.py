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
            all_legs.append(live_slate(lg, hist, models, os.environ["ODDS_API_KEY"], a.model_weight))
        else:
            from demo import mock_slate
            from slate import build_slate
            g, p = mock_slate(hist)
            all_legs.append(build_slate(hist, g, p, models, lg, a.model_weight))

    if a.parlays:
        legs = pd.concat(all_legs, ignore_index=True)
        for i, pl in enumerate(build_parlays(legs, a.n, a.legs, a.mode), 1):
            print(f"\nParlay {i}: {pl['american']:+d}  hit {pl['hit_prob']:.1%}  EV {pl['ev']:+.1%}  stake {pl['kelly_stake_pct']:.1f}% of bankroll")
            for l in pl["legs"]:
                print(f"   {l['player']:<24} {l['side']} {l['line']} {MARKETS[l['market']]['label'].lower():<16} "
                      f"proj {l['projection']:<6} p={l['p_final']:.0%} @ {l['book']}")


if __name__ == "__main__":
    main()
