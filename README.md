# Gridiron Props

XGBoost projections for NFL, college football and MLB player props, turned into the best 5–6 parlays for the upcoming weekend.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py          # demo lines work with zero API keys
```

For real sportsbook lines, get free keys and export them (or paste them in the sidebar):

```bash
export ODDS_API_KEY=...   # the-odds-api.com   (free tier: 500 credits/month)
export CFBD_API_KEY=...   # collegefootballdata.com (free; only needed for college)
```

CLI equivalent: `python train.py --league nfl cfb --parlays --live --mode safest`

## How it works

| File | Job |
|---|---|
| `data_nfl.py` | Player-game box scores + closing spreads/totals from nflverse (no key needed) |
| `data_cfb.py` | Same for college via CollegeFootballData; caches past seasons locally |
| `features.py` | Rolling form (3/5 games, EWMA, season-to-date, volatility), usage share, team pace, opponent defense vs. position, Vegas implied team total. Every feature uses only prior games. |
| `model.py` | One XGBoost regressor per market (pass yds, completions, rush yds, rec yds, receptions). Walk-forward out-of-fold residuals, bucketed by projection size, turn a point projection into P(over line). |
| `odds.py` | Pulls lines from every book, finds the consensus line, strips the vig, and line-shops the best price. |
| `slate.py` | Matches props to players, projects each, blends model probability with the market's, computes EV. |
| `parlay.py` | Beam search for the best non-overlapping parlays (one leg per game, each leg +EV on its own), with quarter-Kelly stake sizing. |
| `data_mlb.py` | MLB box scores from the official MLB Stats API (free, no key), cached per season |
| `features_mlb.py` | Batter form and per-PA rates, lineup spot, opposing starter, team offense/defense, ballpark; pitcher form vs. opposing lineup |
| `slate_mlb.py` | Matches MLB props to players, pulls probable pitchers for each game |
| `sgp.py` | Same-game parlays: learns how stats move together, simulates the game, prices fair odds |
| `app.py` | Streamlit front end. |

## Things to know before you bet a dollar

- **Books are sharp on props.** The "trust in model" slider blends your model with the book's no-vig probability. At 1.0 you'll see huge fake edges. Keep it at 0.3–0.5 until you've proven the model on real closing lines.
- **Big disagreements are usually missing news**, not free money (a backup QB, a WR returning from injury). Those legs are flagged and kept out of parlays by default.
- **The validation hit rate is against a proxy line** (5-game average), because free historical prop lines don't exist. Real hit rates will be lower. The honest test: log every pick with the line you got and the closing line. If you consistently beat the close (CLV), the model has an edge.
- **Parlays multiply the vig.** A 3-leg parlay of -110 legs needs each leg to hit ~57%+ just to break even. That's why every leg must be +EV on its own.
- Demo lines are synthetic and deliberately noisy, so demo EVs are inflated.

## Good next upgrades

1. Injury/inactive filter: `nflreadpy.load_injuries()` and `load_snap_counts()` as features.
2. Weather and roof (already in the nflverse schedule: `temp`, `wind`, `roof`).
3. A pick log + CLV tracker so you can measure real performance week over week.
4. Correlation modeling for same-game parlays (QB yards and his WR1's yards move together).
5. Anytime-TD markets with a classifier (`XGBClassifier`, `binary:logistic`).
