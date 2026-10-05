"""Gridiron Props: pick your games, get the model's safest bets.
Run locally:  streamlit run app.py      Deploy: see DEPLOY.md
"""
import os
import pandas as pd
import streamlit as st

from config import MARKETS
from parlay import build_parlays
from picklog import LOG_PATH, grade_log, load_log, save_picks
from pipeline import get_models, live_games, live_props, load_history, seasons_back
from slate import build_slate

st.set_page_config(page_title="Gridiron Props", page_icon="🏈", layout="wide")
st.markdown("""
<style>
  h1 { letter-spacing: -0.02em; margin-bottom: 0; }
  .step { font-weight: 700; color: #0C2340; font-size: 1.15rem; margin: 1.4rem 0 .3rem;
          border-left: 6px solid #C99700; padding-left: .7rem; }
  .parlay-head { display:flex; gap:1.25rem; align-items:baseline; flex-wrap:wrap; margin-top: 1.1rem; }
  .parlay-odds { font-size: 1.9rem; font-weight: 800; color: #0C2340; }
  .parlay-stat { color: #4A5B74; }
  .parlay-stat b { color: #0C2340; }
</style>""", unsafe_allow_html=True)


def secret(name):
    """Keys come from Streamlit Cloud secrets or environment variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return os.getenv(name, "").strip()


def american(dec):
    return f"{(dec - 1) * 100:+.0f}" if dec >= 2 else f"{-100 / (dec - 1):.0f}"


def short(team, league):
    """'New Orleans Saints' -> 'Saints' for the NFL; college names stay whole."""
    return team.split()[-1] if league == "nfl" else team


def money(price, book=None):
    if pd.isna(price):
        return "–"
    p = f"{int(price):+d}" if price > 0 else f"{int(price)}"
    return f"{p} ({book})" if book else p


def kickoff(t):
    return pd.Timestamp(t).tz_convert("America/New_York").strftime("%a %-I:%M %p ET")


LEAGUES = {"NFL": "nfl", "College": "cfb"}

# ---------------- settings (sidebar, collapsed on phones) ----------------
with st.sidebar:
    st.header("Settings")
    live = st.toggle("Use real sportsbook lines", value=bool(secret("ODDS_API_KEY")),
                     help="Off = practice lines, no API key needed.")
    leagues = st.multiselect("Leagues", list(LEAGUES), default=["NFL"]) if live else ["NFL"]
    odds_key = secret("ODDS_API_KEY")
    cfbd_key = secret("CFBD_API_KEY")
    if live and not odds_key:
        odds_key = st.text_input("The Odds API key", type="password")
    if "College" in leagues and not cfbd_key:
        cfbd_key = st.text_input("CollegeFootballData key", type="password")

    st.subheader("How careful to be")
    min_prob = st.slider("Minimum hit chance per bet", 0.50, 0.75, 0.55, 0.01)
    model_weight = st.slider("Trust in model vs. sportsbook", 0.0, 1.0, 0.3, 0.05,
                             help="Lower = lean on the sportsbook more. 0.3 is a sensible, cautious default.")
    max_dis = st.slider("Flag bets where model and book disagree by more than", 0.10, 0.50, 0.25, 0.05)

    st.subheader("Parlays")
    n_parlays = st.slider("How many parlays", 1, 8, 6)
    n_legs = st.slider("Legs per parlay", 2, 5, 3)
    mode = st.radio("Optimize for", ["safest", "balanced", "value"], index=0, horizontal=True)
    bankroll = st.number_input("Bankroll ($)", 0, 1_000_000, 500, 50)


@st.cache_resource(ttl=6 * 3600, max_entries=2, show_spinner="Loading player stats…")
def cached_history(league, seasons, key):
    return load_history(league, list(seasons), key)


@st.cache_resource(max_entries=2, show_spinner="Training the models (first time takes a minute or two)…")
def cached_models(league, seasons, key):
    return get_models(league, cached_history(league, seasons, key))


SEASONS = tuple(seasons_back(4))

st.title("Gridiron Props")
st.caption("Practice lines" if not live else "Live sportsbook lines")

# ---------------- step 1: games ----------------
st.markdown("<div class='step'>1. Find this week's games</div>", unsafe_allow_html=True)
if st.button("Find this week's games", width="stretch"):
    frames, demo_props = [], []
    with st.spinner("Looking up games…"):
        for name in leagues:
            lg = LEAGUES[name]
            try:
                if live:
                    if not odds_key:
                        st.error("Add your Odds API key in Settings, or turn off real lines.")
                        st.stop()
                    g = live_games(lg, odds_key)
                else:
                    from demo import mock_slate
                    g, p = mock_slate(cached_history(lg, SEASONS, cfbd_key))
                    demo_props.append(p)
                frames.append(g.assign(league=lg))
            except Exception as e:
                st.error(f"{name}: couldn't load games ({e})")
    games = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not games.empty:
        games["label"] = (games["away_team"] + " @ " + games["home_team"] + ", "
                          + games["commence_time"].map(kickoff))
        games = games.sort_values("commence_time")
    st.session_state.update(games=games, legs=None,
                            demo_props=pd.concat(demo_props) if demo_props else None)

games = st.session_state.get("games")
if games is None:
    st.info("Tap the button to load the games for the next 7 days.")
    st.stop()
if games.empty:
    st.warning("No games found in the next 7 days.")
    st.stop()

# ---------------- step 2: choose games ----------------
st.markdown("<div class='step'>2. Choose games</div>", unsafe_allow_html=True)
teams = sorted(set(games["home_team"]) | set(games["away_team"]))
team = st.selectbox("Team", ["All teams"] + teams)
pool = games if team == "All teams" else games[(games.home_team == team) | (games.away_team == team)]
labels = pool["label"].tolist()
if team == "All teams":
    quick = st.radio("Show", ["Next game day", "All games", "Pick my own"], horizontal=True)
    if quick == "Next game day":
        first_day = pd.Timestamp(pool["commence_time"].iloc[0]).tz_convert("America/New_York").date()
        default = [l for l, t in zip(labels, pool["commence_time"])
                   if pd.Timestamp(t).tz_convert("America/New_York").date() == first_day]
    else:
        default = labels if quick == "All games" else []
else:
    default = labels
chosen = st.multiselect("Games", labels, default=default, label_visibility="collapsed",
                        key=f"games_{team}")

# ---------------- game odds (already paid for when games were loaded) ----------------
if chosen:
    with st.expander(f"Game odds ({len(chosen)} game{'s' if len(chosen) > 1 else ''})", expanded=True):
        sel = games[games["label"].isin(chosen)]
        rows = []
        for g in sel.itertuples():
            h, aw = short(g.home_team, g.league), short(g.away_team, g.league)
            hs = getattr(g, "home_spread", float("nan"))
            spread = "–" if pd.isna(hs) else (f"{h} {-hs:+g}" if hs != 0 else "Pick'em")
            rows.append({
                "Game": f"{aw} @ {h}", "Kickoff": kickoff(g.commence_time),
                f"Moneyline (away)": money(getattr(g, "away_ml", float("nan")), getattr(g, "away_ml_book", None)),
                f"Moneyline (home)": money(getattr(g, "home_ml", float("nan")), getattr(g, "home_ml_book", None)),
                "Home win chance": "–" if pd.isna(getattr(g, "home_win_prob", float("nan")))
                                   else f"{g.home_win_prob:.0%}",
                "Spread": spread,
                "Spread price (home / away)": f"{money(getattr(g, 'spread_home_price', float('nan')))} / "
                                              f"{money(getattr(g, 'spread_away_price', float('nan')))}",
                "Total": "–" if pd.isna(g.game_total) else f"{g.game_total:g}",
                "Over / Under price": f"{money(getattr(g, 'over_price', float('nan')))} / "
                                      f"{money(getattr(g, 'under_price', float('nan')))}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Best price across sportsbooks, with the book in parentheses. Win chance is the "
                   "sportsbooks' own estimate with their cut removed. These are market odds, not model picks.")
if live and chosen:
    st.caption(f"Getting picks uses about {5 * len(chosen)} Odds API credits.")

st.markdown("<div class='step'>3. Get picks</div>", unsafe_allow_html=True)
if st.button("Get picks", type="primary", width="stretch", disabled=not chosen):
    sel = games[games["label"].isin(chosen)]
    all_legs = []
    for lg, g in sel.groupby("league"):
        with st.spinner(f"Running the {lg.upper()} models…"):
            try:
                hist = cached_history(lg, SEASONS, cfbd_key)
                models = cached_models(lg, SEASONS, cfbd_key)
                if live:
                    props, remaining = live_props(lg, odds_key, g["event_id"].tolist(), log=st.caption)
                    st.session_state["credits"] = remaining
                else:
                    dp = st.session_state["demo_props"]
                    props = dp[dp.event_id.isin(g["event_id"])]
                all_legs.append(build_slate(hist, g, props, models, lg, model_weight))
            except Exception as e:
                st.error(f"{lg.upper()}: {e}")
    legs = pd.concat([l for l in all_legs if not l.empty], ignore_index=True) if all_legs else pd.DataFrame()
    st.session_state.update(legs=legs, n_games=len(sel))

legs = st.session_state.get("legs")
if legs is None:
    st.stop()
if legs.empty:
    st.warning("No player props are posted for these games yet. Books usually post most props "
               "Wednesday to Saturday. Try again later.")
    st.stop()
if st.session_state.get("credits"):
    st.caption(f"{st.session_state['credits']} Odds API credits left this month.")

ok = legs[(legs.ev > 0) & (legs.p_final >= min_prob) & (legs.disagreement <= max_dis)]
singles = ok.sort_values("p_final", ascending=False).head(10)
flagged = legs[(legs.ev > 0) & (legs.disagreement > max_dis)]
parlays = build_parlays(legs, n_parlays, n_legs, mode, min_prob, 0.0, max_dis) \
    if st.session_state.get("n_games", 0) >= n_legs else []


def leg_table(rows):
    return pd.DataFrame([{
        "Player": r["player"], "Bet": f"{r['side']} {r['line']} {r['market_label'].lower()}",
        "Projection": r["projection"], "Hit chance": f"{r['p_final']:.0%}",
        "Book says": f"{r['p_market']:.0%}", "Price": american(r["decimal"]), "Book": r["book"],
        "Game": r["game"]} for r in rows])


tab_s, tab_p, tab_n, tab_r = st.tabs(["Safest bets", "Parlays", f"Check the news ({len(flagged)})", "My record"])

with tab_s:
    st.write("Single bets the model likes that the sportsbook doesn't strongly disagree with. "
             "These are the most conservative plays.")
    if singles.empty:
        st.info("Nothing passes your filters. Try lowering the minimum hit chance in Settings.")
    else:
        st.dataframe(leg_table(singles.to_dict("records")), hide_index=True, width="stretch")

with tab_p:
    if st.session_state.get("n_games", 0) < n_legs:
        st.info(f"Parlays need {n_legs} different games (one bet per game). Choose more games, or use "
                "single bets. Same-game parlays are riskier: the bets rise and fall together, and books "
                "charge extra on them.")
    elif not parlays:
        st.info("Not enough good bets for parlays. Lower the minimum hit chance or use fewer legs.")
    for i, pl in enumerate(parlays, 1):
        st.markdown(
            f"<div class='parlay-head'><span class='parlay-odds'>{pl['american']:+d}</span>"
            f"<span class='parlay-stat'>Parlay {i}</span>"
            f"<span class='parlay-stat'>Hit chance <b>{pl['hit_prob']:.1%}</b></span>"
            f"<span class='parlay-stat'>Suggested stake <b>${bankroll * pl['kelly_stake_pct'] / 100:,.2f}</b></span>"
            f"</div>", unsafe_allow_html=True)
        st.dataframe(leg_table(pl["legs"]), hide_index=True, width="stretch")

with tab_n:
    st.write("The model and the sportsbook disagree a lot on these. That usually means news the model "
             "can't see: an injury, a benching, a role change. Look each one up before considering it.")
    if not flagged.empty:
        st.dataframe(leg_table(flagged.sort_values("disagreement", ascending=False).to_dict("records"))
                     .drop(columns=["Hit chance"]), hide_index=True, width="stretch")

if st.button("Save these picks to my record", width="stretch"):
    n = save_picks(singles, parlays)
    st.success(f"Saved {n} picks. Check the My record tab after the games.")

with tab_r:
    st.write("Results fill in the morning after each game.")
    if st.button("Update results"):
        with st.spinner("Checking box scores…"):
            log, s = grade_log()
        st.session_state["record"] = (log, s)
    log, s = st.session_state.get("record", (load_log(), None))
    if s:
        c1, c2, c3 = st.columns(3)
        w, l = s["singles_w"], s["singles_l"]
        c1.metric("Single bets", f"{w}-{l}", f"{w / (w + l):.0%} hit rate" if w + l else None)
        c2.metric("Profit (1 unit per bet)", f"{s['singles_profit'] + s['parlays_profit']:+.2f}u")
        c3.metric("Waiting on results", s["pending"])
        st.caption("At typical -110 prices you need to win 52.4% of single bets to break even.")
    if not log.empty:
        st.dataframe(log[["logged_at", "kind", "player", "market", "side", "line", "actual", "result", "book"]]
                     .iloc[::-1], hide_index=True, width="stretch")
        st.download_button("Download my record (backup)", log.to_csv(index=False), "picks.csv", "text/csv")
    up = st.file_uploader("Restore a backup", type="csv")
    if up is not None and st.button("Restore"):
        pd.read_csv(up).to_csv(LOG_PATH, index=False)
        st.success("Restored. Tap Update results.")
