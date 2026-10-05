"""Gridiron Props: XGBoost player-prop projections -> this weekend's best parlays.
Run:  streamlit run app.py
"""
import os
import pandas as pd
import streamlit as st

from config import MARKETS
from parlay import build_parlays
from pipeline import get_models, live_slate, load_history, seasons_back

st.set_page_config(page_title="Gridiron Props", page_icon="🏈", layout="wide")
st.markdown("""
<style>
  h1 { letter-spacing: -0.02em; }
  .parlay-head { display:flex; gap:1.5rem; align-items:baseline; flex-wrap:wrap;
                 border-left: 6px solid #C99700; padding: .4rem 0 .4rem .9rem; margin-top: 1.2rem; }
  .parlay-odds { font-size: 2rem; font-weight: 800; color: #0C2340; }
  .parlay-stat { color: #4A5B74; }
  .parlay-stat b { color: #0C2340; }
</style>""", unsafe_allow_html=True)

LEAGUES = {"NFL": "nfl", "College": "cfb"}

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Setup")
    source = st.radio("Odds source", ["Demo lines (no API keys)", "Live sportsbook lines"])
    live = source.startswith("Live")
    picked = st.multiselect("Leagues", list(LEAGUES), default=["NFL"],
                            help="Demo mode uses NFL only. College needs a CollegeFootballData key.")
    if not live:
        picked = ["NFL"]
    odds_key = st.text_input("The Odds API key", os.getenv("ODDS_API_KEY", ""), type="password") if live else ""
    cfbd_key = st.text_input("CollegeFootballData key", os.getenv("CFBD_API_KEY", ""), type="password") \
        if "College" in picked else ""

    st.header("Parlays")
    n_parlays = st.slider("How many parlays", 3, 8, 6)
    n_legs = st.slider("Legs per parlay", 2, 5, 3)
    mode = st.radio("Optimize for", ["safest", "balanced", "value"], index=1, horizontal=True,
                    help="Safest: highest chance to hit (still +EV). Value: highest expected return. Balanced: in between.")
    min_prob = st.slider("Minimum hit chance per leg", 0.50, 0.75, 0.55, 0.01)

    st.header("Model")
    model_weight = st.slider("Trust in model vs. market", 0.0, 1.0, 0.5, 0.05,
                             help="Blends the model's probability with the sportsbook's no-vig probability. "
                                  "Books are sharp, so 0.3 to 0.5 is sensible.")
    max_dis = st.slider("Skip legs where model and book disagree by more than", 0.10, 0.50, 0.25, 0.05,
                        help="Huge disagreements usually mean injury or role news the model can't see.")
    bankroll = st.number_input("Bankroll ($)", 0, 1_000_000, 500, 50)
    retrain = st.button("Retrain models")


@st.cache_data(ttl=6 * 3600, show_spinner="Loading play-by-play history…")
def cached_history(league, seasons, key):
    return load_history(league, list(seasons), key)


@st.cache_resource(show_spinner="Training XGBoost models…")
def cached_models(league, seasons, key, _nonce):
    return get_models(league, cached_history(league, seasons, key), retrain=_nonce > 0)


if retrain:
    st.session_state["nonce"] = st.session_state.get("nonce", 0) + 1

# ---------------- main ----------------
st.title("Gridiron Props")
st.write("XGBoost projections for every posted player prop, turned into this weekend's best parlays.")

if st.button("Build this weekend's parlays", type="primary"):
    all_legs, reports = [], {}
    for name in picked:
        lg = LEAGUES[name]
        try:
            seasons = tuple(seasons_back(4))
            hist = cached_history(lg, seasons, cfbd_key)
            models = cached_models(lg, seasons, cfbd_key, st.session_state.get("nonce", 0))
            reports[name] = pd.DataFrame({MARKETS[k]["label"]: m.metrics for k, m in models.items()}).T
            if live:
                if not odds_key:
                    st.error("Add a key from the-odds-api.com to pull live props, or switch to demo lines.")
                    st.stop()
                legs = live_slate(lg, hist, models, odds_key, model_weight, log=st.caption)
            else:
                from demo import mock_slate
                from slate import build_slate
                g, p = mock_slate(hist)
                legs = build_slate(hist, g, p, models, lg, model_weight)
            all_legs.append(legs)
        except Exception as e:
            st.error(f"{name}: {e}")
    legs = pd.concat([l for l in all_legs if not l.empty], ignore_index=True) if all_legs else pd.DataFrame()
    st.session_state.update(legs=legs, reports=reports)

legs = st.session_state.get("legs")
if legs is None:
    st.info("Pick your settings on the left, then build parlays. Demo lines work with no API keys.")
    st.stop()
if legs.empty:
    st.warning("No props matched players in the model. Check your API keys or try again closer to kickoff.")
    st.stop()

tab_p, tab_l, tab_m = st.tabs(["Parlays", "Every leg", "Model report"])

with tab_p:
    parlays = build_parlays(legs, n_parlays, n_legs, mode, min_prob, 0.0, max_dis)
    if not parlays:
        st.warning("Not enough +EV legs pass your filters. Lower the minimum hit chance or use fewer legs.")
    for i, pl in enumerate(parlays, 1):
        stake = bankroll * pl["kelly_stake_pct"] / 100
        st.markdown(
            f"<div class='parlay-head'><span class='parlay-odds'>{pl['american']:+d}</span>"
            f"<span class='parlay-stat'>Parlay {i}</span>"
            f"<span class='parlay-stat'>Hit chance <b>{pl['hit_prob']:.1%}</b></span>"
            f"<span class='parlay-stat'>Expected return <b>{pl['ev']:+.1%}</b></span>"
            f"<span class='parlay-stat'>Suggested stake <b>${stake:,.2f}</b></span></div>",
            unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{
            "Player": l["player"], "Team": l["team"], "Game": l["game"],
            "Bet": f"{l['side']} {l['line']} {l['market_label'].lower()}",
            "Projection": l["projection"], "Hit chance": f"{l['p_final']:.0%}",
            "Book says": f"{l['p_market']:.0%}", "Best price": f"{(l['decimal'] - 1) * 100:+.0f}" if l["decimal"] >= 2
            else f"{-100 / (l['decimal'] - 1):.0f}", "Book": l["book"]} for l in pl["legs"]]),
            hide_index=True, width="stretch")
    st.caption("Stakes are quarter-Kelly, capped at 2% of bankroll. Legs come from different games so "
               "their outcomes are roughly independent. Check injury news before betting.")

with tab_l:
    show = legs.assign(flag=legs["disagreement"] > max_dis)[
        ["league", "game", "player", "market_label", "side", "line", "projection",
         "p_model", "p_market", "p_final", "ev", "book", "flag"]]
    st.dataframe(show.rename(columns={"market_label": "market", "p_final": "hit chance", "flag": "check news"}),
                 hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ["p_model", "p_market", "hit chance", "ev"]})

with tab_m:
    st.write("Walk-forward validation: every prediction was made using only games played before it.")
    for name, rep in st.session_state.get("reports", {}).items():
        st.subheader(name)
        rep = rep.rename(columns={"mae_model": "Model error (MAE)", "mae_baseline_r5": "5-game avg error",
                                  "brier": "Brier score", "hit_rate_confident_picks": "Hit rate, confident picks*",
                                  "confident_pick_share": "Share of games with a confident pick"})
        st.dataframe(rep.round(3), width="stretch")
    st.caption("*Measured against a proxy line (the player's 5-game average), since free historical prop "
               "lines don't exist. Real sportsbook lines are sharper, so expect lower real-world hit rates.")
