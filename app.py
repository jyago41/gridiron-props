"""Gridiron Props: XGBoost player-prop picks for football and MLB.
Run locally:  streamlit run app.py      Deploy: see DEPLOY.md
"""
import os
import pandas as pd
import streamlit as st

from config import BOOKS, MARKETS, markets_for
from parlay import build_parlays
from picklog import LOG_PATH, grade_log, load_log, save_picks
from pipeline import get_models, live_games, live_props, load_history, seasons_for
from sgp import american_to_dec, build_sgps, correlation_table
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
  .matchup { font-weight: 700; color: #0C2340; margin-top: .8rem; }
</style>""", unsafe_allow_html=True)


# ====================== helpers ======================
def secret(name):
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return os.getenv(name, "").strip()


def american(dec):
    return f"{(dec - 1) * 100:+.0f}" if dec >= 2 else f"{-100 / (dec - 1):.0f}"


def money(price, book=None):
    if price is None or pd.isna(price):
        return "–"
    p = f"{int(price):+d}" if price > 0 else f"{int(price)}"
    return f"{p} ({book})" if book else p


def kickoff(t):
    return pd.Timestamp(t).tz_convert("America/New_York").strftime("%a %-I:%M %p ET")


def short(team, league):
    if league == "mlb":
        w = team.split()
        return " ".join(w[-2:]) if w[-1] in ("Sox", "Jays") else w[-1]
    return team.split()[-1] if league == "nfl" else team


def val(row, col):
    v = getattr(row, col, None)
    return None if v is None or (not isinstance(v, str) and pd.isna(v)) else v


FOOTBALL_MARKETS = 5
ODDS_KEY, CFBD_KEY = secret("ODDS_API_KEY"), secret("CFBD_API_KEY")


@st.cache_resource(ttl=6 * 3600, max_entries=3,
                   show_spinner="Loading player stats (baseball's first load downloads ~5,000 box scores, a few minutes)…")
def cached_history(league, key):
    return load_history(league, seasons_for(league), key)


@st.cache_resource(max_entries=3, show_spinner="Training the models (first time takes a minute or two)…")
def cached_models(league, key):
    return get_models(league, cached_history(league, key))


@st.cache_resource(max_entries=3, show_spinner="Learning how player stats move together (one time)…")
def cached_corr(league, key):
    from features import build_features
    feat, _ = build_features(cached_history(league, key))
    return correlation_table(feat)


@st.cache_data(ttl=600, show_spinner=False)
def mlb_schedule_info(start, end):
    """Probable starters + series status from MLB (free, no credits)."""
    from data_mlb import schedule
    return schedule(start, end, probables=True)


# ====================== sidebar: shared settings ======================
with st.sidebar:
    st.header("Settings")
    if not ODDS_KEY:
        ODDS_KEY = st.text_input("The Odds API key", type="password")
    fb_live = st.toggle("Football: use real sportsbook lines", value=bool(ODDS_KEY),
                        help="Off = practice NFL lines, no API key needed. MLB always uses real lines.")
    st.subheader("Your sportsbook")
    my_books = st.multiselect("Only show bets I can place at", list(BOOKS), default=["DraftKings"],
                              help="Lines and prices come only from these books, so every pick matches your app.")
    book_keys = [BOOKS[b] for b in my_books] or None
    use_alts = st.toggle("Include milestone bets (2+, 3+, 75+ yards…)", value=True,
                         help="Pulls the X+ ladder your sportsbook offers. Costs about twice the credits.")
    skip_one_plus = st.toggle("Skip 1+ bets (Over 0.5)", value=True,
                              help="Many books don't offer 1+ on hits, total bases, and similar props.")
    st.subheader("How careful to be")
    min_prob = st.slider("Minimum hit chance per bet", 0.20, 0.75, 0.55, 0.01,
                         help="Lower it (and set Optimize for: value) to see milestone bets like 2+ hits, which hit less often but pay more.")
    model_weight = st.slider("Trust in model vs. sportsbook", 0.0, 1.0, 0.3, 0.05,
                             help="Lower = lean on the sportsbook more. 0.3 is a sensible, cautious default.")
    max_dis = st.slider("Flag bets where model and book disagree by more than", 0.10, 0.50, 0.25, 0.05)
    st.subheader("Parlays")
    n_parlays = st.slider("How many parlays", 1, 8, 6)
    n_legs = st.slider("Legs per parlay", 2, 5, 3)
    mode = st.radio("Optimize for", ["safest", "balanced", "value"], index=0, horizontal=True)
    bankroll = st.number_input("Bankroll ($)", 0, 1_000_000, 500, 50)

st.title("Gridiron Props")


# ====================== shared result sections ======================
def bet_text(side, line, label):
    """'Over 1.5 hits' -> '2+ hits', the way sportsbooks show milestones."""
    label = label.lower()
    if side == "Over" and abs(line % 1 - 0.5) < 1e-9:
        return f"{int(line + 0.5)}+ {label}"
    return f"{side} {line:g} {label}"


def best_lines(df):
    """One bet per player and stat: the line that best fits your 'Optimize for' choice."""
    if df.empty:
        return df
    by = "p_final" if mode == "safest" else "ev"
    return df.sort_values(by, ascending=False).drop_duplicates(["event_id", "player", "market"])


def leg_table(rows, mlb=False):
    out = []
    for r in rows:
        d = {"Player": r["player"], "Bet": bet_text(r["side"], r["line"], r["market_label"]),
             "Projection": r["projection"], "Hit chance": f"{r['p_final']:.0%}",
             "Book says": f"{r['p_market']:.0%}", "Price": american(r["decimal"]), "Book": r["book"]}
        if mlb:
            d["Lineup"] = r.get("lineup", "")
        d["Game"] = r["game"]
        out.append(d)
    return pd.DataFrame(out)


def parlays_section(ns, legs, n_games, sports):
    mlb = ns == "mlb"
    ptype = st.radio("Parlay type", ["Same game", "Different games"], horizontal=True,
                     index=0 if n_games < n_legs else 1, key=f"{ns}_ptype")
    if ptype == "Different games":
        parlays = build_parlays(legs, n_parlays, n_legs, mode, min_prob, 0.0, max_dis) if n_games >= n_legs else []
        if n_games < n_legs:
            st.info(f"Choose at least {n_legs} games for this type (one bet per game), or switch to Same game.")
        elif not parlays:
            st.info("Not enough good bets for parlays. Lower the minimum hit chance or use fewer legs.")
        for i, pl in enumerate(parlays, 1):
            st.markdown(
                f"<div class='parlay-head'><span class='parlay-odds'>{pl['american']:+d}</span>"
                f"<span class='parlay-stat'>Parlay {i}</span>"
                f"<span class='parlay-stat'>Hit chance <b>{pl['hit_prob']:.1%}</b></span>"
                f"<span class='parlay-stat'>Suggested stake <b>${bankroll * pl['kelly_stake_pct'] / 100:,.2f}</b></span>"
                f"</div>", unsafe_allow_html=True)
            st.dataframe(leg_table(pl["legs"], mlb), hide_index=True, width="stretch")
        return parlays

    st.write("Stats in the same game move together, so these chances account for that. Your sportsbook "
             "sets its own payout for same-game parlays: build it in your sportsbook app, then type its odds below.")
    with st.spinner("Simulating each game 20,000 times…"):
        corr = {}
        for lg in sports:
            corr.update(cached_corr(lg, CFBD_KEY))
        sgps = build_sgps(legs, corr, n_parlays, n_legs, min_prob, max_dis)
    if not sgps:
        st.info("Not enough good bets in these games for a same-game parlay. Lower the minimum hit chance, "
                "use fewer legs, or try again closer to game time when more props are posted.")
    for i, sg in enumerate(sgps, 1):
        st.markdown(
            f"<div class='parlay-head'><span class='parlay-odds'>{sg['fair_american']:+d}</span>"
            f"<span class='parlay-stat'>Fair odds, same-game parlay {i}</span>"
            f"<span class='parlay-stat'>Chance all hit <b>{sg['hit_prob']:.1%}</b></span>"
            f"<span class='parlay-stat'>If the legs were unrelated <b>{sg['indep_prob']:.1%}</b></span>"
            f"</div>", unsafe_allow_html=True)
        st.caption(sg["game"])
        st.dataframe(leg_table(sg["legs"], mlb).drop(columns=["Game"]), hide_index=True, width="stretch")
        book = st.number_input("Your sportsbook's odds for this parlay (e.g. 250 for +250)",
                               value=None, step=5, format="%d", key=f"{ns}_sgp_{i}")
        if book is not None:
            if -100 < book < 100:
                st.error("American odds are +100 or higher, or -100 or lower.")
            else:
                dec = american_to_dec(book)
                ev = sg["hit_prob"] * dec - 1
                if ev > 0:
                    stake = bankroll * min(max(ev / (dec - 1) * 0.25, 0), 0.02)
                    st.success(f"Worth it: {int(book):+d} beats the fair price of {sg['fair_american']:+d}. "
                               f"Expected return {ev:+.1%}. Suggested stake ${stake:,.2f}.")
                else:
                    st.warning(f"Skip it: {int(book):+d} pays less than the fair price of "
                               f"{sg['fair_american']:+d}. Expected return {ev:+.1%}.")
    return []


def record_section(ns, league_names):
    st.write("Results fill in after each game (football: the next morning; MLB: once the game is final).")
    if st.button("Update results", key=f"{ns}_grade"):
        with st.spinner("Checking box scores…"):
            st.session_state["record"] = grade_log()
    log, s = st.session_state.get("record", (load_log(), None))
    log = log[log["league"].isin(league_names)] if not log.empty else log
    if not log.empty:
        done = log[log.result.isin(["win", "loss"]) & (log.kind == "single")]
        w, l = (done.result == "win").sum(), (done.result == "loss").sum()
        profit = ((done.result == "win") * (done.decimal - 1) - (done.result == "loss")).sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Single bets", f"{w}-{l}", f"{w / (w + l):.0%} hit rate" if w + l else None)
        c2.metric("Profit (1 unit per bet)", f"{profit:+.2f}u")
        c3.metric("Waiting on results", int(log.result.isna().sum()))
        st.caption("At typical -110 prices you need to win 52.4% of single bets to break even.")
        st.dataframe(log[["logged_at", "kind", "player", "market", "side", "line", "actual", "result", "book"]]
                     .iloc[::-1], hide_index=True, width="stretch")
    full = load_log()
    if not full.empty:
        st.download_button("Download my record (backup)", full.to_csv(index=False), "picks.csv", "text/csv",
                           key=f"{ns}_dl")
    up = st.file_uploader("Restore a backup", type="csv", key=f"{ns}_up")
    if up is not None and st.button("Restore", key=f"{ns}_restore"):
        pd.read_csv(up).to_csv(LOG_PATH, index=False)
        st.success("Restored. Tap Update results.")


def pitching_matchups(games_sel, hist):
    st.write("Each starter's last five starts. Short rest or short recent outings usually mean fewer outs "
             "and strikeouts, especially in the playoffs when managers go to the bullpen early.")
    P = hist[hist.pos_group == "SP"]
    for g in games_sel.itertuples():
        st.markdown(f"<div class='matchup'>{g.label}</div>", unsafe_allow_html=True)
        if val(g, "series"):
            st.caption(g.series)
        cols = st.columns(2)
        for col, side in zip(cols, ("away", "home")):
            pid, name = val(g, f"{side}_pp"), val(g, f"{side}_pp_name")
            team = short(getattr(g, f"{side}_team"), "mlb")
            with col:
                if pid is None:
                    st.write(f"**{team}:** starter not announced yet")
                    continue
                starts = P[P.player_id == f"P{int(pid)}"].sort_values("game_order").tail(5)
                if starts.empty:
                    st.write(f"**{name}** ({team}): no recent starts on record")
                    continue
                last = pd.to_datetime(starts["game_date"].iloc[-1])
                today = pd.Timestamp(g.commence_time).tz_convert("America/New_York").tz_localize(None).normalize()
                rest = (today - last).days
                st.write(f"**{name}** ({team}) · {rest} days since last start")
                t = starts.iloc[::-1]
                st.dataframe(pd.DataFrame({
                    "Date": pd.to_datetime(t.game_date).dt.strftime("%b %-d"),
                    "Opponent": t.opponent.map(lambda x: short(x, "mlb")),
                    "Innings": (t.p_outs // 3).astype(int).astype(str) + "." + (t.p_outs % 3).astype(int).astype(str),
                    "K": t.p_strikeouts.astype(int), "Hits": t.p_hits.astype(int), "Runs (ER)": t.p_er.astype(int),
                    "Pitches": t.p_pitches.fillna(0).astype(int)}), hide_index=True, width="stretch")


# ====================== one sport's page ======================
def sport_page(ns: str):
    mlb = ns == "mlb"
    if mlb:
        leagues, live = ["mlb"], True
        if not ODDS_KEY:
            st.warning("MLB needs real sportsbook lines. Add your Odds API key in Settings.")
            return
        mk_all = markets_for("mlb")
        labels = {MARKETS[k]["label"]: k for k in mk_all}
        picked = st.multiselect("Bet types to pull (each costs about 1 credit per game)", list(labels),
                                default=list(labels), key="mlb_markets")
        market_keys = [labels[x] for x in picked]
    else:
        live = fb_live
        names = st.multiselect("Leagues", ["NFL", "College"], default=["NFL"], key="fb_leagues") if live else ["NFL"]
        leagues = [{"NFL": "nfl", "College": "cfb"}[n] for n in names]
        if "cfb" in leagues and not CFBD_KEY:
            st.warning("College needs a CollegeFootballData key (add CFBD_API_KEY to your Streamlit secrets).")
            leagues = [l for l in leagues if l != "cfb"]
        market_keys = None
        if not live:
            st.caption("Practice lines (turn on real lines in Settings)")

    # ---------- 1. games ----------
    st.markdown("<div class='step'>1. Find games</div>", unsafe_allow_html=True)
    if st.button("Find upcoming games", key=f"{ns}_find", width="stretch"):
        frames, demo_props = [], []
        with st.spinner("Looking up games…"):
            for lg in leagues:
                try:
                    if live:
                        g = live_games(lg, ODDS_KEY, bookmakers=book_keys)
                    else:
                        from demo import mock_slate
                        g, p = mock_slate(cached_history(lg, CFBD_KEY))
                        demo_props.append(p)
                    frames.append(g.assign(league=lg))
                except Exception as e:
                    st.error(f"{lg.upper()}: couldn't load games ({e})")
        games = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if mlb and not games.empty:
            try:   # attach probable starters + series status (free MLB data)
                from slate_mlb import match_schedule
                d = pd.to_datetime(games["commence_time"]).dt.tz_convert("America/New_York").dt.date
                sched = mlb_schedule_info(str(d.min() - pd.Timedelta(days=1)), str(d.max() + pd.Timedelta(days=1)))
                info = match_schedule(games, sched)
                if not info.empty:
                    keep = ["event_id", "home_pp", "away_pp", "home_pp_name", "away_pp_name", "series"]
                    info = info.merge(sched[["game_pk", "home_pp_name", "away_pp_name", "series"]], on="game_pk", how="left")
                    games = games.merge(info[[c for c in keep if c in info]], on="event_id", how="left")
            except Exception as e:
                st.caption(f"Couldn't load probable starters right now ({e}).")
        if not games.empty:
            games["label"] = (games["away_team"] + " @ " + games["home_team"] + ", "
                              + games["commence_time"].map(kickoff))
            games = games.sort_values("commence_time")
        st.session_state[f"{ns}_games"] = games
        st.session_state[f"{ns}_legs"] = None
        st.session_state[f"{ns}_demo_props"] = pd.concat(demo_props) if demo_props else None

    games = st.session_state.get(f"{ns}_games")
    if games is None:
        st.info("Tap the button to load the games for the next 7 days.")
        return
    if games.empty:
        st.warning("No games found in the next 7 days.")
        return

    # ---------- 2. choose ----------
    st.markdown("<div class='step'>2. Choose games</div>", unsafe_allow_html=True)
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    team = st.selectbox("Team", ["All teams"] + teams, key=f"{ns}_team")
    pool = games if team == "All teams" else games[(games.home_team == team) | (games.away_team == team)]
    labels = pool["label"].tolist()
    if team == "All teams":
        quick = st.radio("Show", ["Next game day", "All games", "Pick my own"], horizontal=True, key=f"{ns}_quick")
        if quick == "Next game day":
            first = pd.Timestamp(pool["commence_time"].iloc[0]).tz_convert("America/New_York").date()
            default = [l for l, t in zip(labels, pool["commence_time"])
                       if pd.Timestamp(t).tz_convert("America/New_York").date() == first]
        else:
            default = labels if quick == "All games" else []
    else:
        default = labels
    chosen = st.multiselect("Games", labels, default=default, label_visibility="collapsed", key=f"{ns}_games_{team}")
    if not chosen:
        return
    sel = games[games["label"].isin(chosen)]

    with st.expander(f"Game odds ({len(chosen)} game{'s' if len(chosen) > 1 else ''})", expanded=True):
        rows = []
        for g in sel.itertuples():
            lg = g.league
            h, aw = short(g.home_team, lg), short(g.away_team, lg)
            hs = val(g, "home_spread")
            d = {"Game": f"{aw} @ {h}", "Time": kickoff(g.commence_time)}
            if mlb:
                d["Series"] = val(g, "series") or ""
                d["Starters"] = f"{val(g, 'away_pp_name') or 'TBD'} vs {val(g, 'home_pp_name') or 'TBD'}"
            d.update({
                "Moneyline (away)": money(val(g, "away_ml"), val(g, "away_ml_book")),
                "Moneyline (home)": money(val(g, "home_ml"), val(g, "home_ml_book")),
                "Home win chance": f"{g.home_win_prob:.0%}" if val(g, "home_win_prob") is not None else "–",
                ("Run line" if mlb else "Spread"): "–" if hs is None else (f"{h} {-hs:+g}" if hs != 0 else "Pick'em"),
                "Price (home / away)": f"{money(val(g, 'spread_home_price'))} / {money(val(g, 'spread_away_price'))}",
                "Total": "–" if val(g, "game_total") is None else f"{g.game_total:g}",
                "Over / Under price": f"{money(val(g, 'over_price'))} / {money(val(g, 'under_price'))}",
            })
            rows.append(d)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Best price across sportsbooks, with the book in parentheses. Win chance is the sportsbooks' "
                   "own estimate with their cut removed. These are market odds, not model picks.")

    # ---------- 3. picks ----------
    st.markdown("<div class='step'>3. Get picks</div>", unsafe_allow_html=True)
    if live:
        per = (len(market_keys) if mlb else FOOTBALL_MARKETS) * (2 if use_alts else 1)
        st.caption(f"Getting picks uses about {per * len(sel)} Odds API credits"
                   f"{' (milestone bets double it)' if use_alts else ''}.")
    if st.button("Get picks", type="primary", width="stretch", key=f"{ns}_go", disabled=mlb and not market_keys):
        all_legs = []
        for lg, g in sel.groupby("league"):
            with st.spinner(f"Running the {lg.upper()} models…"):
                try:
                    hist = cached_history(lg, CFBD_KEY)
                    models = cached_models(lg, CFBD_KEY)
                    if live:
                        props, remaining = live_props(lg, ODDS_KEY, g["event_id"].tolist(), log=st.caption,
                                                      markets=market_keys, bookmakers=book_keys, alternates=use_alts)
                        st.session_state["credits"] = remaining
                    else:
                        dp = st.session_state[f"{ns}_demo_props"]
                        props = dp[dp.event_id.isin(g["event_id"])]
                    all_legs.append(build_slate(hist, g, props, models, lg, model_weight))
                except Exception as e:
                    st.error(f"{lg.upper()}: {e}")
        legs = pd.concat([l for l in all_legs if not l.empty], ignore_index=True) if all_legs else pd.DataFrame()
        st.session_state[f"{ns}_legs"] = legs
        st.session_state[f"{ns}_n_games"] = len(sel)
        st.session_state[f"{ns}_sel"] = sel

    legs = st.session_state.get(f"{ns}_legs")
    if legs is None:
        return
    if st.session_state.get("credits"):
        st.caption(f"{st.session_state['credits']} Odds API credits left this month.")
    if legs.empty:
        st.warning("No player props are posted for these games yet. Try again closer to game time.")
        return

    # MLB: batters not in the posted lineup / pitchers who aren't starting are never picks
    blocked = legs["lineup"].isin(["Not in lineup", "Not the listed starter"]) if mlb and "lineup" in legs \
        else pd.Series(False, index=legs.index)
    if skip_one_plus:
        legs = legs[~((legs.side == "Over") & (legs.line == 0.5))]
        blocked = blocked.loc[legs.index]
    usable = legs[~blocked]
    ok = best_lines(usable[(usable.ev > 0) & (usable.p_final >= min_prob) & (usable.disagreement <= max_dis)])
    singles = ok.sort_values("p_final" if mode == "safest" else "ev", ascending=False).head(10)
    flagged = legs[((legs.ev > 0) & (legs.disagreement > max_dis)) | blocked] \
        .sort_values("disagreement", ascending=False).drop_duplicates(["event_id", "player", "market"])
    sports = sorted(legs["league"].str.lower().unique())

    names = ["Safest bets", "Parlays"] + (["Pitching matchups"] if mlb else []) + \
            [f"Check the news ({len(flagged)})", "My record"]
    tabs = dict(zip(names, st.tabs(names)))

    with tabs["Safest bets"]:
        st.write("Single bets the model likes that the sportsbook doesn't strongly disagree with, "
                 + ("most likely to hit first." if mode == "safest" else "best expected return first."))
        if mlb and (usable["lineup"] == "Lineup not posted yet").any():
            st.caption("Some lineups aren't posted yet (usually 2 to 4 hours before first pitch). "
                       "Re-run closer to game time to confirm every batter is playing.")
        if singles.empty:
            st.info("Nothing passes your filters. Try lowering the minimum hit chance in Settings.")
        else:
            st.dataframe(leg_table(singles.to_dict("records"), mlb), hide_index=True, width="stretch")
    with tabs["Parlays"]:
        parlays = parlays_section(ns, ok, st.session_state.get(f"{ns}_n_games", 0), sports)
    if mlb:
        with tabs["Pitching matchups"]:
            pitching_matchups(st.session_state.get(f"{ns}_sel", sel), cached_history("mlb", CFBD_KEY))
    with tabs[names[-2]]:
        st.write("Check these before betting. Big model-vs-book disagreements usually mean news the model "
                 "can't see (an injury, a benching, a role change)." +
                 (" MLB players not in the posted lineup, or pitchers who aren't the listed starter, are also here; "
                  "most books void those bets." if mlb else ""))
        if not flagged.empty:
            st.dataframe(leg_table(flagged.sort_values("disagreement", ascending=False).to_dict("records"), mlb)
                         .drop(columns=["Hit chance"]), hide_index=True, width="stretch")
    with tabs["My record"]:
        record_section(ns, ["MLB"] if mlb else ["NFL", "CFB"])

    if st.button("Save these picks to my record", width="stretch", key=f"{ns}_save"):
        n = save_picks(singles, parlays)
        st.success(f"Saved {n} picks. Check the My record tab after the games.")


tab_fb, tab_mlb = st.tabs(["🏈 Football", "⚾ MLB"])
with tab_fb:
    sport_page("fb")
with tab_mlb:
    sport_page("mlb")
