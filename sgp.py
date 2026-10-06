"""Same-game parlays, priced honestly.

Player stats in the same game move together (a QB's big day is his WR's big day). So the chance
that every leg hits is NOT the product of the leg chances. We:
  1. learn, from past games, how each kind of stat "surprise" (actual minus recent average)
     correlates with every other kind, for the same player / a teammate / an opponent;
  2. simulate tonight's game thousands of times with those correlations (a Gaussian copula),
     keeping each leg's own hit chance from the model;
  3. report the chance all legs hit and the FAIR odds. Your sportsbook's SGP builder sets its own
     payout; enter it and the app says whether it beats fair odds.
"""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import norm

from config import MARKETS, markets_for
from model import eligible_mask
from odds import decimal_to_american

SHRINK = 150  # pairs needed before we trust a correlation halfway (shrinks noisy ones toward 0)


def correlation_table(feat: pd.DataFrame, max_games: int = 1500) -> dict:
    """{(relation, 'mkt|POS', 'mkt|POS'): rho} learned from historical box scores.
    Uses the most recent `max_games` games so memory stays small on free hosting."""
    league = feat["league"].iloc[0]
    recent = feat.drop_duplicates("game_id").sort_values("game_order").tail(max_games)["game_id"] \
        if "game_order" in feat else feat["game_id"].unique()
    feat = feat[feat["game_id"].isin(set(recent))]
    longs = []
    for mkt, m in markets_for(league).items():
        d = feat[eligible_mask(feat, mkt) & feat[m["stat"]].notna() & feat[f"{m['stat']}_r5"].notna()]
        longs.append(pd.DataFrame({
            "game_id": d["game_id"].values, "player_id": d["player_id"].values, "team": d["team"].values,
            "key": (mkt + "|" + d["pos_group"]).values,
            "resid": (d[m["stat"]] - d[f"{m['stat']}_r5"]).values}))
    L = pd.concat(longs, ignore_index=True)
    # standardize within each key so big-yardage markets don't dominate
    L["z"] = L.groupby("key")["resid"].transform(lambda s: (s - s.mean()) / (s.std() or 1)).astype("float32")
    # compact integer codes keep the pairwise join small enough for free hosting
    keys = sorted(L["key"].unique())
    L = pd.DataFrame({"g": pd.factorize(L["game_id"])[0].astype("int32"),
                      "p": pd.factorize(L["player_id"])[0].astype("int32"),
                      "t": pd.factorize(L["team"])[0].astype("int32"),
                      "k": pd.Categorical(L["key"], categories=keys).codes.astype("int16"),
                      "z": L["z"].values})
    # process games in small batches, keeping only running sums, so memory stays flat
    games = np.unique(L["g"].values)
    sums = None
    for chunk in np.array_split(games, max(1, len(games) // 150)):
        Lc = L[L["g"].isin(chunk)]
        P = Lc.merge(Lc, on="g", suffixes=("_a", "_b"))
        P = P[(P.k_a < P.k_b) | ((P.k_a == P.k_b) & (P.p_a < P.p_b))]  # each unordered pair once
        rel = np.where(P.p_a.values == P.p_b.values, 0, np.where(P.t_a.values == P.t_b.values, 1, 2)).astype("int8")
        za, zb = P.z_a.values.astype("float64"), P.z_b.values.astype("float64")
        agg = pd.DataFrame({"rel": rel, "k1": P.k_a.values, "k2": P.k_b.values, "n": 1.0,
                            "a": za, "b": zb, "aa": za * za, "bb": zb * zb, "ab": za * zb}
                           ).groupby(["rel", "k1", "k2"]).sum()
        sums = agg if sums is None else sums.add(agg, fill_value=0)
        del P, agg
    names = {0: "same_player", 1: "teammate", 2: "opponent"}
    table = {}
    for (r, k1, k2), x in sums.iterrows():
        n = x["n"]
        if n < 15:
            continue
        cov = x["ab"] / n - (x["a"] / n) * (x["b"] / n)
        va, vb = x["aa"] / n - (x["a"] / n) ** 2, x["bb"] / n - (x["b"] / n) ** 2
        if va <= 0 or vb <= 0:
            continue
        rho = cov / np.sqrt(va * vb)
        table[(names[r], keys[k1], keys[k2])] = float(rho * n / (n + SHRINK))
    return table


def _rho(table, a, b):
    rel = "same_player" if a["player_id"] == b["player_id"] else ("teammate" if a["team"] == b["team"] else "opponent")
    k = sorted([f"{a['market']}|{a['pos_group']}", f"{b['market']}|{b['pos_group']}"])
    rho = table.get((rel, k[0], k[1]), 0.0)
    # an Under is the mirror image of an Over
    return rho * (1 if a["side"] == b["side"] else -1)


def _nearest_psd(C):
    w, v = np.linalg.eigh(C)
    C = v @ np.diag(np.clip(w, 1e-6, None)) @ v.T
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def build_sgps(legs: pd.DataFrame, table: dict, n_parlays=6, n_legs=3, min_prob=0.55,
               max_disagreement=0.25, max_reuse=2, pool_size=15, n_sims=20000, seed=0) -> list[dict]:
    """Best same-game parlays across the chosen games, ranked by chance that every leg hits."""
    rng = np.random.default_rng(seed)
    cands = []
    for eid, g in legs.groupby("event_id"):
        # each leg must be one the model leans toward on its own
        pool = g[(g.p_final >= min_prob) & (g.p_final > g.p_market) & (g.disagreement <= max_disagreement)]
        pool = pool.sort_values("p_final", ascending=False).head(pool_size).reset_index(drop=True)
        k = len(pool)
        if k < n_legs:
            continue
        recs = pool.to_dict("records")
        C = np.eye(k)
        for i in range(k):
            for j in range(i + 1, k):
                C[i, j] = C[j, i] = _rho(table, recs[i], recs[j])
        Z = rng.standard_normal((n_sims, k)) @ np.linalg.cholesky(_nearest_psd(C)).T
        # leg i hits when its latent draw lands in its own hit region (probability p_i)
        hits = norm.cdf(Z) < pool["p_final"].values
        for combo in combinations(range(k), n_legs):
            p_joint = hits[:, combo].all(axis=1).mean()
            p_indep = float(np.prod(pool["p_final"].values[list(combo)]))
            cands.append((p_joint, p_indep, eid, combo, recs))
    cands.sort(key=lambda c: c[0], reverse=True)

    out, used = [], {}
    for p_joint, p_indep, eid, combo, recs in cands:
        keys = [(eid, recs[i]["player"], recs[i]["market"]) for i in combo]
        if any(used.get(k, 0) >= max_reuse for k in keys):
            continue
        for k in keys:
            used[k] = used.get(k, 0) + 1
        fair_dec = 1 / max(p_joint, 1e-6)
        out.append({"event_id": eid, "game": recs[0]["game"], "legs": [recs[i] for i in combo],
                    "hit_prob": p_joint, "indep_prob": p_indep, "fair_decimal": fair_dec,
                    "fair_american": decimal_to_american(fair_dec),
                    "naive_decimal": float(np.prod([recs[i]["decimal"] for i in combo]))})
        if len(out) >= n_parlays:
            break
    return out


def american_to_dec(a: float) -> float:
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)
