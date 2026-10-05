"""Build a diverse set of parlays from scored legs.

Rules that keep this honest:
  * one leg per game  -> legs are ~independent, so joint prob = product. (Same-game legs are
    correlated and books price SGPs differently, so they're excluded by default.)
  * every leg must be +EV on its own; a parlay of -EV legs compounds the house edge.
  * each leg can appear in at most `max_reuse` parlays so one bad beat can't sink the whole card.
"""
import numpy as np
import pandas as pd

from odds import decimal_to_american


def _score(p, dec, mode):
    ev = p * dec - 1
    if mode == "safest":
        return p if ev > 0 else -1
    if mode == "value":
        return ev
    return ev * np.sqrt(p)   # balanced: EV, tilted toward parlays that actually hit


def kelly_fraction(p, dec, frac=0.25, cap=0.02):
    b = dec - 1
    f = (p * b - (1 - p)) / b
    return float(np.clip(f * frac, 0, cap))


def build_parlays(legs: pd.DataFrame, n_parlays=6, n_legs=3, mode="balanced",
                  min_leg_prob=0.55, min_leg_ev=0.0, max_disagreement=0.25, max_reuse=2,
                  beam_width=300) -> list[dict]:
    pool = legs[(legs.p_final >= min_leg_prob) & (legs.ev > min_leg_ev)
                & (legs.disagreement <= max_disagreement)].copy()
    pool = pool.sort_values("p_final" if mode == "safest" else "ev", ascending=False).head(60).reset_index(drop=True)
    if len(pool) < n_legs:
        return []

    usage = np.zeros(len(pool), dtype=int)
    parlays, seen = [], set()
    for _ in range(n_parlays):
        avail = [i for i in range(len(pool)) if usage[i] < max_reuse]
        beams = [((), 1.0, 1.0)]  # (leg idxs, joint p, joint decimal)
        for _depth in range(n_legs):
            nxt = []
            for idxs, p, d in beams:
                games = {pool.at[i, "event_id"] for i in idxs}
                players = {pool.at[i, "player"] for i in idxs}
                start = idxs[-1] + 1 if idxs else 0
                for j in avail:
                    if j < start or pool.at[j, "event_id"] in games or pool.at[j, "player"] in players:
                        continue
                    nxt.append((idxs + (j,), p * pool.at[j, "p_final"], d * pool.at[j, "decimal"]))
            if not nxt:
                break
            nxt.sort(key=lambda b: _score(b[1], b[2], mode), reverse=True)
            beams = nxt[:beam_width]
        beams = [b for b in beams if len(b[0]) == n_legs and frozenset(b[0]) not in seen]
        if not beams:
            break
        idxs, p, d = beams[0]
        seen.add(frozenset(idxs))
        usage[list(idxs)] += 1
        parlays.append({
            "legs": pool.loc[list(idxs)].to_dict("records"),
            "hit_prob": p, "decimal": d, "american": decimal_to_american(d),
            "ev": p * d - 1, "kelly_stake_pct": kelly_fraction(p, d) * 100,
        })
    return parlays
