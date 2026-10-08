"""Per-market XGBoost models that output a full distribution, not just a point estimate.

XGBoost predicts the expected stat. Walk-forward (time-ordered) out-of-fold residuals are then
bucketed by predicted value, so P(over line) = share of (prediction + historical residual) > line.
This captures skew (yards have long right tails) and heteroscedasticity (a 300-yd QB is noisier
than a 30-yd RB) without assuming a normal distribution.
"""
import gc
import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from config import MARKETS, XGB_PARAMS, markets_for

N_BINS = 5
MODEL_DIR = "models"


def eligible_mask(df: pd.DataFrame, market: str) -> pd.Series:
    m = MARKETS[market]
    usage = f"{m['usage']}_r5"
    pos_ok = df["pos_group"].isin(m["positions"])
    if df.loc[pos_ok, usage].notna().mean() < 0.3:   # e.g. CFB has no targets -> fall back to the stat itself
        usage, thresh = f"{m['stat']}_r5", 1.0
    else:
        thresh = m["min_usage"]
    return pos_ok & (df[usage] >= thresh) & (df["games_played"] >= 2)


class PropModel:
    def __init__(self, league: str, market: str):
        self.league, self.market = league, market
        self.stat = MARKETS[market]["stat"]
        self.params = {**XGB_PARAMS, **({"objective": MARKETS[market]["objective"]} if "objective" in MARKETS[market] else {})}
        self.model = None
        self.features = None
        self.bin_edges = None
        self.bin_resid = None
        self.metrics = {}

    # ---------- training ----------
    def fit(self, df: pd.DataFrame, features: list[str], n_folds: int = 4):
        mask = eligible_mask(df, self.market) & df[self.stat].notna()
        cols = list(dict.fromkeys(features + [self.stat, "game_order", f"{self.stat}_r5"]))
        data = df.loc[mask, cols].sort_values("game_order")
        self.features = features
        # one compact float32 copy of the training data (keeps memory low on free hosting)
        X = data[features].to_numpy(np.float32)
        y = data[self.stat].to_numpy(np.float32)
        go = data["game_order"].to_numpy()

        # Walk-forward CV: train on everything before each block of games, predict the block.
        orders = np.unique(go)
        start = int(len(orders) * 0.4)
        blocks = np.array_split(orders[start:], n_folds)
        oof = np.full(len(data), np.nan)
        for blk in blocks:
            tr, te = go < blk[0], np.isin(go, blk)
            mdl = XGBRegressor(**self.params).fit(X[tr], y[tr])
            oof[te] = mdl.predict(X[te])
            del mdl
            gc.collect()

        ok = ~np.isnan(oof)
        pred, actual = oof[ok], y[ok]
        resid = actual - pred
        self.bin_edges = np.quantile(pred, np.linspace(0, 1, N_BINS + 1)[1:-1])
        bins = np.digitize(pred, self.bin_edges)
        self.bin_resid = [np.sort(resid[bins == b]) for b in range(N_BINS)]

        baseline = data[f"{self.stat}_r5"].to_numpy(np.float64)[ok]
        self.metrics = {
            "rows": int(len(data)),
            "oof_rows": int(ok.sum()),
            "mae_model": float(np.mean(np.abs(resid))),
            "mae_baseline_r5": float(np.nanmean(np.abs(actual - baseline))),
            **self._calibration(pred, actual, baseline),
        }
        self.model = XGBRegressor(**self.params).fit(X, y)
        del X, y, data
        gc.collect()
        return self

    def _calibration(self, pred, actual, baseline):
        """Proxy calibration: use a book-like line (player's 5-game avg, on the hook) since free
        historical prop lines aren't available. Brier < 0.25 means better than a coin flip."""
        mask = ~np.isnan(baseline)
        lines = np.floor(baseline[mask]) + 0.5
        p = self.prob_over(pred[mask], lines)
        hit = (actual[mask] > lines).astype(float)
        conf = np.abs(p - 0.5) >= 0.1
        return {
            "brier": float(np.mean((p - hit) ** 2)),
            "hit_rate_confident_picks": float(np.mean(np.where(p[conf] > 0.5, hit[conf], 1 - hit[conf]))) if conf.any() else float("nan"),
            "confident_pick_share": float(conf.mean()),
        }

    # ---------- inference ----------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(X[self.features].to_numpy(np.float32)), 0, None)

    def prob_over(self, pred, line) -> np.ndarray:
        pred, line = np.atleast_1d(pred).astype(float), np.broadcast_to(line, np.shape(np.atleast_1d(pred))).astype(float)
        bins = np.digitize(pred, self.bin_edges)
        out = np.empty(len(pred))
        for i, (pr, ln, b) in enumerate(zip(pred, line, bins)):
            r = self.bin_resid[b]
            # stat > line  <=>  resid > line - pred ; stats are floored at 0
            out[i] = 1.0 - np.searchsorted(r, ln - pr, side="right") / len(r)
        return np.clip(out, 0.01, 0.99)

    def quantiles(self, pred, qs=(0.25, 0.75)) -> np.ndarray:
        """Likely range: the qs quantiles of outcomes for this projection (rows x len(qs))."""
        pred = np.atleast_1d(pred).astype(float)
        bins = np.digitize(pred, self.bin_edges)
        out = np.array([[np.quantile(self.bin_resid[b], q) + p for q in qs] for p, b in zip(pred, bins)])
        return np.clip(out, 0, None)

    # ---------- persistence ----------
    def path(self):
        return os.path.join(MODEL_DIR, f"{self.league}_{self.market}.joblib")

    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self, self.path())

    @staticmethod
    def load(league, market):
        return joblib.load(os.path.join(MODEL_DIR, f"{league}_{market}.joblib"))


def train_all(df_feat: pd.DataFrame, features, league: str, save=True) -> dict:
    """features: one list for every market, or {role: list} (MLB batters vs pitchers)."""
    models = {}
    for market, spec in markets_for(league).items():
        f = features[spec["role"]] if isinstance(features, dict) else features
        m = PropModel(league, market).fit(df_feat, f)
        if save:
            m.save()
        models[market] = m
    return models
