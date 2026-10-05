"""Per-market XGBoost models that output a full distribution, not just a point estimate.

XGBoost predicts the expected stat. Walk-forward (time-ordered) out-of-fold residuals are then
bucketed by predicted value, so P(over line) = share of (prediction + historical residual) > line.
This captures skew (yards have long right tails) and heteroscedasticity (a 300-yd QB is noisier
than a 30-yd RB) without assuming a normal distribution.
"""
import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from config import MARKETS, XGB_PARAMS

N_BINS = 5
MODEL_DIR = "models"


def eligible_mask(df: pd.DataFrame, market: str) -> pd.Series:
    m = MARKETS[market]
    usage = f"{m['usage']}_r5"
    if df[usage].notna().mean() < 0.3:          # e.g. CFB has no targets -> fall back to the stat itself
        usage, thresh = f"{m['stat']}_r5", 1.0
    else:
        thresh = m["min_usage"]
    return df["pos_group"].isin(m["positions"]) & (df[usage] >= thresh) & (df["games_played"] >= 2)


class PropModel:
    def __init__(self, league: str, market: str):
        self.league, self.market = league, market
        self.stat = MARKETS[market]["stat"]
        self.model = None
        self.features = None
        self.bin_edges = None
        self.bin_resid = None
        self.metrics = {}

    # ---------- training ----------
    def fit(self, df: pd.DataFrame, features: list[str], n_folds: int = 5):
        data = df[eligible_mask(df, self.market) & df[self.stat].notna()].copy()
        data = data.sort_values("game_order")
        self.features = features
        X, y = data[features], data[self.stat].values

        # Walk-forward CV: train on everything before each block of weeks, predict the block.
        orders = np.sort(data["game_order"].unique())
        start = int(len(orders) * 0.4)
        blocks = np.array_split(orders[start:], n_folds)
        oof = np.full(len(data), np.nan)
        for blk in blocks:
            tr = data["game_order"] < blk[0]
            te = data["game_order"].isin(blk)
            mdl = XGBRegressor(**XGB_PARAMS).fit(X[tr], y[tr])
            oof[te.values] = mdl.predict(X[te])

        ok = ~np.isnan(oof)
        pred, actual = oof[ok], y[ok]
        resid = actual - pred
        self.bin_edges = np.quantile(pred, np.linspace(0, 1, N_BINS + 1)[1:-1])
        bins = np.digitize(pred, self.bin_edges)
        self.bin_resid = [np.sort(resid[bins == b]) for b in range(N_BINS)]

        baseline = data[f"{self.stat}_r5"].values[ok]
        self.metrics = {
            "rows": int(len(data)),
            "oof_rows": int(ok.sum()),
            "mae_model": float(np.mean(np.abs(resid))),
            "mae_baseline_r5": float(np.nanmean(np.abs(actual - baseline))),
            **self._calibration(pred, actual, baseline),
        }
        self.model = XGBRegressor(**XGB_PARAMS).fit(X, y)
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
        return np.clip(self.model.predict(X[self.features]), 0, None)

    def prob_over(self, pred, line) -> np.ndarray:
        pred, line = np.atleast_1d(pred).astype(float), np.broadcast_to(line, np.shape(np.atleast_1d(pred))).astype(float)
        bins = np.digitize(pred, self.bin_edges)
        out = np.empty(len(pred))
        for i, (pr, ln, b) in enumerate(zip(pred, line, bins)):
            r = self.bin_resid[b]
            # stat > line  <=>  resid > line - pred ; stats are floored at 0
            out[i] = 1.0 - np.searchsorted(r, ln - pr, side="right") / len(r)
        return np.clip(out, 0.01, 0.99)

    # ---------- persistence ----------
    def path(self):
        return os.path.join(MODEL_DIR, f"{self.league}_{self.market}.joblib")

    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self, self.path())

    @staticmethod
    def load(league, market):
        return joblib.load(os.path.join(MODEL_DIR, f"{league}_{market}.joblib"))


def train_all(df_feat: pd.DataFrame, features: list[str], league: str, save=True) -> dict:
    models = {}
    for market in MARKETS:
        m = PropModel(league, market).fit(df_feat, features)
        if save:
            m.save()
        models[market] = m
    return models
