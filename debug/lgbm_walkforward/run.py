"""Final verdict on the technical factors: can LightGBM beat single-factor IC?

Single-factor RankIC on this panel is ~0 (see debug/eval_baseline). This asks the
one remaining question: does a non-linear combination of the same 11 weak factors
carry usable cross-sectional signal?

Design (pre-registered before looking at results):
  - target  : per-day cross-sectional percentile rank of fwd_5d
  - features: per-day cross-sectional percentile rank of each factor
  - split   : walk-forward, 24m train -> 5-day purge gap -> 3m test, roll 3m
  - metric  : out-of-sample daily RankIC, then ICIR / t-stat across test days
  - controls: best single factor, equal-weight factor blend, shuffled-label null

Pass bar: OOS mean RankIC >= 0.02 AND t >= 2 AND quintile long-short spread
positive after 0.2% round-trip cost. Anything less is a fail.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

PANEL = Path("debug/eval_baseline/results/factor_panel.parquet")
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

HORIZON = 5           # trading days the label looks ahead
PURGE = HORIZON       # gap between train end and test start, kills label leakage
TRAIN_M, TEST_M = 24, 3
COST = 0.002          # 0.2% round-trip, charged to the long-short spread
SEED = 42

FEATURES = [
    "tech_rsi_14", "tech_macd_hist", "tech_macd_cross",
    "tech_kdj_k", "tech_kdj_d", "tech_kdj_j", "tech_kdj_cross",
    "tech_ma_gap_5_20", "tech_price_vs_ma20",
    "tech_volatility_20d", "tech_volume_ratio_5_20",
]

LGB_PARAMS = dict(
    objective="regression", metric="l2",
    num_leaves=31, learning_rate=0.05,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    min_data_in_leaf=200, lambda_l2=1.0,
    verbose=-1, seed=SEED, num_threads=4,
)


def load() -> pd.DataFrame:
    df = duckdb.connect().execute(f'select * from "{PANEL}"').fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=[f"fwd_{HORIZON}d"]).sort_values(["date", "symbol"])

    # Cross-sectional percentile ranks: the model should learn who beats whom
    # today, not what the market regime looked like that year.
    g = df.groupby("date")
    for c in FEATURES:
        df[f"r_{c}"] = g[c].rank(pct=True)
    df["y"] = g[f"fwd_{HORIZON}d"].rank(pct=True)
    return df.dropna(subset=[f"r_{c}" for c in FEATURES] + ["y"])


def windows(dates: np.ndarray) -> list[tuple]:
    """Expanding-origin walk-forward: (train_start, train_end, test_start, test_end)."""
    out = []
    d0, dmax = pd.Timestamp(dates.min()), pd.Timestamp(dates.max())
    test_start = d0 + pd.DateOffset(months=TRAIN_M)
    while test_start + pd.DateOffset(months=TEST_M) <= dmax:
        train_end = test_start - pd.Timedelta(days=PURGE * 2)  # calendar pad for 5 trading days
        out.append((d0, train_end, test_start, test_start + pd.DateOffset(months=TEST_M)))
        test_start += pd.DateOffset(months=TEST_M)
    return out


def daily_ic(df: pd.DataFrame, pred_col: str) -> pd.Series:
    """Spearman IC per day between a prediction and realised forward return."""
    def one(g):
        if len(g) < 20:
            return np.nan
        return spearmanr(g[pred_col], g[f"fwd_{HORIZON}d"]).correlation
    return df.groupby("date").apply(one).dropna()


def quintile_spread(df: pd.DataFrame, pred_col: str) -> float:
    """Mean daily top-minus-bottom quintile return, net of round-trip cost."""
    def one(g):
        if len(g) < 20:
            return np.nan
        q = g[pred_col].rank(pct=True)
        return g.loc[q > .8, f"fwd_{HORIZON}d"].mean() - g.loc[q < .2, f"fwd_{HORIZON}d"].mean()
    s = df.groupby("date").apply(one).dropna()
    return float(s.mean() - COST)


def summarise(ic: pd.Series) -> dict:
    n = len(ic)
    mean, sd = float(ic.mean()), float(ic.std())
    return {
        "days": n,
        "mean_ic": round(mean, 4),
        "icir": round(mean / sd, 3) if sd else 0.0,
        "t_stat": round(mean / sd * np.sqrt(n), 2) if sd else 0.0,
        "hit_rate": round(float((ic > 0).mean()), 3),
    }


def main() -> None:
    import lightgbm as lgb

    df = load()
    rcols = [f"r_{c}" for c in FEATURES]
    wins = windows(df["date"].values.astype("datetime64[ns]"))
    print(f"panel: {len(df):,} rows  {df.symbol.nunique()} symbols  "
          f"{df.date.min():%Y-%m-%d}..{df.date.max():%Y-%m-%d}", file=sys.stderr)
    print(f"walk-forward windows: {len(wins)}\n", file=sys.stderr)

    preds, gains = [], []
    for i, (a, b, c, d) in enumerate(wins, 1):
        tr = df[(df.date >= a) & (df.date <= b)]
        te = df[(df.date >= c) & (df.date < d)].copy()
        if len(te) < 500 or len(tr) < 5000:
            continue

        # Last 10% of the training span is the early-stopping set.
        cut = tr.date.quantile(0.9)
        fit, val = tr[tr.date <= cut], tr[tr.date > cut]

        model = lgb.train(
            LGB_PARAMS,
            lgb.Dataset(fit[rcols], fit["y"]),
            num_boost_round=500,
            valid_sets=[lgb.Dataset(val[rcols], val["y"])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        te["pred"] = model.predict(te[rcols])
        preds.append(te)
        gains.append(pd.Series(model.feature_importance("gain"), index=rcols))

        ic = daily_ic(te, "pred")
        print(f"  win {i:2d}  train {a:%Y-%m}..{b:%Y-%m}  test {c:%Y-%m}..{d:%Y-%m}  "
              f"n={len(te):6,}  IC={ic.mean():+.4f}  iters={model.best_iteration}",
              file=sys.stderr)

    oos = pd.concat(preds).sort_values(["date", "symbol"])

    # Controls, scored on exactly the same out-of-sample rows.
    rng = np.random.default_rng(SEED)
    oos["ctrl_best_single"] = -oos["r_tech_volatility_20d"]   # strongest single factor, sign-corrected
    oos["ctrl_equal_blend"] = oos[rcols].mean(axis=1)
    oos["ctrl_shuffled"] = rng.permutation(oos["pred"].values)

    report = {}
    for name, col in [("lightgbm", "pred"), ("best_single_factor", "ctrl_best_single"),
                      ("equal_weight_blend", "ctrl_equal_blend"), ("shuffled_null", "ctrl_shuffled")]:
        r = summarise(daily_ic(oos, col))
        r["quintile_spread_net"] = round(quintile_spread(oos, col), 5)
        report[name] = r

    imp = pd.concat(gains, axis=1).mean(axis=1).sort_values(ascending=False)
    report["feature_gain"] = {k.replace("r_", ""): round(v, 1) for k, v in imp.items()}

    m = report["lightgbm"]
    report["verdict"] = {
        "bar": "mean_ic>=0.02 and t_stat>=2 and quintile_spread_net>0",
        "passed": bool(m["mean_ic"] >= 0.02 and m["t_stat"] >= 2
                       and m["quintile_spread_net"] > 0),
    }

    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    oos[["date", "symbol", "pred", f"fwd_{HORIZON}d"]].to_parquet(OUT / "oos_predictions.parquet")
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
