"""Why does a positive RankIC come with a negative long-short spread?

run.py produced mean IC +0.028 (t=4.83) yet a quintile spread of -0.19% gross.
Those two can only disagree if the return distribution inside the buckets is
skewed, so this pulls the buckets apart by mean vs median, and re-tests the
t-stat accounting for the fact that 5-day-horizon daily ICs overlap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

H = "fwd_5d"
PRED = Path(__file__).parent / "results" / "oos_predictions.parquet"


def main() -> None:
    d = pd.read_parquet(PRED)

    def by_quintile(g: pd.DataFrame, col: str) -> pd.Series:
        q = pd.qcut(g["pred"].rank(method="first"), 5, labels=False)
        return g[col].groupby(q).agg(["mean", "median"]).stack()

    tab = d.groupby("date")[["pred", H]].apply(
        lambda g: by_quintile(g, H)).mean().unstack()
    print("每个五分位的 5 日收益 (Q0=模型最看空, Q4=最看多)")
    print((tab * 100).round(3).to_string(), "\n")

    # Daily ICs overlap for HORIZON days, so the naive t-stat is inflated.
    ic = d.groupby("date")[["pred", H]].apply(
        lambda g: spearmanr(g["pred"], g[H]).correlation).dropna()
    n, m, s = len(ic), ic.mean(), ic.std()
    e = ic - m
    lrv = (e ** 2).mean() + 2 * sum(
        (1 - k / 6) * (e[:-k] * e[k:].values).mean() for k in range(1, 6))
    print(f"mean IC = {m:+.4f}")
    print(f"naive t = {m / s * np.sqrt(n):.2f}   (assumes independent days)")
    print(f"Newey-West t (5 lag) = {m / np.sqrt(lrv / n):.2f}   (honest)")


if __name__ == "__main__":
    main()
