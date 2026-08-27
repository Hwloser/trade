"""Fetch 5y of daily klines for current S&P 500 constituents via yfinance.

Phase 0 of the US-equity route (docs/美股路线计划.md): reuse the A-share factor
evaluation harnesses on US data. Output matches the schema the harnesses read
(symbol, date, open, high, low, close, volume), one parquet per symbol, under
debug/us_baseline/data/kline/.

Caveat carried over from the A-share run: current constituents backfilled
through history means survivorship bias — results are the optimistic case.

Usage: uv run python debug/us_baseline/fetch_sp500.py
"""

from __future__ import annotations

import io
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd
import yfinance as yf

OUT = Path(__file__).parent / "data" / "kline"
CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
    "main/data/constituents.csv"
)
START, END = "2021-01-01", "2026-08-27"
CHUNK = 50


def sp500_symbols() -> list[str]:
    with urllib.request.urlopen(CONSTITUENTS_URL, timeout=30) as r:
        df = pd.read_csv(io.BytesIO(r.read()))
    # Yahoo uses dashes where the official listing uses dots (BRK.B -> BRK-B).
    return sorted(df["Symbol"].str.replace(".", "-", regex=False))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    symbols = sp500_symbols()
    print(f"{len(symbols)} constituents", file=sys.stderr)

    done = failed = 0
    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i:i + CHUNK]
        # auto_adjust gives split/dividend-adjusted OHLC, the hfq equivalent.
        raw = yf.download(chunk, start=START, end=END, auto_adjust=True,
                          progress=False, group_by="ticker", threads=True)
        for sym in chunk:
            try:
                df = raw[sym].dropna(subset=["Close"])
            except KeyError:
                failed += 1
                continue
            if len(df) < 250:  # under a year of history: too short to evaluate
                failed += 1
                continue
            out = pd.DataFrame({
                "symbol": sym,
                "date": df.index.strftime("%Y-%m-%d"),
                "open": df["Open"].values,
                "high": df["High"].values,
                "low": df["Low"].values,
                "close": df["Close"].values,
                "volume": df["Volume"].values,
            })
            out.to_parquet(OUT / f"{sym}.parquet", index=False)
            done += 1
        print(f"  [{min(i + CHUNK, len(symbols))}/{len(symbols)}] "
              f"saved={done} skipped={failed}", file=sys.stderr)
        time.sleep(1)

    print(f"done: {done} symbols saved, {failed} skipped -> {OUT}", file=sys.stderr)
    return 0 if done > 400 else 1


if __name__ == "__main__":
    sys.exit(main())
