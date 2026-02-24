"""Eastmoney fund flow fetcher for A-share stocks.

Fetches daily large-order / institutional fund flow data from Eastmoney's
public market data API and stores it in Parquet for offline feature computation.

The key derived metric is large_order_net_ratio:
    (超大单净流入 + 大单净流入) / 总成交额

A positive ratio means institutional/large-account money is net-buying;
a negative ratio signals net-selling (potential distribution pattern).

Storage:
    data/fund_flow/{symbol}.parquet

Usage:
    fetcher = FundFlowFetcher("data")
    fetcher.fetch_and_save("600111.SH", days=60)
    df = fetcher.load("600111.SH")
    latest = fetcher.latest_ratio("600111.SH")
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Eastmoney daily fund flow K-line API
# Returns N days of intraday fund flow statistics per symbol.
_FFLOW_URL = (
    "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://data.eastmoney.com/zjlx/",
}

# Field indices in the returned array per trading day:
# [date, xl_in, xl_out, l_in, l_out, m_in, m_out, s_in, s_out, total_in, total_out, close]
# xl = 超大单, l = 大单, m = 中单, s = 小单 (all in 万元 = 10,000 CNY)
_FFIELDS = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62"


def _seccode_to_market(symbol: str) -> tuple[str, str]:
    """Extract (seccode, market_code) from 'NNNNNN.SH' or 'NNNNNN.SZ'."""
    parts = symbol.split(".")
    seccode = parts[0]
    suffix = parts[1].upper() if len(parts) > 1 else "SH"
    market = "1" if suffix == "SH" else "0"
    return seccode, market


def _fetch_raw(symbol: str, days: int = 60) -> list[list]:
    """Fetch raw fund flow time-series from Eastmoney API."""
    seccode, market = _seccode_to_market(symbol)
    params = {
        "lmt":    str(days),
        "klt":    "101",          # daily
        "secid":  f"{market}.{seccode}",
        "fields1": "f1,f2,f3,f7",
        "fields2": _FFIELDS,
    }
    try:
        resp = requests.get(
            _FFLOW_URL, params=params, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        klines = (data.get("data") or {}).get("klines") or []
        rows = []
        for line in klines:
            parts = line.split(",")
            if len(parts) < 12:
                continue
            rows.append(parts)
        return rows
    except Exception as exc:
        logger.warning("FundFlowFetcher: fetch failed for %s: %s", symbol, exc)
        return []


def _parse_rows(symbol: str, raw: list[list]) -> pd.DataFrame:
    """Parse raw API response into a typed DataFrame."""
    records = []
    for parts in raw:
        try:
            trade_date   = str(parts[0])[:10]   # YYYY-MM-DD
            xl_in        = float(parts[1]  or 0) * 1e4   # 万元 → 元
            xl_out       = float(parts[2]  or 0) * 1e4
            l_in         = float(parts[3]  or 0) * 1e4
            l_out        = float(parts[4]  or 0) * 1e4
            m_in         = float(parts[5]  or 0) * 1e4
            m_out        = float(parts[6]  or 0) * 1e4
            s_in         = float(parts[7]  or 0) * 1e4
            s_out        = float(parts[8]  or 0) * 1e4
            total_in     = float(parts[9]  or 0) * 1e4
            total_out    = float(parts[10] or 0) * 1e4
            # parts[11] is close price – not stored here

            xl_net = xl_in - xl_out
            l_net  = l_in  - l_out
            m_net  = m_in  - m_out
            s_net  = s_in  - s_out
            total_turnover = total_in + total_out
            # large_order_net_ratio = (超大单 + 大单 net) / total turnover
            large_net = xl_net + l_net
            ratio = large_net / total_turnover if total_turnover > 1e-6 else 0.0
            # sentiment_behavior_divergence:
            #   retail (small) buying while institutions selling → divergence signal
            # positive = institutions net-buying; negative = net-selling while retail buys
            sbd = (large_net - s_net) / total_turnover if total_turnover > 1e-6 else 0.0

            records.append({
                "symbol":                       symbol,
                "date":                         trade_date,
                "xl_net":                       xl_net,
                "large_net":                    l_net,
                "medium_net":                   m_net,
                "small_net":                    s_net,
                "total_turnover":               total_turnover,
                "large_order_net_ratio":        round(ratio, 6),
                "sentiment_behavior_divergence": round(sbd, 6),
            })
        except (ValueError, IndexError):
            continue

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df = df.sort_values("date").reset_index(drop=True)
    return df


class FundFlowFetcher:
    """Fetch and persist daily fund flow data from Eastmoney.

    Stores one Parquet file per symbol under data_root/fund_flow/.
    The key output feature is large_order_net_ratio, consumed by FeatureBuilder
    as part of Group D (market environment).
    """

    def __init__(self, data_root: str | Path = "data") -> None:
        self._root = Path(data_root) / "fund_flow"
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str) -> Path:
        safe = symbol.replace(".", "_")
        return self._root / f"{safe}.parquet"

    def load(self, symbol: str) -> pd.DataFrame:
        """Load cached fund flow data for a symbol."""
        p = self._path(symbol)
        if not p.exists():
            return pd.DataFrame()
        return pd.read_parquet(p)

    def fetch_and_save(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """Fetch the latest N days of fund flow data and merge with cache.

        Args:
            symbol: Stock code e.g. "600111.SH"
            days:   Number of trading days to fetch (max ~120 from API)

        Returns:
            Combined DataFrame (existing + newly fetched, deduped by date)
        """
        raw = _fetch_raw(symbol, days=days)
        new_df = _parse_rows(symbol, raw)
        if new_df.empty:
            logger.warning("FundFlowFetcher: no data fetched for %s", symbol)
            return self.load(symbol)

        existing = self.load(symbol)
        if not existing.empty:
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["symbol", "date"], keep="last")
            combined = combined.sort_values("date").reset_index(drop=True)
        else:
            combined = new_df

        combined.to_parquet(self._path(symbol), index=False)
        logger.info("FundFlowFetcher: saved %d rows for %s", len(combined), symbol)
        return combined

    def fetch_batch(self, symbols: list[str], days: int = 60) -> None:
        """Fetch fund flow for a list of symbols."""
        for sym in symbols:
            self.fetch_and_save(sym, days=days)

    def latest_ratio(self, symbol: str,
                     as_of: date | None = None) -> float:
        """Return the most recent large_order_net_ratio for a symbol.

        Args:
            symbol: Stock code
            as_of:  Reference date (returns latest row on or before this date)

        Returns:
            Ratio in [-1, +1] range; 0.0 if no data available.
        """
        df = self.load(symbol)
        if df.empty:
            return 0.0
        df["date"] = pd.to_datetime(df["date"])
        if as_of is not None:
            df = df[df["date"] <= pd.Timestamp(as_of)]
        if df.empty:
            return 0.0
        val = df.sort_values("date").iloc[-1]["large_order_net_ratio"]
        return float(val) if val is not None else 0.0

    def divergence_signal(self, symbol: str,
                          as_of: date | None = None) -> float:
        """Return the sentiment_behavior_divergence value.

        Positive: institutions net-buying, retail also buying (aligned).
        Negative: institutions net-selling while retail buys (distribution warning).
        """
        df = self.load(symbol)
        if df.empty:
            return 0.0
        df["date"] = pd.to_datetime(df["date"])
        if as_of is not None:
            df = df[df["date"] <= pd.Timestamp(as_of)]
        if df.empty:
            return 0.0
        val = df.sort_values("date").iloc[-1]["sentiment_behavior_divergence"]
        return float(val) if val is not None else 0.0

    def rolling_ratio(self, symbol: str, window: int = 5,
                      end_date: date | None = None) -> float:
        """Return the rolling mean of large_order_net_ratio over window days."""
        df = self.load(symbol)
        if df.empty:
            return 0.0
        df["date"] = pd.to_datetime(df["date"])
        if end_date is not None:
            df = df[df["date"] <= pd.Timestamp(end_date)]
        if df.empty:
            return 0.0
        tail = df.sort_values("date").tail(window)
        return float(tail["large_order_net_ratio"].mean())
