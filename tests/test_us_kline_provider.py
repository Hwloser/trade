"""US kline provider: market detection, unit conventions, chain wiring."""

from __future__ import annotations

import pandas as pd
import pytest

from trade_py.data.market.kline.providers import (
    YfinanceKlineProvider, _finalize_frame, build_provider_chain,
)
from trade_py.utils.market_symbols import detect_market


@pytest.mark.parametrize("symbol,market", [
    ("600000.SH", "cn"),
    ("000001.SZ", "cn"),
    ("430047.BJ", "cn"),
    ("600000", "cn"),
    ("AAPL", "us"),
    ("BRK-B", "us"),
    ("MSFT.US", "us"),
    ("", "us"),
])
def test_detect_market(symbol: str, market: str) -> None:
    assert detect_market(symbol) == market


def _fake_yahoo_frame() -> pd.DataFrame:
    idx = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["AAPL"]],
        names=["Price", "Ticker"],
    )
    return pd.DataFrame(
        [[100.0, 105.0, 99.0, 104.0, 1_000_000],
         [104.0, 106.0, 103.0, 105.5, 2_000_000]],
        index=pd.DatetimeIndex(["2026-01-02", "2026-01-05"], name="Date"),
        columns=idx,
    )


def test_normalize_flattens_yahoo_multiindex() -> None:
    df = YfinanceKlineProvider._normalize(_fake_yahoo_frame())
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert len(df) == 2


def test_finalize_us_volume_stays_in_shares() -> None:
    df = YfinanceKlineProvider._normalize(_fake_yahoo_frame())
    out = _finalize_frame("AAPL", df)
    assert out["symbol"].iloc[0] == "AAPL"
    assert out["volume"].iloc[0] == 1_000_000
    # Yahoo gives no turnover amount; vwap must be NaN, never fabricated.
    assert out["vwap"].isna().all()
    # prev_close derives from the prior close.
    assert out["prev_close"].iloc[1] == pytest.approx(104.0)


def test_finalize_cn_volume_still_means_lots() -> None:
    df = pd.DataFrame({
        "date": ["2026-01-05", "2026-01-06"],
        "open": [10.0, 10.2], "high": [10.5, 10.6],
        "low": [9.9, 10.1], "close": [10.2, 10.4],
        "volume": [5_000, 6_000],           # lots (手)
        "amount": [5_100_000.0, 6_240_000.0],
    })
    out = _finalize_frame("600000.SH", df)
    # amount / (volume * 100) — the A-share lot convention must survive.
    assert out["vwap"].iloc[0] == pytest.approx(5_100_000.0 / 500_000)


def test_chain_registration() -> None:
    chain = build_provider_chain("yfinance")
    assert [p.name for p in chain._providers] == ["yfinance"]
