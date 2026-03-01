from __future__ import annotations

"""Window quality scorer for the watchlist.

Computes a 0-100 score for each symbol indicating how "clean" the current
price action window is for making a decision.

Score components (all 0-100 then weighted sum):
    A. Turnover/volume: is volume drying up (potential breakout setup)?  25 pts
    B. Large-order net flow: institutional accumulation signal?           25 pts
    C. Technical position: RSI, MA position, MACD momentum?              25 pts
    D. Price behaviour: gap_up, distance from 52-week high/low?          25 pts

The score is stored in signal_cache via SettingsDB.
"""

import logging
import sys
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_DATA_ROOT = "data"


# ── Individual score components ────────────────────────────────────────────────

def _score_volume(df: pd.DataFrame) -> float:
    """Volume drying up in recent 5 days vs 20-day MA → higher score = quieter."""
    if len(df) < 20:
        return 50.0
    vol_5d  = df["volume"].iloc[-5:].mean()
    vol_20d = df["volume"].iloc[-20:].mean()
    if vol_20d == 0:
        return 50.0
    ratio = vol_5d / vol_20d  # < 1 means drying, > 1 means expansion
    # Ideal for accumulation: 0.5–0.8 (quiet but not dead)
    if   ratio < 0.3:  return 30.0  # too quiet, may be delisted/suspended
    elif ratio < 0.5:  return 75.0
    elif ratio < 0.8:  return 95.0
    elif ratio < 1.0:  return 70.0
    elif ratio < 1.5:  return 50.0
    else:              return 20.0  # volume expansion, chasing not ideal


def _score_large_order(symbol: str, data_root: str) -> float:
    """Read fund_flow parquet and score based on large-order net flow trend."""
    ff_path = Path(data_root) / "fund_flow" / f"{symbol.replace('.', '_')}.parquet"
    if not ff_path.exists():
        return 50.0  # neutral when no data
    try:
        df = pd.read_parquet(ff_path)
        if df.empty or "large_order_net_ratio" not in df.columns:
            return 50.0
        df = df.sort_values("date").tail(5)
        recent = df["large_order_net_ratio"].dropna()
        if recent.empty:
            return 50.0
        latest = recent.iloc[-1]
        trend_3d = recent.diff().dropna().mean() if len(recent) >= 3 else 0
        # Score: positive and rising is best
        base = 50.0 + latest * 200  # ±25 range for ±12.5% net ratio
        base = max(0.0, min(100.0, base))
        if trend_3d > 0:
            base = min(100.0, base + 10)
        elif trend_3d < 0:
            base = max(0.0, base - 10)
        return base
    except Exception:
        return 50.0


def _score_technical(df: pd.DataFrame) -> float:
    """Score RSI position and MACD momentum (simple inline computation)."""
    if len(df) < 26:
        return 50.0
    close = df["close"]

    # RSI-14
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, float("nan"))
    rsi   = 100 - 100 / (1 + rs)
    rsi_val = rsi.iloc[-1]

    # MACD: 12-26-9
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal
    # Bullish: histogram positive and rising
    hist_val  = hist.iloc[-1]
    hist_prev = hist.iloc[-2] if len(hist) >= 2 else hist_val

    # RSI score: 30-50 is "oversold recovery" (ideal for entry) = 80+
    if   rsi_val < 20:  rsi_score = 40.0   # extremely oversold / possible trap
    elif rsi_val < 35:  rsi_score = 80.0   # oversold recovery zone
    elif rsi_val < 55:  rsi_score = 70.0   # neutral/mild bull
    elif rsi_val < 70:  rsi_score = 50.0   # overbought watch zone
    else:               rsi_score = 20.0   # overbought, avoid chasing

    # MACD score
    if hist_val > 0 and hist_val > hist_prev:
        macd_score = 80.0   # bullish and strengthening
    elif hist_val > 0:
        macd_score = 60.0   # bullish but weakening
    elif hist_val < 0 and hist_val < hist_prev:
        macd_score = 20.0   # bearish and weakening
    else:
        macd_score = 40.0   # transitional

    return (rsi_score + macd_score) / 2


def _score_price_behaviour(df: pd.DataFrame) -> float:
    """Score based on distance from 52-week high/low and gap-up pattern."""
    if len(df) < 5:
        return 50.0
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    high_52w = high.tail(252).max()
    low_52w  = low.tail(252).min()
    current  = close.iloc[-1]

    # Distance to 52-week high (0=at high, 1=at low)
    if high_52w > low_52w:
        dist = (high_52w - current) / (high_52w - low_52w)
    else:
        dist = 0.5

    # Consolidation near lows with recent recovery is ideal
    # dist ~0.6-0.8: recovering from low = good setup
    if   dist < 0.1:  dist_score = 30.0   # at 52w high, chasing
    elif dist < 0.3:  dist_score = 55.0
    elif dist < 0.5:  dist_score = 70.0
    elif dist < 0.7:  dist_score = 80.0   # sweet spot
    elif dist < 0.9:  dist_score = 65.0
    else:             dist_score = 40.0   # near 52w low, possible value trap

    # Gap-up signal in recent 3 days
    prev = df["prev_close"]
    opens = df["open"]
    gap_pcts = ((opens - prev) / prev.replace(0, float("nan"))).tail(3)
    has_gap_up = (gap_pcts > 0.02).any()
    if has_gap_up:
        dist_score = min(100.0, dist_score + 10)

    return dist_score


# ── Master scorer ──────────────────────────────────────────────────────────────

def compute_window_score(
    symbol: str,
    kline_df: pd.DataFrame,
    data_root: str = _DEFAULT_DATA_ROOT,
) -> int:
    """Compute composite window score [0-100] for a symbol.

    Args:
        symbol:    Stock code (e.g., "600000.SH")
        kline_df:  DataFrame with columns: date, open, high, low, close, volume,
                   amount, turnover_rate, prev_close, vwap. Sorted by date.
        data_root: Path to the data root directory.

    Returns:
        Integer score 0-100.
    """
    if kline_df is None or kline_df.empty or len(kline_df) < 5:
        return 0

    w_vol  = 0.25
    w_flow = 0.25
    w_tech = 0.25
    w_price = 0.25

    s_vol   = _score_volume(kline_df)
    s_flow  = _score_large_order(symbol, data_root)
    s_tech  = _score_technical(kline_df)
    s_price = _score_price_behaviour(kline_df)

    composite = (w_vol * s_vol + w_flow * s_flow +
                 w_tech * s_tech + w_price * s_price)
    return max(0, min(100, round(composite)))


def score_watchlist(
    data_root: str = _DEFAULT_DATA_ROOT,
    date_str: str | None = None,
) -> dict[str, int]:
    """Score all symbols in the watchlist and cache results.

    Returns a dict mapping symbol → score.
    """
    import datetime

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from trade_py.db.settings_db import SettingsDB

    db = SettingsDB(data_root)
    symbols = db.watchlist_get()
    if not symbols:
        logger.info("Watchlist is empty, nothing to score")
        return {}

    date_str = date_str or datetime.date.today().isoformat()
    kline_dir = Path(data_root) / "kline"

    scores: dict[str, int] = {}
    for symbol in symbols:
        sym_file = symbol.replace(".", "_") + ".parquet"
        frames = []
        if kline_dir.exists():
            for month_dir in sorted(kline_dir.iterdir()):
                p = month_dir / sym_file
                if p.exists():
                    frames.append(pd.read_parquet(p))

        if frames:
            df = pd.concat(frames, ignore_index=True)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").tail(260)
        else:
            df = pd.DataFrame()

        score = compute_window_score(symbol, df, data_root)
        scores[symbol] = score
        logger.info("%s  window_score=%d", symbol, score)

        # Cache in DB
        db.signal_cache_upsert(date_str, symbol, window_score=score)

    return scores
