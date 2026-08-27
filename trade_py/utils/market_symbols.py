"""Market detection for mixed CN/US symbol universes.

The A-share helpers in a_share_symbols.py already pass non-A-share tickers
through unchanged, so US tickers like AAPL survive the symbol layer as-is.
What the rest of the pipeline needs is a way to ask which market a symbol
belongs to, because unit conventions differ (A-share volume is stored in
lots of 100 shares, US volume in shares).
"""

from __future__ import annotations

_CN_SUFFIXES = {"SH", "SZ", "BJ"}


def detect_market(symbol: str) -> str:
    """Classify a canonical symbol as "cn" or "us".

    cn: 6-digit code with or without .SH/.SZ/.BJ suffix (600000, 600000.SH)
    us: everything else (AAPL, BRK-B, MSFT)
    """
    value = str(symbol).strip().upper()
    if not value:
        return "us"
    if "." in value:
        code, suffix = value.split(".", 1)
        if suffix in _CN_SUFFIXES and code.isdigit():
            return "cn"
        return "us"
    if value.isdigit() and len(value) == 6:
        return "cn"
    return "us"
