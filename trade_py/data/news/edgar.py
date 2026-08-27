"""SEC EDGAR filings ingestion (US equities event source).

Phase 1b of the US-equity route (docs/美股路线计划.md). 8-K filings are the
SEC-mandated "material event" disclosures and carry official item codes
(e.g. 2.02 = results announcement, 5.02 = officer changes) — structured event
labels that the A-share pipeline had to guess from news text with an LLM.

Data lands in the existing Bronze layout, one parquet per filing day:
    data/sentiment/bronze/edgar/YYYY/MM/YYYY-MM-DD.parquet
Sync is idempotent: a day whose file already exists is skipped unless
overwrite is requested. Filing metadata is immutable once published, so
re-fetching is never needed except after a bug.

SEC fair-access policy requires a User-Agent identifying the caller with a
contact address; set SEC_EDGAR_USER_AGENT (e.g. "myproject me@example.com")
in the environment / .env. Requests are throttled well under SEC's 10 req/s.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from trade_py.data.pipeline.paths import bronze_path
from trade_py.utils.retry import retry

logger = logging.getLogger(__name__)

SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
PAGE_SIZE = 10           # fixed by the efts API
MAX_PAGES_PER_DAY = 200  # hard stop; a normal day of 8-K is ~20 pages
REQUEST_GAP_SEC = 0.15
TICKER_CACHE_MAX_AGE_SEC = 86400

# Official 8-K item codes -> short English labels, for report rendering.
# The raw codes are what gets stored; this map is presentation only.
ITEM_8K_LABELS = {
    "1.01": "material agreement",
    "1.02": "agreement terminated",
    "1.03": "bankruptcy",
    "2.01": "acquisition/disposition completed",
    "2.02": "results announcement",
    "2.03": "new debt obligation",
    "2.04": "debt acceleration",
    "2.05": "exit/restructuring costs",
    "2.06": "material impairment",
    "3.01": "listing deficiency",
    "3.02": "unregistered equity sale",
    "3.03": "holder rights modified",
    "4.01": "auditor change",
    "4.02": "financials no longer reliable",
    "5.01": "change of control",
    "5.02": "officer/director change",
    "5.03": "charter/bylaws change",
    "5.07": "shareholder vote",
    "7.01": "Reg FD disclosure",
    "8.01": "other material event",
    "9.01": "exhibits",
}


@contextmanager
def _socket_timeout(seconds: float):
    prev = socket.getdefaulttimeout()
    socket.setdefaulttimeout(seconds)
    try:
        yield
    finally:
        socket.setdefaulttimeout(prev)


def _user_agent() -> str:
    ua = os.environ.get("SEC_EDGAR_USER_AGENT", "").strip()
    if not ua:
        raise ValueError(
            "SEC requires an identifying User-Agent with a contact address. "
            'Set SEC_EDGAR_USER_AGENT="<project> <your-email>" in .env.'
        )
    return ua


@retry(delays=(1.0, 3.0, 8.0), on=(Exception,))
def _get_json(url: str) -> dict:
    # efts.sec.gov throws intermittent 500s even on well-formed queries.
    req = urllib.request.Request(url, headers={"User-Agent": _user_agent()})
    with _socket_timeout(30):
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))


def load_cik_ticker_map(data_root: str | Path) -> dict[str, str]:
    """CIK (unpadded str) -> primary ticker, cached on disk for a day."""
    cache = Path(data_root) / "reference" / "cik_tickers.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < TICKER_CACHE_MAX_AGE_SEC:
        return json.loads(cache.read_text())
    raw = _get_json(TICKER_MAP_URL)
    # Entries are ordered by market cap; keep the first ticker seen per CIK
    # (the primary listing), not later share classes.
    mapping: dict[str, str] = {}
    for entry in raw.values():
        cik = str(entry["cik_str"])
        mapping.setdefault(cik, str(entry["ticker"]).upper())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(mapping))
    logger.info("EDGAR ticker map refreshed: %d CIKs", len(mapping))
    return mapping


def _filing_url(cik: str, adsh: str) -> str:
    return (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
        f"{adsh.replace('-', '')}/{adsh}-index.htm"
    )


@dataclass
class EdgarSyncResult:
    days: int = 0
    days_skipped: int = 0
    filings: int = 0
    matched_tickers: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def fetch_form_day(form: str, day: date) -> list[dict]:
    """All filings of `form` filed on `day`, via the full-text search API."""
    rows: list[dict] = []
    for page in range(MAX_PAGES_PER_DAY):
        url = (
            f"{SEARCH_URL}?q=&forms={form}"
            f"&startdt={day.isoformat()}&enddt={day.isoformat()}"
            f"&from={page * PAGE_SIZE}"
        )
        payload = _get_json(url)
        hits = payload.get("hits", {}).get("hits", [])
        for hit in hits:
            src = hit.get("_source", {})
            ciks = src.get("ciks") or [""]
            rows.append({
                "adsh": src.get("adsh", ""),
                "cik": str(int(ciks[0])) if ciks[0] else "",
                # display_names look like "Apple Inc.  (AAPL)  (CIK 0000320193)"
                "company": (src.get("display_names") or [""])[0].split("  (")[0],
                "form": src.get("form", form),
                "file_date": src.get("file_date", day.isoformat()),
                "period_ending": src.get("period_ending", ""),
                "items": json.dumps(src.get("items") or []),
                "sics": json.dumps(src.get("sics") or []),
            })
        if len(hits) < PAGE_SIZE:
            break
        time.sleep(REQUEST_GAP_SEC)
    else:
        logger.warning("EDGAR %s %s: page cap hit (%d pages), day truncated",
                       form, day, MAX_PAGES_PER_DAY)
    # The API returns one hit per document; a filing has several. Dedup by accession.
    return list({r["adsh"]: r for r in rows if r["adsh"]}.values())


def sync_edgar(data_root: str | Path, start: date, end: date,
               forms: tuple[str, ...] = ("8-K",),
               overwrite: bool = False) -> EdgarSyncResult:
    """Fetch filing metadata day by day into the Bronze layer."""
    result = EdgarSyncResult()
    tickers = load_cik_ticker_map(data_root)
    day = start
    while day <= end:
        out = bronze_path(data_root, "edgar", day)
        if out.exists() and not overwrite:
            result.days_skipped += 1
            day += timedelta(days=1)
            continue
        rows: list[dict] = []
        for form in forms:
            rows.extend(fetch_form_day(form, day))
            time.sleep(REQUEST_GAP_SEC)
        if rows:
            df = pd.DataFrame(rows)
            df["ticker"] = df["cik"].map(tickers).fillna("")
            df["url"] = [_filing_url(c, a) if c else "" for c, a in zip(df["cik"], df["adsh"])]
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out, index=False)
            result.filings += len(df)
            result.matched_tickers += int((df["ticker"] != "").sum())
        result.days += 1
        logger.info("EDGAR %s: %d filings", day, len(rows))
        day += timedelta(days=1)
    return result
