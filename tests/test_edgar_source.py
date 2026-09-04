"""EDGAR ingestion: pagination, dedup, ticker mapping, idempotent sync."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

import trade_py.data.news.edgar as edgar


def _hit(adsh: str, cik: str = "320193", items: list[str] | None = None) -> dict:
    return {"_source": {
        "adsh": adsh, "ciks": [cik.zfill(10)],
        "display_names": ["Apple Inc.  (CIK 0000320193)"],
        "form": "8-K", "file_date": "2026-08-25", "period_ending": "2026-08-25",
        "items": items or ["2.02"], "sics": ["3571"],
    }}


def test_fetch_form_day_paginates_and_dedups(monkeypatch) -> None:
    pages = [
        {"hits": {"hits": [_hit("a-1")] * 6 + [_hit("a-2")] * 4}},   # full page
        {"hits": {"hits": [_hit("a-2"), _hit("a-3")]}},              # short -> stop
    ]
    calls: list[str] = []

    def fake_get(url: str) -> dict:
        calls.append(url)
        return pages[len(calls) - 1]

    monkeypatch.setattr(edgar, "_get_json", fake_get)
    monkeypatch.setattr(edgar.time, "sleep", lambda _s: None)
    rows = edgar.fetch_form_day("8-K", date(2026, 8, 25))
    assert len(calls) == 2
    assert sorted(r["adsh"] for r in rows) == ["a-1", "a-2", "a-3"]
    assert rows[0]["company"] == "Apple Inc."
    assert json.loads(rows[0]["items"]) == ["2.02"]


def test_sync_writes_bronze_and_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(edgar, "load_cik_ticker_map", lambda _root: {"320193": "AAPL"})
    monkeypatch.setattr(edgar, "fetch_form_day",
                        lambda form, day: [dict(_hit("a-1")["_source"],
                                                adsh="a-1", cik="320193",
                                                company="Apple Inc.",
                                                items='["2.02"]', sics='["3571"]')])
    monkeypatch.setattr(edgar.time, "sleep", lambda _s: None)

    d = date(2026, 8, 25)
    r1 = edgar.sync_edgar(tmp_path, d, d)
    assert (r1.days, r1.filings, r1.matched_tickers) == (1, 1, 1)
    out = tmp_path / "sentiment" / "bronze" / "edgar" / "2026" / "08" / "2026-08-25.parquet"
    df = pd.read_parquet(out)
    assert df["ticker"].iloc[0] == "AAPL"
    assert df["url"].iloc[0].startswith("https://www.sec.gov/Archives/edgar/data/320193/")

    r2 = edgar.sync_edgar(tmp_path, d, d)   # second run must skip
    assert (r2.days, r2.days_skipped, r2.filings) == (0, 1, 0)


def test_user_agent_required(monkeypatch) -> None:
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    with pytest.raises(ValueError, match="SEC_EDGAR_USER_AGENT"):
        edgar._user_agent()


def test_ticker_map_prefers_primary_listing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "test test@test")
    monkeypatch.setattr(edgar, "_get_json", lambda _url: {
        "0": {"cik_str": 1067983, "ticker": "BRK-B", "title": "Berkshire"},
        "1": {"cik_str": 1067983, "ticker": "BRK-A", "title": "Berkshire"},
    })
    m = edgar.load_cik_ticker_map(tmp_path)
    assert m["1067983"] == "BRK-B"
    # second call hits the cache, no network
    monkeypatch.setattr(edgar, "_get_json",
                        lambda _url: pytest.fail("cache not used"))
    assert edgar.load_cik_ticker_map(tmp_path)["1067983"] == "BRK-B"


FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
    <issuer>
        <issuerCik>0001576427</issuerCik>
        <issuerName>Criteo S.A.</issuerName>
        <issuerTradingSymbol>CRTO</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0002148390</rptOwnerCik>
            <rptOwnerName>McGogney Connor</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>0</isDirector>
            <isOfficer>1</isOfficer>
            <officerTitle>Chief Financial Officer</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <transactionDate><value>2026-08-24</value></transactionDate>
            <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>671</value></transactionShares>
                <transactionPricePerShare><value>17.37</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>185887</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>"""


def test_parse_form4_xml() -> None:
    rows = edgar.parse_form4_xml(FORM4_XML)
    assert len(rows) == 1
    t = rows[0]
    assert t["ticker"] == "CRTO"
    assert t["issuer_cik"] == "1576427"
    assert t["owner_title"] == "Chief Financial Officer"
    assert t["is_officer"] and not t["is_director"]
    assert t["code"] == "S" and t["acquired_disposed"] == "D"
    assert t["shares"] == 671 and t["price"] == 17.37
    assert t["value_usd"] == pytest.approx(671 * 17.37)
    assert t["shares_after"] == 185887


def test_fetch_form4_day_filters_by_issuer(monkeypatch) -> None:
    hits = [
        ("0001-26-000001:wk-form4_1.xml",
         {"adsh": "0001-26-000001", "ciks": ["0002148390", "0001576427"],
          "file_date": "2026-08-25"}),
        ("0001-26-000002:other.xml",
         {"adsh": "0001-26-000002", "ciks": ["0009999999", "0008888888"],
          "file_date": "2026-08-25"}),
    ]
    fetched: list[str] = []
    monkeypatch.setattr(edgar, "_iter_form_hits", lambda form, day: iter(hits))
    monkeypatch.setattr(edgar, "_get_text",
                        lambda url: (fetched.append(url), FORM4_XML)[1])
    monkeypatch.setattr(edgar.time, "sleep", lambda _s: None)

    rows = edgar.fetch_form4_day(date(2026, 8, 25), issuer_ciks={"1576427"})
    assert len(rows) == 1                      # non-universe filing not fetched
    assert len(fetched) == 1
    assert "1576427" in fetched[0]
    assert rows[0]["adsh"] == "0001-26-000001"
    assert rows[0]["ticker"] == "CRTO"
