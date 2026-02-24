"""Fundamental data fetcher for A-share stocks (EastMoney datacenter API).

Fetches quarterly financial statements and stores them in Parquet.
Each symbol gets one Parquet file with all historical quarterly data.

Storage: data/fundamental/{symbol}.parquet

Usage:
    fetcher = FundamentalFetcher("data")
    fetcher.fetch_and_save("600703.SH", limit=20)
    df = fetcher.load("600703.SH")
"""
from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
import requests

logger = logging.getLogger(__name__)
_API_URL = 'https://datacenter.eastmoney.com/securities/api/data/v1/get'
_COLUMNS = (
    'SECCODE,REPORTDATE,REPORTTYPE,EPSBASIC,ROEJQ,'
    'MGRCOMSHARENP,BIZINCOME,MAINBUSIINCOME,'
    'TOTALCASHOPERATEAS,TOTALASSETS,PARENNETPROFIT,BPS,OPERATEINCOME'
)
_HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/120.0.0.0 Safari/537.36'),
    'Referer': 'https://data.eastmoney.com',
}
_PERIOD_MAP = {'1': 'Q1', '2': 'Q2', '3': 'Q3', '4': 'Annual'}


def _to_seccode(symbol: str) -> str:
    return symbol.split('.')[0]


def _fetch_raw(seccode: str, limit: int = 20) -> list[dict]:
    filter_val = '(SECCODE=' + chr(34) + seccode + chr(34) + ')'
    params = {
        'reportName': 'RPT_FIN_INDICATOR_DETAIL',
        'columns': _COLUMNS,
        'filter': filter_val,
        'pageSize': str(limit),
        'sortColumns': 'REPORTDATE',
        'sortTypes': '-1',
        'source': 'WEB',
        'client': 'WEB',
    }
    try:
        resp = requests.get(_API_URL, params=params, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return (data.get('result') or {}).get('data') or []
    except Exception as exc:
        logger.warning('EastMoney fundamental fetch failed for %s: %s', seccode, exc)
        return []


def _parse_rows(symbol: str, rows: list[dict]) -> pd.DataFrame:
    records = []
    for item in rows:
        report_date_raw = item.get('REPORTDATE') or ''
        if not report_date_raw:
            continue
        report_date = report_date_raw[:10]
        period = _PERIOD_MAP.get(str(item.get('REPORTTYPE', '1')), 'Q1')
        bizincome = item.get('BIZINCOME') or 0.0
        mainbiz   = item.get('MAINBUSIINCOME') or 0.0
        revenue   = bizincome if bizincome else mainbiz
        mgr_np   = item.get('MGRCOMSHARENP') or 0.0
        paren_np = item.get('PARENNETPROFIT') or 0.0
        net_profit = mgr_np if mgr_np else paren_np
        records.append({
            'symbol':       symbol,
            'report_date':  pd.to_datetime(report_date),
            'publish_date': pd.to_datetime(report_date),
            'period':       period,
            'revenue':      float(revenue),
            'net_profit':   float(net_profit),
            'op_profit':    float(item.get('OPERATEINCOME') or 0.0),
            'op_cash_flow': float(item.get('TOTALCASHOPERATEAS') or 0.0),
            'total_assets': float(item.get('TOTALASSETS') or 0.0),
            'eps':          float(item.get('EPSBASIC') or 0.0),
            'bps':          float(item.get('BPS') or 0.0),
            'roe':          float(item.get('ROEJQ') or 0.0),
        })
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df = df.sort_values('report_date').reset_index(drop=True)
    return df


class FundamentalFetcher:
    """Fetch and persist quarterly financial reports from EastMoney."""

    def __init__(self, data_root: str = 'data') -> None:
        self.data_root = Path(data_root)
        self._fundamental_dir = self.data_root / 'fundamental'
        self._fundamental_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str) -> Path:
        safe = symbol.replace('.', '_')
        return self._fundamental_dir / (safe + '.parquet')

    def load(self, symbol: str) -> pd.DataFrame:
        path = self._path(symbol)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def fetch_and_save(self, symbol: str, limit: int = 20) -> pd.DataFrame:
        seccode = _to_seccode(symbol)
        rows = _fetch_raw(seccode, limit=limit)
        new_df = _parse_rows(symbol, rows)
        if new_df.empty:
            logger.warning('No data fetched for %s', symbol)
            return self.load(symbol)
        existing = self.load(symbol)
        if not existing.empty:
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=['symbol', 'report_date'], keep='last'
            )
            combined = combined.sort_values('report_date').reset_index(drop=True)
        else:
            combined = new_df
        combined.to_parquet(self._path(symbol), index=False)
        logger.info('Saved %d rows for %s', len(combined), symbol)
        return combined


def compute_fundamental_features(
    df: pd.DataFrame,
    current_price: float = 0.0,
    total_shares: int = 0,
) -> dict[str, float]:
    """Compute FundamentalSignal fields from DataFrame of historical reports.

    Columns required: roe, net_profit, revenue, op_cash_flow, bps.
    Sorted ascending by report_date. Returns a dict matching C++ FundamentalSignal.
    """
    result: dict[str, float] = {
        'roe_ttm': 0.0, 'roe_momentum': 0.0,
        'profit_growth_yoy': 0.0, 'revenue_growth_yoy': 0.0,
        'cash_flow_quality': 0.0, 'pe_percentile': 0.0,
        'pe_ttm': 0.0, 'pb': 0.0, 'quarters_available': 0,
    }
    if df.empty:
        return result
    n = len(df)
    q_avail = min(n, 12)
    result['quarters_available'] = q_avail
    last4_start = max(0, n - 4)
    last4 = df.iloc[last4_start:]
    result['roe_ttm'] = float(last4['roe'].mean())
    if n >= 5:
        prev4_end = n - 4
        prev4_start = max(0, n - 8)
        prev4 = df.iloc[prev4_start:prev4_end]
        if not prev4.empty:
            result['roe_momentum'] = result['roe_ttm'] - float(prev4['roe'].mean())
    if n >= 5:
        latest = df.iloc[n - 1]
        year_ago = df.iloc[n - 5]
        prior_profit = float(year_ago['net_profit'])
        prior_revenue = float(year_ago['revenue'])
        if abs(prior_profit) > 1.0:
            result['profit_growth_yoy'] = (
                float(latest['net_profit']) - prior_profit
            ) / abs(prior_profit)
        if prior_revenue > 1.0:
            result['revenue_growth_yoy'] = (
                float(latest['revenue']) - prior_revenue
            ) / prior_revenue
    cf_sum = float(last4['op_cash_flow'].sum())
    np_sum = float(last4['net_profit'].sum())
    if abs(np_sum) > 1.0:
        result['cash_flow_quality'] = max(-3.0, min(5.0, cf_sum / np_sum))
    ttm_np = float(last4['net_profit'].sum())
    if abs(ttm_np) > 1.0 and total_shares > 0 and current_price > 0.0:
        pe = (current_price * total_shares) / ttm_np
        result['pe_ttm'] = max(1.0, min(300.0, pe))
    bps = float(df.iloc[n - 1]['bps'])
    if bps > 1e-6 and current_price > 0.0:
        result['pb'] = max(0.1, min(50.0, current_price / bps))
    if result['pe_ttm'] > 0.0 and total_shares > 0 and current_price > 0.0:
        hist_start = n - q_avail
        pe_history: list[float] = []
        for i in range(hist_start, n):
            np_slice = float(df.iloc[max(0, i - 3): i + 1]['net_profit'].sum())
            if abs(np_slice) > 1.0:
                pe_i = (current_price * total_shares) / np_slice
                if pe_i > 0.0:
                    pe_history.append(max(1.0, min(300.0, pe_i)))
        if len(pe_history) > 1:
            rank = sum(1 for p in pe_history if p < result['pe_ttm'])
            result['pe_percentile'] = rank / len(pe_history)
    return result
