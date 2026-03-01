"""Jupyter notebook helper functions.

Thin wrappers around C++ bindings for convenient notebook usage.
Usage:
    import trade_py as tp
    from trade_py.viz import equity_curve, performance_summary
"""

import numpy as np


def make_bar(symbol: str, open_: float, high: float, low: float,
             close: float, volume: int, date_int: int = 0):
    """Create a Bar object with given values.

    Args:
        symbol: stock symbol (e.g., '600000.SH')
        open_, high, low, close: OHLC prices
        volume: trading volume
        date_int: date as days since epoch
    """
    import trade_py as tp
    bar = tp.Bar()
    bar.symbol = symbol
    bar.date = date_int
    bar.open = open_
    bar.high = high
    bar.low = low
    bar.close = close
    bar.volume = volume
    return bar


def make_feature_set(names: list, symbols: list, matrix):
    """Create a FeatureSet from Python data.

    Args:
        names: list of feature names
        symbols: list of stock symbols
        matrix: numpy array (n_stocks x n_features)
    """
    import trade_py as tp
    fs = tp.features.FeatureSet()
    fs.names = names
    fs.symbols = symbols
    fs.matrix = np.asarray(matrix, dtype=np.float64)
    return fs


def compute_covariance(returns_matrix):
    """Compute Ledoit-Wolf shrinkage covariance matrix.

    Args:
        returns_matrix: numpy array (T x N) of daily returns

    Returns:
        (covariance_matrix, shrinkage_intensity)
    """
    import trade_py as tp
    est = tp.risk.CovarianceEstimator()
    cov = est.estimate(np.asarray(returns_matrix, dtype=np.float64))
    return cov, est.shrinkage_intensity()


def compute_var(weights, covariance, returns_matrix=None,
                confidence: float = 0.99):
    """Compute VaR/CVaR for a portfolio.

    Args:
        weights: portfolio weights (numpy array)
        covariance: covariance matrix
        returns_matrix: optional historical returns for historical/MC VaR
        confidence: confidence level (default 0.99)

    Returns:
        CombinedVaR result
    """
    import trade_py as tp
    cfg = tp.risk.VaRConfig()
    cfg.confidence = confidence
    calc = tp.risk.VaRCalculator(cfg)
    w = np.asarray(weights, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    if returns_matrix is not None:
        ret = np.asarray(returns_matrix, dtype=np.float64)
        return calc.compute(w, cov, ret)
    return calc.parametric_var(w, cov)


def detect_regime(index_prices, breadth=None):
    """Detect current market regime.

    Args:
        index_prices: list/array of index closing prices
        breadth: optional MarketBreadth object

    Returns:
        RegimeResult
    """
    import trade_py as tp
    detector = tp.regime.RegimeDetector()
    prices = list(map(float, index_prices))
    if breadth is None:
        breadth = tp.regime.MarketBreadth()
        breadth.total_stocks = 5000
        breadth.up_stocks = 2500
    return detector.detect(prices, breadth)


def quick_risk_summary(weights, covariance, symbols=None):
    """Print a quick risk summary for a portfolio.

    Args:
        weights: portfolio weights
        covariance: covariance matrix
        symbols: optional stock symbols
    """
    import trade_py as tp
    w = np.asarray(weights, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    n = len(w)

    port_var = float(w @ cov @ w)
    port_vol = np.sqrt(port_var * 252)
    var_99 = float(w @ cov @ w) ** 0.5 * 2.326

    print("=== Quick Risk Summary ===")
    print(f"  Positions:          {np.sum(w > 0.001)}")
    print(f"  Gross Exposure:     {np.sum(np.abs(w)) * 100:.1f}%")
    print(f"  Annual Vol (est):   {port_vol * 100:.2f}%")
    print(f"  VaR 99% (1d):      {var_99 * 100:.2f}%")
    print(f"  Max Weight:         {np.max(w) * 100:.2f}%")
    print(f"  HHI:                {float(np.sum(w ** 2)):.4f}")

    if symbols and len(symbols) == n:
        top_idx = np.argsort(-w)[:5]
        print("  Top 5 positions:")
        for i in top_idx:
            if w[i] > 0.001:
                print(f"    {symbols[i]:>12s}: {w[i] * 100:.2f}%")
