"""Visualization module for the trading system.

All data computation is done via C++ (through nanobind bindings).
This module only handles rendering with plotly/matplotlib.
"""

from typing import Optional, Sequence
import numpy as np


def candlestick(bars, title: Optional[str] = None):
    """Plot candlestick chart from a list of trade_py.Bar objects.

    Args:
        bars: list of trade_py.Bar or BarSeries
        title: chart title (default: symbol name)
    """
    import plotly.graph_objects as go

    if hasattr(bars, 'bars'):
        symbol = bars.symbol
        bars = bars.bars
    else:
        symbol = bars[0].symbol if bars else ""

    dates = list(range(len(bars)))
    opens = [b.open for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    closes = [b.close for b in bars]
    volumes = [b.volume for b in bars]

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=dates, open=opens, high=highs, low=lows, close=closes,
        name=symbol
    ))
    fig.add_trace(go.Bar(
        x=dates, y=volumes, name="Volume",
        yaxis="y2", opacity=0.3, marker_color="gray"
    ))
    fig.update_layout(
        title=title or f"{symbol} Candlestick",
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Volume", overlaying="y", side="right",
                    showgrid=False),
        xaxis_rangeslider_visible=False,
    )
    return fig


def equity_curve(nav_series, benchmark_series=None,
                 dates=None, title: str = "Equity Curve"):
    """Plot equity curve with optional benchmark comparison.

    Args:
        nav_series: list/array of NAV values
        benchmark_series: optional benchmark NAV values
        dates: optional date labels for x-axis
        title: chart title
    """
    import plotly.graph_objects as go

    x = dates if dates is not None else list(range(len(nav_series)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=nav_series, name="Strategy", mode="lines"))
    if benchmark_series is not None:
        fig.add_trace(go.Scatter(
            x=x, y=benchmark_series, name="Benchmark",
            mode="lines", line=dict(dash="dash")
        ))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="NAV")
    return fig


def drawdown_chart(drawdown_series, dates=None, title: str = "Drawdown"):
    """Plot drawdown curve.

    Args:
        drawdown_series: list/array of drawdown values (negative)
        dates: optional date labels
        title: chart title
    """
    import plotly.graph_objects as go

    x = dates if dates is not None else list(range(len(drawdown_series)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=drawdown_series, name="Drawdown",
        fill="tozeroy", mode="lines",
        line=dict(color="red")
    ))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Drawdown")
    return fig


def monthly_heatmap(monthly_returns: dict, title: str = "Monthly Returns (%)"):
    """Plot monthly return heatmap.

    Args:
        monthly_returns: dict mapping (year, month) -> return_pct
        title: chart title
    """
    import plotly.graph_objects as go

    if not monthly_returns:
        return None

    years = sorted(set(y for y, m in monthly_returns))
    months = list(range(1, 13))
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    z = []
    for year in years:
        row = [monthly_returns.get((year, m), float('nan')) * 100
               for m in months]
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z, x=month_names, y=[str(y) for y in years],
        colorscale="RdYlGn", zmid=0,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row]
              for row in z],
        texttemplate="%{text}",
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title="Month", yaxis_title="Year")
    return fig


def feature_importance(shap_values, feature_names: Sequence[str],
                       top_n: int = 20,
                       title: str = "Feature Importance (|SHAP|)"):
    """Plot SHAP-based feature importance.

    Args:
        shap_values: array of mean absolute SHAP values per feature
        feature_names: feature names corresponding to shap_values
        top_n: number of top features to show
        title: chart title
    """
    import plotly.graph_objects as go

    pairs = sorted(zip(feature_names, shap_values),
                   key=lambda x: abs(x[1]), reverse=True)[:top_n]
    names, values = zip(*pairs) if pairs else ([], [])

    fig = go.Figure(go.Bar(
        x=list(values), y=list(names), orientation="h",
        marker_color=["green" if v > 0 else "red" for v in values]
    ))
    fig.update_layout(
        title=title, xaxis_title="Mean |SHAP|",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def risk_dashboard(dashboard):
    """Display risk monitoring dashboard from a RiskDashboard object.

    Args:
        dashboard: trade_py.risk.RiskDashboard object
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Ex-Ante Risk", "Portfolio Exposure",
                        "Liquidity", "Alerts"],
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "table"}]]
    )

    # Ex-ante VaR gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=dashboard.ex_ante.var_99 * 100,
        title={"text": "VaR 99% (%)"},
        gauge=dict(
            axis=dict(range=[0, 5]),
            bar=dict(color="darkblue"),
            steps=[
                dict(range=[0, 2], color="lightgreen"),
                dict(range=[2, 3], color="yellow"),
                dict(range=[3, 5], color="red"),
            ],
        ),
    ), row=1, col=1)

    # Gross exposure gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=dashboard.gross_exposure * 100,
        title={"text": "Gross Exposure (%)"},
        gauge=dict(axis=dict(range=[0, 100])),
    ), row=1, col=2)

    # Liquidation days gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=dashboard.liquidity.avg_liquidation_days,
        title={"text": "Avg Liquidation Days"},
        gauge=dict(
            axis=dict(range=[0, 5]),
            steps=[
                dict(range=[0, 2.5], color="lightgreen"),
                dict(range=[2.5, 5], color="red"),
            ],
        ),
    ), row=2, col=1)

    # Alerts table
    alert_data = []
    for a in dashboard.alerts:
        alert_data.append([a.metric_name, f"{a.current_value:.4f}",
                          f"{a.threshold:.4f}", a.message])
    if alert_data:
        headers = ["Metric", "Current", "Threshold", "Message"]
        fig.add_trace(go.Table(
            header=dict(values=headers),
            cells=dict(values=list(zip(*alert_data)) if alert_data else [[], [], [], []]),
        ), row=2, col=2)

    fig.update_layout(title="Risk Dashboard", height=600)
    return fig


def factor_exposure_chart(factor_exposures, title: str = "Factor Exposure Over Time"):
    """Plot factor exposure time series.

    Args:
        factor_exposures: list of trade_py.backtest.FactorExposure objects
        title: chart title
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    for fe in factor_exposures:
        fig.add_trace(go.Scatter(
            y=fe.exposures,
            name=fe.factor_name,
            mode="lines",
        ))
    fig.update_layout(
        title=title, xaxis_title="Trading Day",
        yaxis_title="Exposure (z-score)",
    )
    return fig


def sentiment_chart(sentiment_series, symbol: str,
                    title: Optional[str] = None):
    """Plot sentiment indicators over time.

    Args:
        sentiment_series: list of dicts with 'net_sentiment', 'neg_shock', etc.
        symbol: stock symbol
        title: chart title
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not sentiment_series:
        return None

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Net Sentiment", "Neg Shock"])

    x = list(range(len(sentiment_series)))
    net_sent = [s.get("net_sentiment", 0) for s in sentiment_series]
    neg_shock = [s.get("neg_shock", 0) for s in sentiment_series]

    fig.add_trace(go.Scatter(x=x, y=net_sent, name="Net Sentiment",
                              mode="lines+markers"), row=1, col=1)
    fig.add_trace(go.Bar(x=x, y=neg_shock, name="Neg Shock",
                          marker_color="red"), row=2, col=1)

    fig.update_layout(title=title or f"Sentiment - {symbol}", height=500)
    return fig


def performance_summary(report):
    """Print a formatted text summary of a PerformanceReport.

    Args:
        report: trade_py.backtest.PerformanceReport object
    """
    lines = [
        "=== Performance Summary ===",
        f"  Annualised Return:  {report.annualised_return * 100:>8.2f}%",
        f"  Cumulative Return:  {report.cumulative_return * 100:>8.2f}%",
        f"  Sharpe Ratio:       {report.sharpe_ratio:>8.3f}",
        f"  Sortino Ratio:      {report.sortino_ratio:>8.3f}",
        f"  Calmar Ratio:       {report.calmar_ratio:>8.3f}",
        f"  Max Drawdown:       {report.max_drawdown * 100:>8.2f}%",
        f"  Max DD Duration:    {report.max_drawdown_duration:>8d} days",
        f"  Win Rate:           {report.win_rate * 100:>8.2f}%",
        f"  Profit Factor:      {report.profit_factor:>8.3f}",
        f"  Avg Daily Turnover: {report.avg_daily_turnover * 100:>8.2f}%",
        f"  VaR 99%:            {report.var_99 * 100:>8.2f}%",
    ]
    if report.alpha != 0 or report.beta != 0:
        lines.extend([
            f"  Alpha:              {report.alpha * 100:>8.2f}%",
            f"  Beta:               {report.beta:>8.3f}",
            f"  Tracking Error:     {report.tracking_error * 100:>8.2f}%",
        ])
    if report.sharpe_ci_lower != 0:
        lines.append(
            f"  Sharpe 95% CI:      [{report.sharpe_ci_lower:.3f}, {report.sharpe_ci_upper:.3f}]"
        )
    return "\n".join(lines)
