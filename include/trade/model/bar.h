#pragma once

#include "trade/common/types.h"
#include <vector>
#include <string>
#include <optional>

namespace trade {

// Daily OHLCV bar
struct Bar {
    Symbol symbol;
    Date date;
    double open = 0.0;
    double high = 0.0;
    double low = 0.0;
    double close = 0.0;
    Volume volume = 0;            // shares
    double amount = 0.0;          // turnover in yuan
    double turnover_rate = 0.0;   // turnover ratio
    double prev_close = 0.0;      // for limit price calculation
    double vwap = 0.0;            // volume-weighted average price

    // Price limits (A-share specific)
    double limit_up = 0.0;
    double limit_down = 0.0;
    bool hit_limit_up = false;
    bool hit_limit_down = false;

    // Trading status
    TradingStatus bar_status = TradingStatus::kNormal;
    Board board = Board::kMain;

    // Fund flow (optional, may not be available daily)
    std::optional<double> north_net_buy;      // 北向资金净买入 (万元)
    std::optional<double> margin_balance;      // 融资余额 (万元)
    std::optional<double> short_sell_volume;   // 融券卖出量

    // Derived
    double change_pct() const {
        return prev_close > 0 ? (close - prev_close) / prev_close : 0.0;
    }

    double amplitude() const {
        return prev_close > 0 ? (high - low) / prev_close : 0.0;
    }

    double open_gap() const {
        return prev_close > 0 ? (open - prev_close) / prev_close : 0.0;
    }

    bool is_valid() const {
        return !symbol.empty() && open > 0 && high >= open &&
               low <= open && low > 0 && close > 0 && volume >= 0;
    }
};

// Extended bar with additional A-share specific fields
struct ExtBar : Bar {
    // Price limits
    double limit_up = 0.0;
    double limit_down = 0.0;
    bool hit_limit_up = false;
    bool hit_limit_down = false;

    // Trading status
    TradingStatus status = TradingStatus::kNormal;
    Board board = Board::kMain;

    // Fund flow (optional, may not be available daily)
    std::optional<double> north_net_buy;      // 北向资金净买入 (万元)
    std::optional<double> margin_balance;      // 融资余额 (万元)
    std::optional<double> short_sell_volume;   // 融券卖出量

    // Computed limit prices from prev_close and board
    void compute_limits() {
        double pct = price_limit_pct(board);
        limit_up = prev_close * (1.0 + pct);
        limit_down = prev_close * (1.0 - pct);
        // Round to 2 decimal places (A-share tick size = 0.01)
        limit_up = static_cast<int>(limit_up * 100 + 0.5) / 100.0;
        limit_down = static_cast<int>(limit_down * 100 + 0.5) / 100.0;
        hit_limit_up = (close >= limit_up - 0.005);
        hit_limit_down = (close <= limit_down + 0.005);
    }
};

// Time series of bars for a single symbol
struct BarSeries {
    Symbol symbol;
    std::vector<Bar> bars;

    size_t size() const { return bars.size(); }
    bool empty() const { return bars.empty(); }
    const Bar& operator[](size_t i) const { return bars[i]; }
    Bar& operator[](size_t i) { return bars[i]; }
};

} // namespace trade
