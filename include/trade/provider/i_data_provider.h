#pragma once

#include "trade/model/bar.h"
#include "trade/model/account.h"
#include "trade/model/instrument.h"
#include <vector>
#include <string>
#include <optional>
#include <functional>
#include <unordered_map>

namespace trade {

// Abstract data provider interface
class IDataProvider {
public:
    virtual ~IDataProvider() = default;

    // Provider name (e.g., "eastmoney", "akshare")
    virtual std::string name() const = 0;

    // Fetch daily bars for a symbol in [start, end]
    virtual std::vector<Bar> fetch_daily(const Symbol& symbol,
                                          Date start, Date end) = 0;

    // Fetch instrument list
    virtual std::vector<Instrument> fetch_instruments() = 0;

    // Fetch single instrument metadata (default: empty)
    virtual std::optional<Instrument> fetch_instrument(const Symbol& /*symbol*/) { return std::nullopt; }

    // Test connectivity
    virtual bool ping() = 0;

    // --- Optional capabilities (default: not supported) ---

    // Northbound (HK Connect) net buy per stock for a date
    virtual bool supports_northbound() const { return false; }
    virtual std::unordered_map<Symbol, double> fetch_northbound(Date /*date*/) { return {}; }

    // Margin balance per stock for a date
    virtual bool supports_margin() const { return false; }
    virtual std::unordered_map<Symbol, double> fetch_margin(Date /*date*/) { return {}; }

    // Industry classification
    virtual bool supports_industry() const { return false; }
    virtual std::unordered_map<Symbol, SWIndustry> fetch_industry_map() { return {}; }

    // Share capital
    virtual bool supports_share_capital() const { return false; }
    struct ShareCapital { int64_t total_shares = 0; int64_t float_shares = 0; };
    virtual std::unordered_map<Symbol, ShareCapital> fetch_share_capital() { return {}; }

    // Broker account details (for THS-like account view / sync)
    virtual bool supports_account_snapshot() const { return false; }
    virtual std::optional<AccountSnapshot> fetch_account_snapshot(
        const std::string& /*account_id*/,
        const std::string& /*auth_payload*/ = "{}") {
        return std::nullopt;
    }
};

// Callback for progress reporting
using ProgressCallback = std::function<void(const Symbol& symbol, int current, int total)>;

} // namespace trade
