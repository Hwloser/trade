#pragma once

#include "trade/model/bar.h"
#include "trade/model/instrument.h"
#include <vector>
#include <string>
#include <functional>

namespace trade {

// Abstract data provider interface
class IDataProvider {
public:
    virtual ~IDataProvider() = default;

    // Provider name (e.g., "akshare", "tushare")
    virtual std::string name() const = 0;

    // Fetch daily bars for a symbol in [start, end]
    virtual std::vector<Bar> fetch_daily(const Symbol& symbol,
                                          Date start, Date end) = 0;

    // Fetch instrument list
    virtual std::vector<Instrument> fetch_instruments() = 0;

    // Test connectivity
    virtual bool ping() = 0;
};

// Callback for progress reporting
using ProgressCallback = std::function<void(const Symbol& symbol, int current, int total)>;

} // namespace trade
