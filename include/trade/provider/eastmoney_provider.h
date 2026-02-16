#pragma once

#include "trade/provider/i_data_provider.h"
#include "trade/provider/http_client.h"
#include "trade/common/config.h"
#include <nlohmann/json.hpp>

namespace trade {

class EastMoneyProvider : public IDataProvider {
public:
    EastMoneyProvider();
    explicit EastMoneyProvider(const EastMoneyConfig& cfg);

    std::string name() const override { return "eastmoney"; }
    bool ping() override;
    std::vector<Bar> fetch_daily(const Symbol& symbol, Date start, Date end) override;
    std::vector<Instrument> fetch_instruments() override;
    std::optional<Instrument> fetch_instrument(const Symbol& symbol) override;

    bool supports_northbound() const override { return true; }
    std::unordered_map<Symbol, double> fetch_northbound(Date date) override;
    bool supports_margin() const override { return true; }
    std::unordered_map<Symbol, double> fetch_margin(Date date) override;
    bool supports_industry() const override { return true; }
    std::unordered_map<Symbol, SWIndustry> fetch_industry_map() override;
    bool supports_share_capital() const override { return true; }
    std::unordered_map<Symbol, ShareCapital> fetch_share_capital() override;

private:
    EastMoneyConfig config_;
    HttpClient http_;

    // "600000.SH" → "1.600000"
    static std::string to_secid(const Symbol& symbol);
    // "1.600000" → "600000.SH"
    static Symbol from_secid(const std::string& secid);
    // Parse kline response
    static std::vector<Bar> parse_kline(const Symbol& symbol, const nlohmann::json& j);
    // Parse stock list response
    static std::vector<Instrument> parse_stock_list(const nlohmann::json& j);
};

} // namespace trade
