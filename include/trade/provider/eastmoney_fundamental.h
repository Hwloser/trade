#pragma once
#include "trade/model/financial_report.h"
#include "trade/provider/http_client.h"
#include "trade/common/config.h"
#include <nlohmann/json.hpp>
#include <vector>

namespace trade {

// Fetches quarterly financial statements from EastMoney datacenter API.
// API: https://datacenter.eastmoney.com/securities/api/data/v1/get
//      reportName=RPT_FIN_INDICATOR_DETAIL
class EastMoneyFundamental {
public:
    EastMoneyFundamental();
    explicit EastMoneyFundamental(const EastMoneyConfig& cfg);

    // Fetch up to `limit` most recent quarterly reports for a symbol.
    // Returns sorted ascending by report_date.
    std::vector<FinancialReport> fetch_reports(
        const Symbol& symbol, int limit = 12);

private:
    EastMoneyConfig config_;
    HttpClient http_;

    static std::string to_seccode(const Symbol& symbol);
    static std::vector<FinancialReport> parse_response(
        const Symbol& symbol, const nlohmann::json& j);
};

} // namespace trade
