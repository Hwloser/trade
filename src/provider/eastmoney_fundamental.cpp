#include "trade/provider/eastmoney_fundamental.h"
#include "trade/common/time_utils.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <sstream>

namespace trade {

namespace {

// Parse "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DD" into sys_days
Date parse_report_date(const std::string& s) {
    // Accept at least 10 characters: "YYYY-MM-DD"
    if (s.size() < 10) return Date{};
    int y = std::stoi(s.substr(0, 4));
    int m = std::stoi(s.substr(5, 2));
    int d = std::stoi(s.substr(8, 2));
    return std::chrono::sys_days{
        std::chrono::year{y} /
        std::chrono::month{static_cast<unsigned>(m)} /
        std::chrono::day{static_cast<unsigned>(d)}};
}

// Extract double from JSON, returning 0.0 if missing or null
double json_double(const nlohmann::json& obj, const std::string& key) {
    if (!obj.contains(key) || obj[key].is_null()) return 0.0;
    if (obj[key].is_number()) return obj[key].get<double>();
    // Sometimes it comes as a string
    if (obj[key].is_string()) {
        try { return std::stod(obj[key].get<std::string>()); }
        catch (...) { return 0.0; }
    }
    return 0.0;
}

} // namespace

EastMoneyFundamental::EastMoneyFundamental()
    : EastMoneyFundamental(EastMoneyConfig{}) {}

EastMoneyFundamental::EastMoneyFundamental(const EastMoneyConfig& cfg)
    : config_(cfg),
      http_(HttpClient::Config{
          .timeout_ms     = cfg.timeout_ms,
          .retry_count    = cfg.retry_count,
          .retry_delay_ms = cfg.retry_delay_ms,
          .rate_limit_ms  = cfg.rate_limit_ms,
      }) {}

std::string EastMoneyFundamental::to_seccode(const Symbol& symbol) {
    // "600000.SH" -> "600000", "000001.SZ" -> "000001"
    auto dot = symbol.find('.');
    if (dot != std::string::npos) return symbol.substr(0, dot);
    return symbol;
}

std::vector<FinancialReport> EastMoneyFundamental::fetch_reports(
    const Symbol& symbol, int limit)
{
    std::string seccode = to_seccode(symbol);

    auto resp = http_.get(
        "https://datacenter.eastmoney.com/securities/api/data/v1/get",
        {
            {"reportName",   "RPT_FIN_INDICATOR_DETAIL"},
            {"columns",      "SECCODE,REPORTDATE,REPORTTYPE,EPSBASIC,ROEJQ,"
                             "MGRCOMSHARENP,BIZINCOME,MAINBUSIINCOME,"
                             "TOTALCASHOPERATEAS,TOTALASSETS,PARENNETPROFIT,"
                             "BPS,OPERATEINCOME"},
            {"filter",       "(SECCODE=\"" + seccode + "\")"},
            {"pageSize",     std::to_string(limit)},
            {"sortColumns",  "REPORTDATE"},
            {"sortTypes",    "-1"},
            {"source",       "WEB"},
            {"client",       "WEB"},
        },
        {
            {"User-Agent",
             "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
             "AppleWebKit/537.36 (KHTML, like Gecko) "
             "Chrome/120.0.0.0 Safari/537.36"},
            {"Referer", "https://data.eastmoney.com"},
        });

    if (!resp) {
        spdlog::warn("EastMoneyFundamental: HTTP fetch failed for {}", symbol);
        return {};
    }

    try {
        auto j = nlohmann::json::parse(*resp);
        auto reports = parse_response(symbol, j);
        spdlog::info("EastMoneyFundamental: fetched {} reports for {}",
                     reports.size(), symbol);
        return reports;
    } catch (const std::exception& e) {
        spdlog::warn("EastMoneyFundamental: JSON parse failed for {}: {}",
                     symbol, e.what());
        return {};
    }
}

std::vector<FinancialReport> EastMoneyFundamental::parse_response(
    const Symbol& symbol, const nlohmann::json& j)
{
    std::vector<FinancialReport> reports;

    auto result = j.value("result", nlohmann::json{});
    if (result.is_null() || !result.contains("data")) return reports;

    auto data = result["data"];
    if (!data.is_array()) return reports;

    reports.reserve(data.size());
    for (const auto& item : data) {
        FinancialReport r;
        r.symbol = symbol;

        // Report date
        std::string report_date_str = item.value("REPORTDATE", "");
        if (report_date_str.empty()) continue;
        r.report_date   = parse_report_date(report_date_str);
        r.publish_date  = r.report_date;  // exact publish date not available

        // Period mapping: "1"=Q1, "2"=Q2(半年报), "3"=Q3, "4"=Annual
        std::string report_type = item.value("REPORTTYPE", "1");
        if (report_type == "1")      r.period = ReportPeriod::Q1;
        else if (report_type == "2") r.period = ReportPeriod::Q2;
        else if (report_type == "3") r.period = ReportPeriod::Q3;
        else if (report_type == "4") r.period = ReportPeriod::Annual;
        else                         r.period = ReportPeriod::Q1;

        // Income
        double bizincome   = json_double(item, "BIZINCOME");
        double mainbiz     = json_double(item, "MAINBUSIINCOME");
        r.revenue          = (bizincome > 0.0) ? bizincome : mainbiz;

        double mgr_np      = json_double(item, "MGRCOMSHARENP");
        double paren_np    = json_double(item, "PARENNETPROFIT");
        r.net_profit       = (mgr_np != 0.0) ? mgr_np : paren_np;

        r.op_profit        = json_double(item, "OPERATEINCOME");
        r.op_cash_flow     = json_double(item, "TOTALCASHOPERATEAS");
        r.total_assets     = json_double(item, "TOTALASSETS");

        r.eps              = json_double(item, "EPSBASIC");
        r.bps              = json_double(item, "BPS");
        r.roe              = json_double(item, "ROEJQ");

        // total_equity: not directly available, skip (leave 0)
        // Callers may compute from bps * shares if needed.

        reports.push_back(std::move(r));
    }

    // Sort ascending by report_date (API returns newest first)
    std::sort(reports.begin(), reports.end(),
              [](const FinancialReport& a, const FinancialReport& b) {
                  return a.report_date < b.report_date;
              });

    return reports;
}

} // namespace trade
