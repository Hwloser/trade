#include "trade/provider/eastmoney_provider.h"
#include "trade/common/time_utils.h"
#include <spdlog/spdlog.h>
#include <sstream>

namespace trade {

EastMoneyProvider::EastMoneyProvider() : EastMoneyProvider(EastMoneyConfig{}) {}

EastMoneyProvider::EastMoneyProvider(const EastMoneyConfig& cfg)
    : config_(cfg),
      http_(HttpClient::Config{
          .timeout_ms = cfg.timeout_ms,
          .retry_count = cfg.retry_count,
          .retry_delay_ms = cfg.retry_delay_ms,
          .rate_limit_ms = cfg.rate_limit_ms,
      }) {}

std::string EastMoneyProvider::to_secid(const Symbol& symbol) {
    // "600000.SH" → "1.600000", "000001.SZ" → "0.000001"
    if (symbol.size() < 9) return symbol;
    std::string code = symbol.substr(0, 6);
    std::string suffix = symbol.substr(7);
    if (suffix == "SH") return "1." + code;
    if (suffix == "SZ") return "0." + code;
    if (suffix == "BJ") return "0." + code;
    return "1." + code;
}

Symbol EastMoneyProvider::from_secid(const std::string& secid) {
    // "1.600000" → "600000.SH", "0.000001" → "000001.SZ"
    if (secid.size() < 8) return secid;
    std::string market_prefix = secid.substr(0, 2);
    std::string code = secid.substr(2);
    if (market_prefix == "1.") return code + ".SH";
    if (market_prefix == "0.") {
        if (code.starts_with("8") || code.starts_with("4")) return code + ".BJ";
        return code + ".SZ";
    }
    return code + ".SH";
}

bool EastMoneyProvider::ping() {
    auto resp = http_.get("https://push2.eastmoney.com/api/qt/clist/get",
        {{"pn", "1"}, {"pz", "1"}, {"fs", "m:1+t:2"}, {"fields", "f12"}});
    return resp.has_value();
}

std::vector<Bar> EastMoneyProvider::fetch_daily(
    const Symbol& symbol, Date start, Date end) {

    std::string secid = to_secid(symbol);
    std::string start_str, end_str;
    {
        auto s = format_date(start);
        for (char c : s) if (c != '-') start_str += c;
        auto e = format_date(end);
        for (char c : e) if (c != '-') end_str += c;
    }

    // fqt: 1=前复权, 2=后复权, 0=不复权
    std::string fqt = config_.forward_adjust ? "1" : "0";

    auto resp = http_.get(
        "https://push2his.eastmoney.com/api/qt/stock/kline/get",
        {
            {"secid", secid},
            {"fields1", "f1,f2,f3,f4,f5,f6"},
            {"fields2", "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"},
            {"klt", "101"},  // daily
            {"fqt", fqt},
            {"beg", start_str},
            {"end", end_str},
        });

    if (!resp) {
        spdlog::error("Failed to fetch kline for {}", symbol);
        return {};
    }

    try {
        auto j = nlohmann::json::parse(*resp);
        auto bars = parse_kline(symbol, j);
        spdlog::info("Fetched {} bars for {} from eastmoney", bars.size(), symbol);
        return bars;
    } catch (const std::exception& e) {
        spdlog::error("Failed to parse kline for {}: {}", symbol, e.what());
        return {};
    }
}

std::vector<Bar> EastMoneyProvider::parse_kline(
    const Symbol& symbol, const nlohmann::json& j) {
    std::vector<Bar> bars;

    auto data = j.value("data", nlohmann::json{});
    if (data.is_null() || !data.contains("klines")) return bars;

    auto klines = data["klines"];
    if (!klines.is_array()) return bars;

    bars.reserve(klines.size());
    for (const auto& line : klines) {
        std::string s = line.get<std::string>();
        // Format: "date,open,close,high,low,volume,amount,amplitude,chg_pct,chg,turnover"
        // NOTE: close is at index 2 (before high), not index 5
        std::vector<std::string> fields;
        std::istringstream iss(s);
        std::string field;
        while (std::getline(iss, field, ',')) {
            fields.push_back(field);
        }
        if (fields.size() < 11) continue;

        Bar bar;
        bar.symbol = symbol;
        bar.date = parse_date(fields[0]);
        bar.open = std::stod(fields[1]);
        bar.close = std::stod(fields[2]);
        bar.high = std::stod(fields[3]);
        bar.low = std::stod(fields[4]);
        bar.volume = std::stoll(fields[5]);
        bar.amount = std::stod(fields[6]);
        // fields[7] = amplitude (%), fields[8] = chg_pct (%), fields[9] = chg
        bar.turnover_rate = std::stod(fields[10]) / 100.0;  // convert % to ratio

        bars.push_back(std::move(bar));
    }

    return bars;
}

std::vector<Instrument> EastMoneyProvider::fetch_instruments() {
    std::vector<Instrument> all;

    // Fetch from multiple market segments
    // m:0+t:6  = SZ main board
    // m:0+t:80 = SZ ChiNext (创业板)
    // m:1+t:2  = SH main board
    // m:1+t:23 = SH STAR (科创板)
    // m:0+t:81+s:2048 = BSE (北交所)
    std::vector<std::string> segments = {
        "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048"
    };

    for (const auto& fs : segments) {
        int page = 1;
        while (true) {
            auto resp = http_.get(
                "https://push2.eastmoney.com/api/qt/clist/get",
                {
                    {"pn", std::to_string(page)},
                    {"pz", "5000"},
                    {"fs", fs},
                    {"fields", "f12,f14,f13"},
                    // f12=code, f14=name, f13=market_id
                });

            if (!resp) break;

            try {
                auto j = nlohmann::json::parse(*resp);
                auto parsed = parse_stock_list(j);
                if (parsed.empty()) break;
                all.insert(all.end(), parsed.begin(), parsed.end());
                // If we got fewer than page size, we're done
                if (parsed.size() < 5000) break;
                ++page;
            } catch (const std::exception& e) {
                spdlog::error("Failed to parse stock list: {}", e.what());
                break;
            }
        }
    }

    spdlog::info("Fetched {} instruments from eastmoney", all.size());
    return all;
}

std::vector<Instrument> EastMoneyProvider::parse_stock_list(
    const nlohmann::json& j) {
    std::vector<Instrument> instruments;

    auto data = j.value("data", nlohmann::json{});
    if (data.is_null() || !data.contains("diff")) return instruments;

    auto diff = data["diff"];
    if (!diff.is_array()) return instruments;

    for (const auto& item : diff) {
        std::string code = item.value("f12", "");
        std::string name_val = item.value("f14", "");
        int market_id = item.value("f13", 0);

        if (code.empty()) continue;

        Instrument inst;
        if (market_id == 1) {
            inst.symbol = code + ".SH";
            inst.market = Market::kSH;
        } else {
            if (code.starts_with("8") || code.starts_with("4")) {
                inst.symbol = code + ".BJ";
                inst.market = Market::kBJ;
            } else {
                inst.symbol = code + ".SZ";
                inst.market = Market::kSZ;
            }
        }

        inst.name = name_val;

        // Determine board
        if (code.starts_with("688")) {
            inst.board = Board::kSTAR;
        } else if (code.starts_with("3")) {
            inst.board = Board::kChiNext;
        } else if (code.starts_with("8") || code.starts_with("4")) {
            inst.board = Board::kBSE;
        } else {
            inst.board = Board::kMain;
        }

        // ST detection
        if (name_val.find("*ST") != std::string::npos) {
            inst.status = TradingStatus::kStarST;
            inst.board = Board::kST;
        } else if (name_val.find("ST") != std::string::npos) {
            inst.status = TradingStatus::kST;
            inst.board = Board::kST;
        }

        instruments.push_back(std::move(inst));
    }

    return instruments;
}

std::unordered_map<Symbol, double> EastMoneyProvider::fetch_northbound(Date date) {
    std::unordered_map<Symbol, double> result;

    std::string date_str = format_date(date);

    auto resp = http_.get(
        "https://datacenter-web.eastmoney.com/api/data/v1/get",
        {
            {"reportName", "RPT_MUTUAL_STOCK_NORTHSTA"},
            {"columns", "ALL"},
            {"filter", "(TRADE_DATE='" + date_str + "')"},
            {"pageSize", "5000"},
            {"sortColumns", "ADD_MARKET_CAP"},
            {"sortTypes", "-1"},
            {"source", "WEB"},
            {"client", "WEB"},
        });

    if (!resp) return result;

    try {
        auto j = nlohmann::json::parse(*resp);
        auto data = j.value("result", nlohmann::json{});
        if (data.is_null() || !data.contains("data")) return result;

        for (const auto& item : data["data"]) {
            std::string code = item.value("SECURITY_CODE", "");
            std::string market = item.value("SECUCODE", "");
            double net_buy = item.value("ADD_MARKET_CAP", 0.0);

            if (code.empty()) continue;

            // Build symbol from SECUCODE (e.g., "600000.SH")
            Symbol sym;
            if (market.size() >= 9) {
                sym = market.substr(0, 9);
            } else {
                // Guess from code prefix
                if (code.starts_with("6")) sym = code + ".SH";
                else sym = code + ".SZ";
            }

            // net_buy is in yuan, convert to 万元
            result[sym] = net_buy / 10000.0;
        }
    } catch (const std::exception& e) {
        spdlog::error("Failed to parse northbound data: {}", e.what());
    }

    return result;
}

std::unordered_map<Symbol, double> EastMoneyProvider::fetch_margin(Date date) {
    std::unordered_map<Symbol, double> result;

    std::string date_str = format_date(date);

    auto resp = http_.get(
        "https://datacenter-web.eastmoney.com/api/data/v1/get",
        {
            {"reportName", "RPTA_WEB_RZRQ_GGMX"},
            {"columns", "ALL"},
            {"filter", "(TRADE_DATE='" + date_str + "')"},
            {"pageSize", "5000"},
            {"sortColumns", "RZYE"},
            {"sortTypes", "-1"},
            {"source", "WEB"},
            {"client", "WEB"},
        });

    if (!resp) return result;

    try {
        auto j = nlohmann::json::parse(*resp);
        auto data = j.value("result", nlohmann::json{});
        if (data.is_null() || !data.contains("data")) return result;

        for (const auto& item : data["data"]) {
            std::string secucode = item.value("SECUCODE", "");
            double margin_balance = item.value("RZYE", 0.0);

            if (secucode.empty()) continue;

            Symbol sym;
            if (secucode.size() >= 9) {
                sym = secucode.substr(0, 9);
            }
            if (sym.empty()) continue;

            // margin_balance is in yuan, convert to 万元
            result[sym] = margin_balance / 10000.0;
        }
    } catch (const std::exception& e) {
        spdlog::error("Failed to parse margin data: {}", e.what());
    }

    return result;
}

std::unordered_map<Symbol, SWIndustry> EastMoneyProvider::fetch_industry_map() {
    // TODO: implement via eastmoney sector API
    // For now, return empty (will be enhanced in future iteration)
    return {};
}

std::unordered_map<Symbol, IDataProvider::ShareCapital>
EastMoneyProvider::fetch_share_capital() {
    // TODO: implement via eastmoney datacenter API
    // For now, return empty (will be enhanced in future iteration)
    return {};
}

} // namespace trade
