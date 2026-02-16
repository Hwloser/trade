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
    // "600000.SH" → "1.600000", "000001.SZ" → "0.000001", "430047.BJ" → "0.430047"
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
        if (code.starts_with("8") || code.starts_with("4") || code.starts_with("9"))
            return code + ".BJ";
        return code + ".SZ";
    }
    return code + ".SH";
}

bool EastMoneyProvider::ping() {
    auto resp = http_.get("https://82.push2.eastmoney.com/api/qt/clist/get",
        {{"pn", "1"}, {"pz", "1"}, {"np", "1"}, {"fltt", "2"},
         {"fs", "m:1+t:2"}, {"fields", "f12"}});
    return resp.has_value();
}

// ============================================================================
// fetch_daily — 日K线 (参考 AkShare stock_zh_a_hist)
// ============================================================================
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

    // fqt: 1=前复权(qfq), 2=后复权(hfq), 0=不复权
    std::string fqt = config_.forward_adjust ? "1" : "0";

    auto resp = http_.get(
        "https://push2his.eastmoney.com/api/qt/stock/kline/get",
        {
            {"fields1", "f1,f2,f3,f4,f5,f6"},
            {"fields2", "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"},
            {"ut", "7eea3edcaed734bea9cbfc24409ed989"},
            {"klt", "101"},  // daily
            {"fqt", fqt},
            {"secid", secid},
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

// ============================================================================
// fetch_instrument — 单只股票元数据 (使用 push2 spot API)
// ============================================================================
std::optional<Instrument> EastMoneyProvider::fetch_instrument(const Symbol& symbol) {
    std::string secid = to_secid(symbol);
    auto resp = http_.get(
        "https://push2.eastmoney.com/api/qt/stock/get",
        {
            {"secid", secid},
            {"ut", "fa5fd1943c7b386f172d6893dbbd1104"},
            {"fields", "f57,f58,f84,f85,f116,f117,f162,f163,f189"},
            // f57=code, f58=name, f84=total_shares, f85=float_shares,
            // f116=total_cap, f117=float_cap, f162=PE, f163=PB,
            // f189=list_date (YYYYMMDD)
        });

    if (!resp) return std::nullopt;

    try {
        auto j = nlohmann::json::parse(*resp);
        auto data = j.value("data", nlohmann::json{});
        if (data.is_null()) return std::nullopt;

        std::string code = data.value("f57", "");
        std::string name_val = data.value("f58", "");
        if (code.empty()) return std::nullopt;

        Instrument inst;
        inst.symbol = symbol;
        inst.name = name_val;

        // Market/board from symbol
        if (symbol.size() >= 9) {
            std::string suffix = symbol.substr(7);
            if (suffix == "SH") inst.market = Market::kSH;
            else if (suffix == "SZ") inst.market = Market::kSZ;
            else if (suffix == "BJ") inst.market = Market::kBJ;
        }

        if (code.starts_with("688")) inst.board = Board::kSTAR;
        else if (code.starts_with("3")) inst.board = Board::kChiNext;
        else if (code.starts_with("8") || code.starts_with("4") || code.starts_with("9")) inst.board = Board::kBSE;
        else inst.board = Board::kMain;

        // ST detection
        if (name_val.find("*ST") != std::string::npos) {
            inst.status = TradingStatus::kStarST;
            inst.board = Board::kST;
        } else if (name_val.find("ST") != std::string::npos) {
            inst.status = TradingStatus::kST;
            inst.board = Board::kST;
        }

        // total_shares (f84, 股), float_shares (f85, 股)
        int64_t total_shares = data.value("f84", static_cast<int64_t>(0));
        if (total_shares > 0) inst.total_shares = total_shares;
        int64_t float_shares = data.value("f85", static_cast<int64_t>(0));
        if (float_shares > 0) inst.float_shares = float_shares;

        // list_date (f189, YYYYMMDD integer)
        int list_date_int = data.value("f189", 0);
        if (list_date_int > 19900101) {
            int y = list_date_int / 10000;
            int m = (list_date_int % 10000) / 100;
            int d = list_date_int % 100;
            inst.list_date = std::chrono::year{y} / std::chrono::month{static_cast<unsigned>(m)} /
                             std::chrono::day{static_cast<unsigned>(d)};
        }

        return inst;
    } catch (const std::exception& e) {
        spdlog::debug("Failed to fetch instrument for {}: {}", symbol, e.what());
        return std::nullopt;
    }
}

// ============================================================================
// fetch_instruments — 股票列表 (参考 AkShare stock_zh_a_spot_em)
// ============================================================================
std::vector<Instrument> EastMoneyProvider::fetch_instruments() {
    std::vector<Instrument> all;

    // 使用 AkShare 相同的 URL 和参数 (参考 stock_zh_a_spot_em)
    // 服务端限制每页最多 100 条，需分页遍历
    // fs: m:0+t:6 = SZ主板, m:0+t:80 = 创业板, m:1+t:2 = SH主板,
    //     m:1+t:23 = 科创板, m:0+t:81+s:2048 = 北交所
    static constexpr int kPageSize = 100;
    int page = 1;
    while (true) {
        auto resp = http_.get(
            "https://82.push2.eastmoney.com/api/qt/clist/get",
            {
                {"pn", std::to_string(page)},
                {"pz", std::to_string(kPageSize)},
                {"po", "1"},
                {"np", "1"},
                {"ut", "bd1d9ddb04089700cf9c27f6f7426281"},
                {"fltt", "2"},
                {"invt", "2"},
                {"fid", "f12"},
                {"fs", "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048"},
                {"fields", "f12,f14,f13"},
            });

        if (!resp) break;

        try {
            auto j = nlohmann::json::parse(*resp);
            auto parsed = parse_stock_list(j);
            if (parsed.empty()) break;
            all.insert(all.end(), parsed.begin(), parsed.end());
            if (static_cast<int>(parsed.size()) < kPageSize) break;
            ++page;
        } catch (const std::exception& e) {
            spdlog::error("Failed to parse stock list page {}: {}", page, e.what());
            break;
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

    // diff 可能是 array (np=1时) 或 object (默认时)
    // 统一处理：提取每个 item
    auto process_item = [&](const nlohmann::json& item) {
        std::string code = item.value("f12", "");
        std::string name_val = item.value("f14", "");
        int market_id = item.value("f13", 0);

        if (code.empty()) return;

        Instrument inst;
        if (market_id == 1) {
            inst.symbol = code + ".SH";
            inst.market = Market::kSH;
        } else {
            if (code.starts_with("8") || code.starts_with("4") || code.starts_with("9")) {
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
        } else if (code.starts_with("8") || code.starts_with("4") || code.starts_with("9")) {
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
    };

    if (diff.is_array()) {
        for (const auto& item : diff) {
            process_item(item);
        }
    } else if (diff.is_object()) {
        for (const auto& [key, item] : diff.items()) {
            process_item(item);
        }
    }

    return instruments;
}

// ============================================================================
// fetch_northbound — 北向资金净买入 (参考 AkShare stock_hsgt_hist_em)
// 使用 RPT_MUTUAL_DEAL_HISTORY 获取市场级别北向资金流向
// ============================================================================
std::unordered_map<Symbol, double> EastMoneyProvider::fetch_northbound(Date date) {
    std::unordered_map<Symbol, double> result;

    // RPT_MUTUAL_STOCK_NORTHSTA 已停止更新(截止2024-08)
    // 改用 RPT_MUTUAL_STOCK_NORTHSTA 查最近数据
    // 如果无数据则返回空
    std::string date_str = format_date(date);

    auto resp = http_.get(
        "https://datacenter-web.eastmoney.com/api/data/v1/get",
        {
            {"sortColumns", "ADD_MARKET_CAP"},
            {"sortTypes", "-1"},
            {"pageSize", "5000"},
            {"pageNumber", "1"},
            {"reportName", "RPT_MUTUAL_STOCK_NORTHSTA"},
            {"columns", "ALL"},
            {"source", "WEB"},
            {"client", "WEB"},
            {"filter", "(TRADE_DATE='" + date_str + "')(INTERVAL_TYPE=\"1\")"},
        });

    if (!resp) return result;

    try {
        auto j = nlohmann::json::parse(*resp);
        bool success = j.value("success", false);
        if (!success) {
            spdlog::debug("Northbound data unavailable for {}", date_str);
            return result;
        }

        auto data = j.value("result", nlohmann::json{});
        if (data.is_null() || !data.contains("data")) return result;

        for (const auto& item : data["data"]) {
            std::string secucode = item.value("SECUCODE", "");
            double hold_cap = item.value("ADD_MARKET_CAP", 0.0);

            if (secucode.empty()) continue;

            // SECUCODE format: "600000.SH"
            Symbol sym;
            if (secucode.size() >= 9) {
                sym = secucode.substr(0, 9);
            }
            if (sym.empty()) continue;

            // ADD_MARKET_CAP is in 万元
            result[sym] = hold_cap;
        }
    } catch (const std::exception& e) {
        spdlog::error("Failed to parse northbound data: {}", e.what());
    }

    return result;
}

// ============================================================================
// fetch_margin — 融资融券余额 (参考 AkShare stock_margin_em)
// 使用 EastMoney datacenter RPTA_WEB_RZRQ_GGMX，按 DATE 过滤
// 返回沪深两市全部个股融资融券数据
// ============================================================================
std::unordered_map<Symbol, double> EastMoneyProvider::fetch_margin(Date date) {
    std::unordered_map<Symbol, double> result;

    std::string date_str = format_date(date);

    int page = 1;
    while (true) {
        auto resp = http_.get(
            "https://datacenter-web.eastmoney.com/api/data/v1/get",
            {
                {"reportName", "RPTA_WEB_RZRQ_GGMX"},
                {"columns", "SECUCODE,RZYE,RZRQYE"},
                {"pageSize", "5000"},
                {"pageNumber", std::to_string(page)},
                {"sortColumns", "RZYE"},
                {"sortTypes", "-1"},
                {"source", "WEB"},
                {"client", "WEB"},
                {"filter", "(DATE='" + date_str + "')"},
            });

        if (!resp) break;

        try {
            auto j = nlohmann::json::parse(*resp);
            bool success = j.value("success", false);
            if (!success) {
                spdlog::debug("Margin data unavailable for {}", date_str);
                break;
            }

            auto res = j.value("result", nlohmann::json{});
            if (res.is_null() || !res.contains("data")) break;

            auto data = res["data"];
            if (!data.is_array() || data.empty()) break;

            for (const auto& item : data) {
                std::string secucode = item.value("SECUCODE", "");
                double rzye = item.value("RZYE", 0.0);

                if (secucode.empty()) continue;

                // SECUCODE format: "601318.SH" or "300059.SZ"
                Symbol sym;
                if (secucode.size() >= 9) {
                    sym = secucode.substr(0, 9);
                }
                if (sym.empty()) continue;

                // RZYE is in 元, convert to 万元
                result[sym] = rzye / 10000.0;
            }

            int pages = res.value("pages", 0);
            if (page >= pages) break;
            ++page;
        } catch (const std::exception& e) {
            spdlog::error("Failed to parse margin data: {}", e.what());
            break;
        }
    }

    if (!result.empty()) {
        spdlog::info("Fetched margin data for {} stocks on {}", result.size(), date_str);
    }
    return result;
}

// ============================================================================
// fetch_industry_map / fetch_share_capital — TODO
// ============================================================================
std::unordered_map<Symbol, SWIndustry> EastMoneyProvider::fetch_industry_map() {
    return {};
}

std::unordered_map<Symbol, IDataProvider::ShareCapital>
EastMoneyProvider::fetch_share_capital() {
    return {};
}

} // namespace trade
