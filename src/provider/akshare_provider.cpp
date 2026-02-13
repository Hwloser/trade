#include "trade/provider/i_data_provider.h"
#include "trade/common/config.h"
#include "trade/common/time_utils.h"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <thread>

namespace trade {

// AkShare provider: calls a local AkShare HTTP proxy
// (Python aktools server or custom FastAPI wrapper)
class AkShareProvider : public IDataProvider {
public:
    explicit AkShareProvider(const AkShareConfig& config);
    std::string name() const override;
    std::vector<Bar> fetch_daily(const Symbol& symbol, Date start, Date end) override;
    std::vector<Instrument> fetch_instruments() override;
    bool ping() override;
private:
    AkShareConfig config_;
};

AkShareProvider::AkShareProvider(const AkShareConfig& config) : config_(config) {}

std::string AkShareProvider::name() const { return "akshare"; }

std::vector<Bar> AkShareProvider::fetch_daily(const Symbol& symbol, Date start, Date end) {
    std::vector<Bar> bars;

    std::string code = symbol.substr(0, 6);
    std::string start_str = format_date(start);
    std::string end_str = format_date(end);

    std::string start_compact, end_compact;
    for (char c : start_str) if (c != '-') start_compact += c;
    for (char c : end_str) if (c != '-') end_compact += c;

    std::string path = fmt::format(
        "/api/public/stock_zh_a_hist?"
        "symbol={}&period=daily&start_date={}&end_date={}&adjust=qfq",
        code, start_compact, end_compact);

    for (int attempt = 0; attempt <= config_.retry_count; ++attempt) {
        httplib::Client cli(config_.base_url);
        cli.set_connection_timeout(config_.timeout_ms / 1000);
        cli.set_read_timeout(config_.timeout_ms / 1000);

        auto res = cli.Get(path);
        if (!res || res->status != 200) {
            spdlog::warn("AkShare fetch failed for {} (attempt {}): {}",
                         symbol, attempt,
                         res ? std::to_string(res->status) : "connection failed");
            if (attempt < config_.retry_count) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(config_.retry_delay_ms));
            }
            continue;
        }

        try {
            auto json = nlohmann::json::parse(res->body);
            for (const auto& row : json) {
                Bar bar;
                bar.symbol = symbol;
                std::string date_str = row.value("日期", "");
                if (date_str.empty()) date_str = row.value("date", "");
                if (date_str.size() >= 10) {
                    bar.date = parse_date(date_str.substr(0, 10));
                }
                bar.open = row.value("开盘", row.value("open", 0.0));
                bar.high = row.value("最高", row.value("high", 0.0));
                bar.low = row.value("最低", row.value("low", 0.0));
                bar.close = row.value("收盘", row.value("close", 0.0));
                bar.volume = row.value("成交量", row.value("volume", int64_t(0)));
                bar.amount = row.value("成交额", row.value("amount", 0.0));
                bar.turnover_rate = row.value("换手率", row.value("turnover_rate", 0.0));
                bars.push_back(std::move(bar));
            }
            spdlog::info("Fetched {} bars for {} from AkShare", bars.size(), symbol);
            return bars;
        } catch (const std::exception& e) {
            spdlog::error("Failed to parse AkShare response for {}: {}", symbol, e.what());
            return {};
        }
    }

    spdlog::error("All retries exhausted for {}", symbol);
    return bars;
}

std::vector<Instrument> AkShareProvider::fetch_instruments() {
    std::vector<Instrument> instruments;

    httplib::Client cli(config_.base_url);
    cli.set_connection_timeout(config_.timeout_ms / 1000);
    cli.set_read_timeout(config_.timeout_ms / 1000);

    auto res = cli.Get("/api/public/stock_info_a_code_name");
    if (!res || res->status != 200) {
        spdlog::error("Failed to fetch instrument list from AkShare");
        return instruments;
    }

    try {
        auto json = nlohmann::json::parse(res->body);
        for (const auto& row : json) {
            Instrument inst;
            std::string code = row.value("code", "");
            std::string name_val = row.value("name", "");

            if (code.starts_with("6")) {
                inst.symbol = code + ".SH";
                inst.market = Market::kSH;
            } else if (code.starts_with("0") || code.starts_with("3")) {
                inst.symbol = code + ".SZ";
                inst.market = Market::kSZ;
            } else if (code.starts_with("8") || code.starts_with("4")) {
                inst.symbol = code + ".BJ";
                inst.market = Market::kBJ;
            } else {
                continue;
            }

            inst.name = name_val;

            if (code.starts_with("688")) {
                inst.board = Board::kSTAR;
            } else if (code.starts_with("3")) {
                inst.board = Board::kChiNext;
            } else {
                inst.board = Board::kMain;
            }

            if (name_val.find("*ST") != std::string::npos) {
                inst.status = TradingStatus::kStarST;
                inst.board = Board::kST;
            } else if (name_val.find("ST") != std::string::npos) {
                inst.status = TradingStatus::kST;
                inst.board = Board::kST;
            }

            instruments.push_back(std::move(inst));
        }
        spdlog::info("Fetched {} instruments from AkShare", instruments.size());
    } catch (const std::exception& e) {
        spdlog::error("Failed to parse instrument list: {}", e.what());
    }
    return instruments;
}

bool AkShareProvider::ping() {
    httplib::Client cli(config_.base_url);
    cli.set_connection_timeout(3);
    auto res = cli.Get("/");
    return res && res->status == 200;
}

} // namespace trade
