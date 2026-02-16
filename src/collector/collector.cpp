#include "trade/collector/collector.h"
#include "trade/common/time_utils.h"
#include "trade/storage/parquet_reader.h"
#include <algorithm>
#include <filesystem>
#include <map>
#include <set>
#include <spdlog/spdlog.h>
#include <unordered_map>

namespace trade {

namespace {

void merge_and_write_bars(const std::string& path,
                          std::vector<Bar> incoming,
                          Board default_board) {
    std::vector<Bar> merged;

    if (std::filesystem::exists(path)) {
        try {
            merged = ParquetReader::read_bars(path);
        } catch (const std::exception& e) {
            spdlog::warn("Failed to read existing partition {}: {}", path, e.what());
        }
    }

    std::unordered_map<Date, size_t> idx_by_date;
    idx_by_date.reserve(merged.size() + incoming.size());
    for (size_t i = 0; i < merged.size(); ++i) {
        idx_by_date[merged[i].date] = i;
    }

    for (auto& bar : incoming) {
        auto it = idx_by_date.find(bar.date);
        if (it == idx_by_date.end()) {
            idx_by_date[bar.date] = merged.size();
            merged.push_back(std::move(bar));
        } else {
            merged[it->second] = std::move(bar);
        }
    }

    merged = BarNormalizer::normalize(std::move(merged));
    if (!merged.empty()) {
        Board board = merged.front().board != Board::kMain ? merged.front().board : default_board;
        BarNormalizer::compute_limits(merged, board);
    }

    StoragePath::ensure_dir(path);
    ParquetWriter::write_bars(path, merged);
}

} // namespace

Collector::Collector(std::unique_ptr<IDataProvider> provider,
                     const Config& config)
    : provider_(std::move(provider)),
      paths_(config.data.data_root),
      metadata_(paths_.metadata_db()),
      config_(config) {}

QualityReport Collector::collect_symbol(const Symbol& symbol, Date start, Date end) {
    spdlog::info("Collecting {} [{}, {}]", symbol, format_date(start), format_date(end));

    // 1. Fetch from provider
    auto bars = provider_->fetch_daily(symbol, start, end);
    if (bars.empty()) {
        spdlog::warn("No data returned for {}", symbol);
        return QualityReport{};
    }

    // 2. Normalize (sort, fill prev_close, compute vwap)
    bars = BarNormalizer::normalize(std::move(bars));

    // 3. Compute price limits
    Board board = (!bars.empty() && bars[0].board != Board::kMain)
        ? bars[0].board : Board::kMain;
    BarNormalizer::compute_limits(bars, board);

    // 4. Merge northbound data by date (optional)
    if (provider_->supports_northbound() && !bars.empty()) {
        std::set<Date> dates;
        for (const auto& bar : bars) dates.insert(bar.date);

        std::unordered_map<Date, std::unordered_map<Symbol, double>> north_by_date;
        for (const auto& d : dates) {
            auto north = provider_->fetch_northbound(d);
            if (!north.empty()) {
                north_by_date.emplace(d, std::move(north));
            }
        }

        for (auto& bar : bars) {
            auto dit = north_by_date.find(bar.date);
            if (dit == north_by_date.end()) continue;
            auto sit = dit->second.find(bar.symbol);
            if (sit != dit->second.end()) {
                bar.north_net_buy = sit->second;
            }
        }
    }

    // 5. Merge margin data by date (optional)
    if (provider_->supports_margin() && !bars.empty()) {
        std::set<Date> dates;
        for (const auto& bar : bars) dates.insert(bar.date);

        std::unordered_map<Date, std::unordered_map<Symbol, double>> margin_by_date;
        for (const auto& d : dates) {
            auto margin = provider_->fetch_margin(d);
            if (!margin.empty()) {
                margin_by_date.emplace(d, std::move(margin));
            }
        }

        for (auto& bar : bars) {
            auto dit = margin_by_date.find(bar.date);
            if (dit == margin_by_date.end()) continue;
            auto sit = dit->second.find(bar.symbol);
            if (sit != dit->second.end()) {
                bar.margin_balance = sit->second;
            }
        }
    }

    // 6. Validate fetched window
    auto report = DataValidator::validate(bars);
    if (!report.is_clean()) {
        for (const auto& w : report.warnings) {
            spdlog::warn("{}: {}", symbol, w);
        }
    }

    // 7. Partition by bar year and merge-write (incremental safe)
    std::map<int, std::vector<Bar>> bars_by_year;
    for (const auto& bar : bars) {
        bars_by_year[date_year(bar.date)].push_back(bar);
    }

    for (auto& [year, year_bars] : bars_by_year) {
        auto raw_path = paths_.raw_daily(symbol, year);
        auto curated_path = paths_.curated_daily(symbol, year);

        // raw and curated currently share the same normalized bar model;
        // keeping both paths for future divergence.
        merge_and_write_bars(raw_path, year_bars, board);
        merge_and_write_bars(curated_path, std::move(year_bars), board);
    }

    // 8. Upsert instrument record
    auto inst_opt = provider_->fetch_instrument(symbol);
    if (inst_opt) {
        metadata_.upsert_instrument(*inst_opt);
    }

    // 9. Record download and advance watermark
    metadata_.record_download(symbol,
                              bars.front().date,
                              bars.back().date,
                              static_cast<int64_t>(bars.size()));
    metadata_.upsert_watermark(provider_->name(), "cn_a_daily_bar", symbol, bars.back().date);

    spdlog::info("Collected {} bars for {} (quality: {:.1f}%)",
                 bars.size(), symbol, report.quality_score() * 100);
    return report;
}

void Collector::collect_all(Date start, Date end, ProgressCallback progress) {
    auto instruments = provider_->fetch_instruments();
    spdlog::info("Found {} instruments, collecting [{}, {}]",
                 instruments.size(), format_date(start), format_date(end));

    std::unordered_map<Symbol, SWIndustry> industry_map;
    if (provider_->supports_industry()) {
        industry_map = provider_->fetch_industry_map();
    }
    std::unordered_map<Symbol, IDataProvider::ShareCapital> capital_map;
    if (provider_->supports_share_capital()) {
        capital_map = provider_->fetch_share_capital();
    }

    for (size_t i = 0; i < instruments.size(); ++i) {
        auto& inst = instruments[i];

        if (!industry_map.empty()) {
            auto it = industry_map.find(inst.symbol);
            if (it != industry_map.end()) {
                inst.industry = it->second;
            }
        }

        if (!capital_map.empty()) {
            auto it = capital_map.find(inst.symbol);
            if (it != capital_map.end()) {
                inst.total_shares = it->second.total_shares;
                inst.float_shares = it->second.float_shares;
            }
        }

        metadata_.upsert_instrument(inst);

        if (progress) {
            progress(inst.symbol, static_cast<int>(i + 1),
                     static_cast<int>(instruments.size()));
        }

        collect_symbol(inst.symbol, start, end);
    }
}

void Collector::update_all(ProgressCallback progress) {
    auto today = std::chrono::floor<std::chrono::days>(
        std::chrono::system_clock::now());

    auto symbols = metadata_.symbols_needing_update(today);
    spdlog::info("Found {} symbols needing update", symbols.size());

    for (size_t i = 0; i < symbols.size(); ++i) {
        if (progress) {
            progress(symbols[i], static_cast<int>(i + 1),
                     static_cast<int>(symbols.size()));
        }

        auto last = metadata_.last_download_date(symbols[i]);
        Date start = last ? next_trading_day(*last) : parse_date("2020-01-01");
        collect_symbol(symbols[i], start, today);
    }
}

} // namespace trade
