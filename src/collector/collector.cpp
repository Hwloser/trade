#include "trade/collector/collector.h"
#include "trade/common/time_utils.h"
#include "trade/storage/parquet_reader.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <spdlog/spdlog.h>
#include <unordered_map>

namespace trade {

namespace {

std::vector<Bar> merge_by_date(std::vector<Bar> existing, std::vector<Bar> incoming) {
    std::unordered_map<Date, size_t> idx_by_date;
    idx_by_date.reserve(existing.size() + incoming.size());

    for (size_t i = 0; i < existing.size(); ++i) {
        idx_by_date[existing[i].date] = i;
    }

    for (auto& bar : incoming) {
        auto it = idx_by_date.find(bar.date);
        if (it == idx_by_date.end()) {
            idx_by_date[bar.date] = existing.size();
            existing.push_back(std::move(bar));
        } else {
            existing[it->second] = std::move(bar);
        }
    }

    std::sort(existing.begin(), existing.end(),
              [](const Bar& a, const Bar& b) { return a.date < b.date; });
    return existing;
}

void merge_and_write_raw_bars(const std::string& path,
                              std::vector<Bar> incoming,
                              std::optional<Date> partition_max_date = std::nullopt) {
    std::vector<Bar> merged;
    if (std::filesystem::exists(path)) {
        try {
            merged = ParquetReader::read_bars(path);
        } catch (const std::exception& e) {
            spdlog::warn("Failed to read existing raw partition {}: {}", path, e.what());
        }
    }

    merged = merge_by_date(std::move(merged), std::move(incoming));

    // Raw layer keeps provider fields close to source semantics.
    ParquetWriter::write_bars(path, merged, ParquetWriter::MergeMode::kReplace, partition_max_date);
}

void merge_and_write_silver_bars(const std::string& path,
                                 std::vector<Bar> incoming,
                                 Board default_board,
                                 std::optional<Date> partition_max_date = std::nullopt) {
    std::vector<Bar> merged;

    if (std::filesystem::exists(path)) {
        try {
            merged = ParquetReader::read_bars(path);
        } catch (const std::exception& e) {
            spdlog::warn("Failed to read existing silver partition {}: {}", path, e.what());
        }
    }

    merged = merge_by_date(std::move(merged), std::move(incoming));
    merged = BarNormalizer::normalize(std::move(merged));
    if (!merged.empty()) {
        Board board = merged.front().board != Board::kMain ? merged.front().board : default_board;
        BarNormalizer::compute_limits(merged, board);
    }

    ParquetWriter::write_bars(path, merged, ParquetWriter::MergeMode::kReplace, partition_max_date);
}

void record_market_quality(MetadataStore& metadata,
                           const std::string& run_id,
                           const Symbol& symbol,
                           const QualityReport& report,
                           std::optional<Date> event_date) {
    MetadataStore::QualityCheckRecord qc;
    qc.run_id = run_id;
    qc.dataset_id = "silver.cn_a.daily";
    qc.check_name = "quality_score";
    qc.metric_value = report.quality_score();
    qc.threshold_value = 0.95;
    qc.status = qc.metric_value >= qc.threshold_value ? "pass" : "warn";
    qc.severity = qc.metric_value >= qc.threshold_value ? "info" : "warning";
    qc.message = symbol + " quality=" + std::to_string(qc.metric_value);
    qc.event_date = event_date;
    metadata.record_quality_check(qc);

    MetadataStore::QualityCheckRecord dup = qc;
    dup.check_name = "duplicate_dates";
    dup.metric_value = static_cast<double>(report.duplicate_dates);
    dup.threshold_value = 0.0;
    dup.status = report.duplicate_dates == 0 ? "pass" : "warn";
    dup.severity = report.duplicate_dates == 0 ? "info" : "warning";
    dup.message = symbol + " duplicates=" + std::to_string(report.duplicate_dates);
    metadata.record_quality_check(dup);

    MetadataStore::QualityCheckRecord pa = qc;
    pa.check_name = "price_anomalies";
    pa.metric_value = static_cast<double>(report.price_anomalies);
    pa.threshold_value = 0.0;
    pa.status = report.price_anomalies == 0 ? "pass" : "warn";
    pa.severity = report.price_anomalies == 0 ? "info" : "warning";
    pa.message = symbol + " anomalies=" + std::to_string(report.price_anomalies);
    metadata.record_quality_check(pa);
}

} // namespace

Collector::Collector(std::unique_ptr<IDataProvider> provider,
                     const Config& config)
    : provider_(std::move(provider)),
      paths_(config.data.data_root),
      metadata_(paths_.metadata_db()),
      config_(config) {
    ParquetStore::configure_runtime(config_.data, config_.storage);
}

QualityReport Collector::collect_symbol(const Symbol& symbol, Date start, Date end) {
    spdlog::info("Collecting {} [{}, {}]", symbol, format_date(start), format_date(end));
    const std::string run_id = provider_->name() + "_collect_" + symbol + "_" +
        std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());

    // 1. Fetch from provider
    auto raw_bars = provider_->fetch_daily(symbol, start, end);
    if (raw_bars.empty()) {
        spdlog::warn("No data returned for {}", symbol);
        return QualityReport{};
    }

    // 2. Build silver bars from raw bars
    auto bars = BarNormalizer::normalize(raw_bars);

    // 3. Compute price limits for silver layer
    Board board = (!bars.empty() && bars[0].board != Board::kMain)
        ? bars[0].board : Board::kMain;
    BarNormalizer::compute_limits(bars, board);

    // 4. Merge northbound data by date (optional) into silver layer
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

    // 5. Merge margin data by date (optional) into silver layer
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

    // 6. Validate silver window
    auto report = DataValidator::validate(bars);
    if (!report.is_clean()) {
        for (const auto& w : report.warnings) {
            spdlog::warn("{}: {}", symbol, w);
        }
    }

    // 7. Partition by year and write silver (+ optional raw)
    std::map<int, std::vector<Bar>> silver_by_year;
    for (const auto& bar : bars) {
        silver_by_year[date_year(bar.date)].push_back(bar);
    }

    if (config_.ingestion.write_raw_layer) {
        std::map<int, std::vector<Bar>> raw_by_year;
        for (const auto& bar : raw_bars) {
            raw_by_year[date_year(bar.date)].push_back(bar);
        }

        for (auto& [year, year_raw] : raw_by_year) {
            auto raw_path = paths_.raw_daily(symbol, year);
            Date max_date = start;
            for (const auto& b : year_raw) {
                if (b.date > max_date) max_date = b.date;
            }
            merge_and_write_raw_bars(raw_path, std::move(year_raw), max_date);
        }
    }

    for (auto& [year, year_silver] : silver_by_year) {
        auto silver_path = paths_.silver_daily(symbol, year);
        Date max_date = start;
        for (const auto& b : year_silver) {
            if (b.date > max_date) max_date = b.date;
        }
        merge_and_write_silver_bars(silver_path, std::move(year_silver), board, max_date);
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
    metadata_.upsert_watermark(provider_->name(),
                               config_.ingestion.daily_bar_dataset,
                               symbol,
                               bars.back().date);
    metadata_.upsert_stream_checkpoint(provider_->name(),
                                       config_.ingestion.daily_bar_dataset,
                                       symbol,
                                       "{}",
                                       bars.back().date);
    record_market_quality(metadata_, run_id, symbol, report, bars.back().date);

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

        Date start;
        const int lookback_days = std::max(0, config_.ingestion.incremental_lookback_days);
        auto wm = metadata_.last_watermark_date(provider_->name(),
                                                config_.ingestion.daily_bar_dataset,
                                                symbols[i]);
        if (wm) {
            start = *wm - std::chrono::days{lookback_days};
            Date floor = parse_date(config_.ingestion.min_start_date);
            if (start < floor) start = floor;
        } else {
            auto last = metadata_.last_download_date(symbols[i]);
            start = last ? next_trading_day(*last) : parse_date(config_.ingestion.min_start_date);
        }
        collect_symbol(symbols[i], start, today);
    }
}

} // namespace trade
