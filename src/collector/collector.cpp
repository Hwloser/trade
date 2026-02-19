#include "trade/collector/collector.h"
#include "trade/common/time_utils.h"
#include "trade/storage/parquet_reader.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <spdlog/spdlog.h>
#include <unordered_map>

namespace trade {

namespace {

constexpr double kDoubleEps = 1e-10;

bool nearly_equal(double a, double b) {
    return std::fabs(a - b) <= kDoubleEps;
}

bool optional_nearly_equal(const std::optional<double>& a,
                           const std::optional<double>& b) {
    if (!a && !b) return true;
    if (a && b) return nearly_equal(*a, *b);
    if (a && !b) return nearly_equal(*a, 0.0);
    if (!a && b) return nearly_equal(*b, 0.0);
    return false;
}

bool bar_equal(const Bar& a, const Bar& b) {
    return a.symbol == b.symbol &&
           a.date == b.date &&
           nearly_equal(a.open, b.open) &&
           nearly_equal(a.high, b.high) &&
           nearly_equal(a.low, b.low) &&
           nearly_equal(a.close, b.close) &&
           a.volume == b.volume &&
           nearly_equal(a.amount, b.amount) &&
           nearly_equal(a.turnover_rate, b.turnover_rate) &&
           nearly_equal(a.prev_close, b.prev_close) &&
           nearly_equal(a.vwap, b.vwap) &&
           nearly_equal(a.limit_up, b.limit_up) &&
           nearly_equal(a.limit_down, b.limit_down) &&
           a.hit_limit_up == b.hit_limit_up &&
           a.hit_limit_down == b.hit_limit_down &&
           a.bar_status == b.bar_status &&
           a.board == b.board &&
           optional_nearly_equal(a.north_net_buy, b.north_net_buy) &&
           optional_nearly_equal(a.margin_balance, b.margin_balance) &&
           optional_nearly_equal(a.short_sell_volume, b.short_sell_volume);
}

void sort_by_date(std::vector<Bar>* bars) {
    std::sort(bars->begin(), bars->end(),
              [](const Bar& a, const Bar& b) { return a.date < b.date; });
}

bool bars_equal(std::vector<Bar> lhs, std::vector<Bar> rhs) {
    if (lhs.size() != rhs.size()) return false;
    sort_by_date(&lhs);
    sort_by_date(&rhs);
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (!bar_equal(lhs[i], rhs[i])) return false;
    }
    return true;
}

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

bool merge_and_write_raw_bars(const std::string& path,
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

    auto existing = merged;
    merged = merge_by_date(std::move(merged), std::move(incoming));
    if (bars_equal(existing, merged)) {
        return false;
    }

    // Raw layer keeps provider fields close to source semantics.
    ParquetWriter::write_bars(path, merged, ParquetWriter::MergeMode::kReplace, partition_max_date);
    return true;
}

bool merge_and_write_silver_bars(const std::string& path,
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

    auto existing = merged;
    merged = merge_by_date(std::move(merged), std::move(incoming));
    merged = BarNormalizer::normalize(std::move(merged));
    if (!merged.empty()) {
        Board board = merged.front().board != Board::kMain ? merged.front().board : default_board;
        BarNormalizer::compute_limits(merged, board);
    }
    if (bars_equal(existing, merged)) {
        return false;
    }

    ParquetWriter::write_bars(path, merged, ParquetWriter::MergeMode::kReplace, partition_max_date);
    return true;
}

void record_market_quality(MetadataStore& metadata,
                           const std::string& run_id,
                           const std::string& dataset_id,
                           const Symbol& symbol,
                           const QualityReport& report,
                           std::optional<Date> event_date) {
    MetadataStore::QualityCheckRecord qc;
    qc.run_id = run_id;
    qc.dataset_id = dataset_id;
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

Symbol symbol_from_dataset_file(const std::string& file_path) {
    return std::filesystem::path(file_path).stem().string();
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
    if (config_.ingestion.write_silver_layer) {
        spdlog::debug("write_silver_layer is ignored in collect; use build_silver_* from raw");
    }
    const std::string run_id = provider_->name() + "_collect_" + symbol + "_" +
        std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());

    // 1. Fetch from provider
    auto raw_bars = provider_->fetch_daily(symbol, start, end);
    if (raw_bars.empty()) {
        spdlog::warn("No data returned for {}", symbol);
        return QualityReport{};
    }

    // 2. Validate raw window.
    auto report = DataValidator::validate(raw_bars);
    if (!report.is_clean()) {
        for (const auto& w : report.warnings) {
            spdlog::warn("{}: {}", symbol, w);
        }
    }

    // 3. Partition by year and write raw only.
    if (!config_.ingestion.write_raw_layer) {
        spdlog::warn("write_raw_layer=false ignored: collect always writes raw");
    }
    bool raw_changed = false;
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
        raw_changed = merge_and_write_raw_bars(raw_path, std::move(year_raw), max_date) || raw_changed;
    }
    if (!raw_changed) {
        spdlog::info("{} raw layer unchanged (skip rewrite)", symbol);
    }

    // 8. Upsert instrument record
    auto inst_opt = provider_->fetch_instrument(symbol);
    if (inst_opt) {
        metadata_.upsert_instrument(*inst_opt);
    }

    // 9. Record download and advance watermark
    auto minmax_date = std::minmax_element(
        raw_bars.begin(), raw_bars.end(),
        [](const Bar& a, const Bar& b) { return a.date < b.date; });
    const Date batch_start = minmax_date.first->date;
    const Date batch_end = minmax_date.second->date;

    metadata_.record_download(symbol,
                              batch_start,
                              batch_end,
                              static_cast<int64_t>(raw_bars.size()));
    metadata_.upsert_watermark(provider_->name(),
                               config_.ingestion.daily_bar_dataset,
                               symbol,
                               batch_end);
    metadata_.upsert_stream_checkpoint(provider_->name(),
                                       config_.ingestion.daily_bar_dataset,
                                       symbol,
                                       "{}",
                                       batch_end);
    if (raw_changed) {
        record_market_quality(metadata_, run_id, "raw.cn_a.daily", symbol, report, batch_end);
    } else {
        spdlog::debug("{} no partition changed, skip quality check insert", symbol);
    }

    spdlog::info("Collected {} raw bars for {} (quality: {:.1f}%)",
                 raw_bars.size(),
                 symbol,
                 report.quality_score() * 100);
    return report;
}

QualityReport Collector::build_silver_symbol(const Symbol& symbol, Date start, Date end) {
    spdlog::info("Building silver {} [{}, {}] from raw layer",
                 symbol, format_date(start), format_date(end));

    std::vector<Bar> raw_bars;
    for (int year = date_year(start); year <= date_year(end); ++year) {
        auto raw_path = paths_.raw_daily(symbol, year);
        if (!std::filesystem::exists(raw_path)) continue;
        try {
            auto ybars = ParquetReader::read_bars(raw_path, start, end);
            raw_bars.insert(raw_bars.end(),
                            std::make_move_iterator(ybars.begin()),
                            std::make_move_iterator(ybars.end()));
        } catch (const std::exception& e) {
            spdlog::warn("Failed to read raw partition {}: {}", raw_path, e.what());
        }
    }
    if (raw_bars.empty()) {
        spdlog::warn("No raw data found for {} in [{} - {}]",
                     symbol, format_date(start), format_date(end));
        return QualityReport{};
    }

    auto silver_bars = BarNormalizer::normalize(raw_bars);
    Board board = (!silver_bars.empty() && silver_bars[0].board != Board::kMain)
        ? silver_bars[0].board : Board::kMain;
    BarNormalizer::compute_limits(silver_bars, board);

    if (provider_->supports_northbound() && !silver_bars.empty()) {
        std::set<Date> dates;
        for (const auto& bar : silver_bars) dates.insert(bar.date);
        for (const auto& d : dates) {
            auto cache_it = northbound_cache_.find(d);
            if (cache_it == northbound_cache_.end()) {
                northbound_cache_.emplace(d, provider_->fetch_northbound(d));
            }
        }
        for (auto& bar : silver_bars) {
            auto dit = northbound_cache_.find(bar.date);
            if (dit == northbound_cache_.end()) continue;
            auto sit = dit->second.find(bar.symbol);
            if (sit != dit->second.end()) bar.north_net_buy = sit->second;
        }
    }

    if (provider_->supports_margin() && !silver_bars.empty()) {
        std::set<Date> dates;
        for (const auto& bar : silver_bars) dates.insert(bar.date);
        for (const auto& d : dates) {
            auto cache_it = margin_cache_.find(d);
            if (cache_it == margin_cache_.end()) {
                margin_cache_.emplace(d, provider_->fetch_margin(d));
            }
        }
        for (auto& bar : silver_bars) {
            auto dit = margin_cache_.find(bar.date);
            if (dit == margin_cache_.end()) continue;
            auto sit = dit->second.find(bar.symbol);
            if (sit != dit->second.end()) bar.margin_balance = sit->second;
        }
    }

    auto report = DataValidator::validate(silver_bars);
    if (!report.is_clean()) {
        for (const auto& w : report.warnings) {
            spdlog::warn("{}: {}", symbol, w);
        }
    }

    std::map<int, std::vector<Bar>> silver_by_year;
    for (const auto& bar : silver_bars) {
        silver_by_year[date_year(bar.date)].push_back(bar);
    }

    bool silver_changed = false;
    for (auto& [year, year_silver] : silver_by_year) {
        auto silver_path = paths_.silver_daily(symbol, year);
        Date max_date = start;
        for (const auto& b : year_silver) {
            if (b.date > max_date) max_date = b.date;
        }
        silver_changed = merge_and_write_silver_bars(
            silver_path, std::move(year_silver), board, max_date) || silver_changed;
    }

    auto minmax_date = std::minmax_element(
        silver_bars.begin(), silver_bars.end(),
        [](const Bar& a, const Bar& b) { return a.date < b.date; });
    const Date batch_end = minmax_date.second->date;
    if (silver_changed) {
        const std::string run_id = provider_->name() + "_silver_" + symbol + "_" +
            std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        record_market_quality(metadata_, run_id, "silver.cn_a.daily", symbol, report, batch_end);
        spdlog::info("Built {} silver bars for {} (quality: {:.1f}%)",
                     silver_bars.size(), symbol, report.quality_score() * 100);
    } else {
        spdlog::info("{} silver layer unchanged (skip rewrite)", symbol);
    }
    return report;
}

void Collector::build_silver_all(Date start, Date end, ProgressCallback progress) {
    std::set<Symbol> symbols;
    for (const auto& f : metadata_.list_dataset_files("raw.cn_a.daily")) {
        auto sym = symbol_from_dataset_file(f.file_path);
        if (!sym.empty()) symbols.insert(sym);
    }

    if (symbols.empty()) {
        spdlog::warn("No raw.cn_a.daily files found in metadata; nothing to build for silver");
        return;
    }

    std::vector<Symbol> ordered(symbols.begin(), symbols.end());
    for (size_t i = 0; i < ordered.size(); ++i) {
        if (progress) {
            progress(ordered[i], static_cast<int>(i + 1), static_cast<int>(ordered.size()));
        }
        build_silver_symbol(ordered[i], start, end);
    }
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
