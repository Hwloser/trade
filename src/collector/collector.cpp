#include "trade/collector/collector.h"
#include "trade/common/time_utils.h"
#include <spdlog/spdlog.h>

namespace trade {

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

    // 2. Normalize
    bars = BarNormalizer::normalize(std::move(bars));

    // 3. Validate
    auto report = DataValidator::validate(bars);
    if (!report.is_clean()) {
        for (const auto& w : report.warnings) {
            spdlog::warn("{}: {}", symbol, w);
        }
    }

    // 4. Store raw
    int year = date_year(start);
    auto raw_path = paths_.raw_daily(symbol, year);
    StoragePath::ensure_dir(raw_path);
    ParquetWriter::write_bars(raw_path, bars);

    // 5. Store curated (same as raw for now, will add more processing later)
    auto curated_path = paths_.curated_daily(symbol, year);
    StoragePath::ensure_dir(curated_path);
    ParquetWriter::write_bars(curated_path, bars);

    // 6. Record in metadata
    metadata_.record_download(symbol, start, end, static_cast<int64_t>(bars.size()));

    spdlog::info("Collected {} bars for {} (quality: {:.1f}%)",
                 bars.size(), symbol, report.quality_score() * 100);
    return report;
}

void Collector::collect_all(Date start, Date end, ProgressCallback progress) {
    auto instruments = provider_->fetch_instruments();
    spdlog::info("Found {} instruments, collecting [{}, {}]",
                 instruments.size(), format_date(start), format_date(end));

    for (size_t i = 0; i < instruments.size(); ++i) {
        const auto& inst = instruments[i];
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
