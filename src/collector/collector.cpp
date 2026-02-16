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

    // 2. Normalize (sort, fill prev_close, compute vwap)
    bars = BarNormalizer::normalize(std::move(bars));

    // 3. Compute price limits (uses prev_close and board from bars)
    // Board is already set per-bar if provider fills it;
    // otherwise it stays at default kMain
    if (!bars.empty() && bars[0].board != Board::kMain) {
        BarNormalizer::compute_limits(bars, bars[0].board);
    } else {
        BarNormalizer::compute_limits(bars, Board::kMain);
    }

    // 4. Merge northbound data (optional)
    if (provider_->supports_northbound() && !bars.empty()) {
        auto last_date = bars.back().date;
        auto north = provider_->fetch_northbound(last_date);
        if (!north.empty()) {
            for (auto& bar : bars) {
                auto it = north.find(bar.symbol);
                if (it != north.end()) {
                    bar.north_net_buy = it->second;
                }
            }
        }
    }

    // 5. Merge margin data (optional)
    if (provider_->supports_margin() && !bars.empty()) {
        auto last_date = bars.back().date;
        auto margin = provider_->fetch_margin(last_date);
        if (!margin.empty()) {
            for (auto& bar : bars) {
                auto it = margin.find(bar.symbol);
                if (it != margin.end()) {
                    bar.margin_balance = it->second;
                }
            }
        }
    }

    // 6. Validate
    auto report = DataValidator::validate(bars);
    if (!report.is_clean()) {
        for (const auto& w : report.warnings) {
            spdlog::warn("{}: {}", symbol, w);
        }
    }

    // 7. Store raw (full 20-column schema)
    int year = date_year(start);
    auto raw_path = paths_.raw_daily(symbol, year);
    StoragePath::ensure_dir(raw_path);
    ParquetWriter::write_bars(raw_path, bars);

    // 8. Store curated
    auto curated_path = paths_.curated_daily(symbol, year);
    StoragePath::ensure_dir(curated_path);
    ParquetWriter::write_bars(curated_path, bars);

    // 9. Upsert instrument record
    auto inst_opt = provider_->fetch_instrument(symbol);
    if (inst_opt) {
        metadata_.upsert_instrument(*inst_opt);
    }

    // 10. Record download
    metadata_.record_download(symbol, start, end, static_cast<int64_t>(bars.size()));

    spdlog::info("Collected {} bars for {} (quality: {:.1f}%)",
                 bars.size(), symbol, report.quality_score() * 100);
    return report;
}

void Collector::collect_all(Date start, Date end, ProgressCallback progress) {
    auto instruments = provider_->fetch_instruments();
    spdlog::info("Found {} instruments, collecting [{}, {}]",
                 instruments.size(), format_date(start), format_date(end));

    // Enhance instruments with industry/share capital if provider supports
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

        // Fill industry from provider
        if (!industry_map.empty()) {
            auto it = industry_map.find(inst.symbol);
            if (it != industry_map.end()) {
                inst.industry = it->second;
            }
        }

        // Fill share capital from provider
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
