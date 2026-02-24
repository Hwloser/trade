#include "trade/app/pipelines/download_pipeline.h"

#include "trade/collector/collector.h"
#include "trade/common/time_utils.h"
#include "trade/provider/provider_factory.h"
#include "trade/storage/metadata_store.h"
#include "trade/storage/storage_path.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <unordered_set>
#include <spdlog/spdlog.h>

namespace trade::app {
namespace {

std::pair<Date, Date> resolve_request_dates(const DownloadRequest& request,
                                            const std::string& default_start) {
    auto start = request.start.value_or(parse_date(default_start));
    auto end = request.end.value_or(
        std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now()));
    return {start, end};
}

std::string trim_copy(std::string s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

std::vector<Symbol> parse_symbol_list(const std::string& raw) {
    std::vector<Symbol> out;
    std::unordered_set<std::string> seen;
    std::string token;

    auto flush = [&]() {
        auto sym = trim_copy(token);
        token.clear();
        if (sym.empty()) return;
        if (seen.insert(sym).second) {
            out.push_back(sym);
        }
    };

    for (char ch : raw) {
        if (ch == ',' || ch == ';') {
            flush();
        } else {
            token.push_back(ch);
        }
    }
    flush();
    return out;
}

struct DownloadRuntime {
    const DownloadRequest& request;
    const Config& config;
    const StoragePath& paths;
    const std::string& dataset;
    Date now_day;
    Date default_start;
    Date min_start;
    std::string default_start_str;
    bool full_refresh;
    bool incremental_mode;
};

struct SymbolPlan {
    Date start;
    Date end;
    std::optional<Date> current_wm;
    bool up_to_date = false;
};

struct SymbolRunResult {
    bool success = false;
    bool skipped = false;
};

DownloadRuntime make_runtime(const DownloadRequest& request,
                             const Config& config,
                             const StoragePath& paths) {
    auto now_tp = std::chrono::system_clock::now();
    Date now_day = std::chrono::floor<std::chrono::days>(now_tp);
    int history_days = std::max(1, config.ingestion.default_history_days);
    Date default_start = now_day - std::chrono::days{history_days};
    Date min_start = parse_date(config.ingestion.min_start_date);
    if (default_start < min_start) default_start = min_start;

    return DownloadRuntime{
        request,
        config,
        paths,
        config.ingestion.daily_bar_dataset,
        now_day,
        default_start,
        min_start,
        format_date(default_start),
        request.refresh,
        !request.refresh,
    };
}

bool validate_request(const DownloadRuntime& rt,
                      const std::vector<Symbol>& symbols) {
    // Incremental update contract:
    // 1) Must provide explicit symbol list.
    // 2) Must provide bootstrap start date.
    if (rt.incremental_mode) {
        if (symbols.empty()) {
            spdlog::error("Incremental mode requires --symbol list (comma-separated).");
            return false;
        }
        if (!rt.request.start.has_value()) {
            spdlog::error("Incremental mode requires --start (bootstrap boundary).");
            return false;
        }
    }

    if (rt.request.start && rt.request.end && *rt.request.start > *rt.request.end) {
        spdlog::error("Invalid date range: --start > --end");
        return false;
    }
    return true;
}

int run_full_refresh_all_symbols(const DownloadRuntime& rt,
                                 Collector& collector,
                                 MetadataStore& /*metadata*/) {
    auto [start_all, end_all] = resolve_request_dates(rt.request, rt.default_start_str);
    collector.collect_all(start_all, end_all,
        [](const Symbol& sym, int cur, int total) {
            std::cout << "\r[" << cur << "/" << total << "] " << sym
                      << "                " << std::flush;
        });
    std::cout << "\nDownload complete." << std::endl;
    return 0;
}

SymbolPlan plan_symbol_download(const DownloadRuntime& rt,
                                MetadataStore& metadata,
                                const Symbol& symbol) {
    SymbolPlan plan;
    plan.end = rt.request.end.value_or(rt.now_day);
    plan.start = rt.request.start.value_or(rt.default_start);
    plan.current_wm = metadata.last_watermark_date(rt.request.provider, rt.dataset, symbol);

    if (rt.incremental_mode) {
        const int lookback_days = std::max(0, rt.config.ingestion.incremental_lookback_days);
        Date bootstrap_start = *rt.request.start;
        if (bootstrap_start < rt.min_start) bootstrap_start = rt.min_start;

        if (plan.current_wm) {
            plan.start = *plan.current_wm - std::chrono::days{lookback_days};
            if (plan.start < bootstrap_start) plan.start = bootstrap_start;
            if (plan.start < rt.min_start) plan.start = rt.min_start;
        } else {
            plan.start = bootstrap_start;
        }
        plan.up_to_date = plan.start > plan.end;
    } else if (plan.start < rt.min_start) {
        plan.start = rt.min_start;
    }

    return plan;
}

bool has_local_raw_partition(const StoragePath& paths,
                             const Symbol& symbol,
                             Date start,
                             Date end) {
    for (int y = date_year(start); y <= date_year(end); ++y) {
        for (int m = 1; m <= 12; ++m) {
            if (std::filesystem::exists(paths.kline_monthly(symbol, y, m))) {
                return true;
            }
        }
    }
    return false;
}

SymbolRunResult run_symbol_download(const DownloadRuntime& rt,
                                    Collector& collector,
                                    MetadataStore& /*metadata*/,
                                    const Symbol& symbol,
                                    const SymbolPlan& plan) {
    SymbolRunResult result;
    try {
        auto report = collector.collect_symbol(symbol, plan.start, plan.end);
        std::cout << "Downloaded " << report.total_bars << " bars for " << symbol
                  << " (quality: " << std::fixed << std::setprecision(1)
                  << (report.quality_score() * 100) << "%)" << std::endl;
        result.success = true;
    } catch (const std::exception& e) {
        spdlog::error("Failed to download {}: {}", symbol, e.what());
    }
    return result;
}

} // namespace

int run_download(const DownloadRequest& request, const Config& config) {
    auto provider = ProviderFactory::create(request.provider, config);
    if (!provider->ping()) {
        spdlog::error("Cannot connect to {} provider", request.provider);
        return 1;
    }
    Collector collector(std::move(provider), config);

    StoragePath paths(config.data.data_root);
    MetadataStore metadata(paths.metadata_db());
    auto runtime = make_runtime(request, config, paths);
    auto symbols = parse_symbol_list(request.symbol);

    if (!validate_request(runtime, symbols)) {
        return 1;
    }
    if (symbols.empty()) {
        return run_full_refresh_all_symbols(runtime, collector, metadata);
    }

    int success_symbols = 0;
    int skipped_symbols = 0;
    for (const auto& symbol : symbols) {
        auto plan = plan_symbol_download(runtime, metadata, symbol);
        if (plan.up_to_date) {
            std::cout << "Already up to date for " << symbol
                      << " (last target: " << format_date(plan.end) << ")" << std::endl;
            ++skipped_symbols;
            continue;
        }
        auto result = run_symbol_download(runtime, collector, metadata, symbol, plan);
        if (result.success) ++success_symbols;
        if (result.skipped) ++skipped_symbols;
    }

    if (symbols.size() > 1) {
        std::cout << "Download summary: success=" << success_symbols
                  << ", skipped=" << skipped_symbols
                  << ", total=" << symbols.size() << std::endl;
    }

    return 0;
}

} // namespace trade::app
