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

std::string build_stream_checkpoint_payload(Date resume_from) {
    return std::string{"{\"mode\":\"stream\",\"resume_from\":\""} +
        format_date(resume_from) + "\"}";
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
    bool stream_resume_mode;
};

struct SymbolPlan {
    Date start;
    Date end;
    std::optional<Date> current_wm;
    std::optional<MetadataStore::StreamCheckpointRecord> current_cp;
    bool has_stream_checkpoint = false;
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
        request.use_checkpoint,
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
                                 MetadataStore& metadata) {
    // Keep full-refresh all-symbol backfill behavior.
    std::string run_id = rt.request.provider + "_dl_all_" +
        std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    metadata.begin_ingestion_run(run_id, rt.request.provider, rt.dataset, "*", "full");
    try {
        auto [start_all, end_all] = resolve_request_dates(rt.request, rt.default_start_str);
        collector.collect_all(start_all, end_all,
            [](const Symbol& sym, int cur, int total) {
                std::cout << "\r[" << cur << "/" << total << "] " << sym
                          << "                " << std::flush;
            });
        metadata.finish_ingestion_run(run_id, true, 0, 0);
        std::cout << "\nDownload complete." << std::endl;
    } catch (const std::exception& e) {
        metadata.finish_ingestion_run(run_id, false, 0, 0, e.what());
        throw;
    }
    return 0;
}

SymbolPlan plan_symbol_download(const DownloadRuntime& rt,
                                MetadataStore& metadata,
                                const Symbol& symbol) {
    SymbolPlan plan;
    plan.end = rt.request.end.value_or(rt.now_day);
    plan.start = rt.request.start.value_or(rt.default_start);
    plan.current_wm = metadata.last_watermark_date(rt.request.provider, rt.dataset, symbol);

    if (rt.stream_resume_mode) {
        plan.current_cp = metadata.get_stream_checkpoint(rt.request.provider, rt.dataset, symbol);
        plan.has_stream_checkpoint = plan.current_cp.has_value() &&
            plan.current_cp->last_event_date.has_value() &&
            plan.current_cp->cursor_payload.find("\"mode\":\"stream\"") != std::string::npos;
    }

    if (rt.incremental_mode) {
        const int lookback_days = std::max(0, rt.config.ingestion.incremental_lookback_days);
        Date bootstrap_start = *rt.request.start;
        if (bootstrap_start < rt.min_start) bootstrap_start = rt.min_start;

        if (rt.stream_resume_mode && plan.has_stream_checkpoint) {
            plan.start = next_trading_day(*plan.current_cp->last_event_date);
            if (plan.start < bootstrap_start) plan.start = bootstrap_start;
            if (plan.start < rt.min_start) plan.start = rt.min_start;
            spdlog::info("Incremental {} from checkpoint {} (stream={}, shard={})",
                         symbol,
                         format_date(*plan.current_cp->last_event_date),
                         plan.current_cp->stream,
                         plan.current_cp->shard);
        } else if (plan.current_wm) {
            plan.start = *plan.current_wm - std::chrono::days{lookback_days};
            if (plan.start < bootstrap_start) plan.start = bootstrap_start;
            if (plan.start < rt.min_start) plan.start = rt.min_start;
            spdlog::info("Incremental {} from watermark {} (lookback {}d => {}, bootstrap {})",
                         symbol,
                         format_date(*plan.current_wm),
                         lookback_days,
                         format_date(plan.start),
                         format_date(bootstrap_start));
        } else {
            plan.start = bootstrap_start;
            spdlog::info("Incremental {} bootstrap from start {} (no watermark)",
                         symbol,
                         format_date(plan.start));
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
        if (std::filesystem::exists(paths.raw_daily(symbol, y))) {
            return true;
        }
    }
    return false;
}

std::string run_mode(const DownloadRuntime& rt) {
    if (rt.full_refresh) return "full";
    if (rt.stream_resume_mode) return "stream_incremental";
    return "incremental";
}

void update_stream_checkpoint(const DownloadRuntime& rt,
                              MetadataStore& metadata,
                              const Symbol& symbol) {
    if (!rt.stream_resume_mode) return;
    auto committed_wm = metadata.last_watermark_date(rt.request.provider, rt.dataset, symbol);
    if (!committed_wm.has_value()) return;

    metadata.upsert_stream_checkpoint(rt.request.provider,
                                      rt.dataset,
                                      symbol,
                                      build_stream_checkpoint_payload(*committed_wm),
                                      *committed_wm);
}

SymbolRunResult run_symbol_download(const DownloadRuntime& rt,
                                    Collector& collector,
                                    MetadataStore& metadata,
                                    const Symbol& symbol,
                                    const SymbolPlan& plan) {
    SymbolRunResult result;
    const bool cloud_mode = rt.config.storage.enabled &&
        (rt.config.storage.backend == "baidu_netdisk" || rt.config.storage.backend == "baidu");
    const bool local_raw_exists = has_local_raw_partition(rt.paths, symbol, plan.start, plan.end);
    const int dedup_hours = std::max(0, rt.config.ingestion.request_dedup_hours);
    const bool explicit_window_request = rt.request.start.has_value() || rt.request.end.has_value();
    const bool watermark_covers_end = plan.current_wm.has_value() && (*plan.current_wm >= plan.end);
    const bool checkpoint_covers_end = plan.has_stream_checkpoint &&
        (*plan.current_cp->last_event_date >= plan.end);
    const bool dedup_eligible = explicit_window_request || watermark_covers_end || checkpoint_covers_end;

    if (rt.incremental_mode && !rt.stream_resume_mode && dedup_hours > 0 &&
        dedup_eligible && (local_raw_exists || cloud_mode) &&
        metadata.has_recent_successful_request(rt.request.provider,
                                               rt.dataset,
                                               symbol,
                                               plan.start,
                                               plan.end,
                                               dedup_hours)) {
        std::cout << "Skip duplicate request for " << symbol
                  << " [" << format_date(plan.start) << ", " << format_date(plan.end)
                  << "] within " << dedup_hours << "h window." << std::endl;
        result.skipped = true;
        return result;
    }

    std::string run_id = rt.request.provider + "_dl_" + symbol + "_" +
        std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    metadata.begin_ingestion_run(run_id, rt.request.provider, rt.dataset, symbol, run_mode(rt));

    try {
        auto report = collector.collect_symbol(symbol, plan.start, plan.end);
        metadata.finish_ingestion_run(run_id, true,
                                      static_cast<int64_t>(report.total_bars),
                                      static_cast<int64_t>(report.valid_bars));
        metadata.record_request_fingerprint(rt.request.provider,
                                            rt.dataset,
                                            symbol,
                                            plan.start,
                                            plan.end,
                                            "success",
                                            run_id,
                                            static_cast<int64_t>(report.total_bars));
        update_stream_checkpoint(rt, metadata, symbol);

        std::cout << "Downloaded " << report.total_bars << " bars for " << symbol
                  << " (quality: " << std::fixed << std::setprecision(1)
                  << (report.quality_score() * 100) << "%)" << std::endl;
        result.success = true;
    } catch (const std::exception& e) {
        metadata.finish_ingestion_run(run_id, false, 0, 0, e.what());
        metadata.record_request_fingerprint(rt.request.provider,
                                            rt.dataset,
                                            symbol,
                                            plan.start,
                                            plan.end,
                                            "failed",
                                            run_id,
                                            0);
        throw;
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
