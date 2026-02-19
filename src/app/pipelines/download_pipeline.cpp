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
    const std::string& dataset = config.ingestion.daily_bar_dataset;

    auto now_tp = std::chrono::system_clock::now();
    auto now_day = std::chrono::floor<std::chrono::days>(now_tp);
    int history_days = std::max(1, config.ingestion.default_history_days);
    auto default_start = now_day - std::chrono::days{history_days};
    auto min_start = parse_date(config.ingestion.min_start_date);
    if (default_start < min_start) default_start = min_start;
    std::string default_start_str = format_date(default_start);
    auto symbols = parse_symbol_list(request.symbol);
    const bool full_refresh = request.refresh;      // --refresh
    const bool incremental_mode = !full_refresh;    // default mode without --refresh
    const bool stream_resume_mode = request.use_checkpoint;

    // Incremental update contract:
    // 1) Must provide explicit symbol list.
    // 2) Must provide bootstrap start date.
    if (incremental_mode) {
        if (symbols.empty()) {
            spdlog::error("Incremental mode requires --symbol list (comma-separated).");
            return 1;
        }
        if (!request.start.has_value()) {
            spdlog::error("Incremental mode requires --start (bootstrap boundary).");
            return 1;
        }
    }

    if (request.start && request.end && *request.start > *request.end) {
        spdlog::error("Invalid date range: --start > --end");
        return 1;
    }

    if (symbols.empty()) {
        // Keep full-refresh all-symbol backfill behavior.
        std::string run_id = request.provider + "_dl_all_" +
            std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                now_tp.time_since_epoch()).count());
        metadata.begin_ingestion_run(run_id, request.provider, dataset, "*", "full");
        try {
            auto [start_all, end_all] = resolve_request_dates(request, default_start_str);
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

    int success_symbols = 0;
    int skipped_symbols = 0;
    for (const auto& symbol : symbols) {
        Date end = request.end.value_or(now_day);
        Date start = request.start.value_or(default_start);
        std::optional<Date> current_wm =
            metadata.last_watermark_date(request.provider, dataset, symbol);
        std::optional<MetadataStore::StreamCheckpointRecord> current_cp;
        bool has_stream_checkpoint = false;
        if (request.use_checkpoint) {
            current_cp = metadata.get_stream_checkpoint(request.provider, dataset, symbol);
            has_stream_checkpoint = current_cp.has_value() &&
                current_cp->last_event_date.has_value() &&
                current_cp->cursor_payload.find("\"mode\":\"stream\"") != std::string::npos;
        }

        if (incremental_mode) {
            const int lookback_days = std::max(0, config.ingestion.incremental_lookback_days);
            Date bootstrap_start = *request.start;
            if (bootstrap_start < min_start) bootstrap_start = min_start;

            if (request.use_checkpoint && has_stream_checkpoint) {
                start = next_trading_day(*current_cp->last_event_date);
                if (start < bootstrap_start) start = bootstrap_start;
                if (start < min_start) start = min_start;
                spdlog::info("Incremental {} from checkpoint {} (stream={}, shard={})",
                             symbol,
                             format_date(*current_cp->last_event_date),
                             current_cp->stream,
                             current_cp->shard);
            } else if (current_wm) {
                start = *current_wm - std::chrono::days{lookback_days};
                if (start < bootstrap_start) start = bootstrap_start;
                if (start < min_start) start = min_start;
                spdlog::info("Incremental {} from watermark {} (lookback {}d => {}, bootstrap {})",
                             symbol,
                             format_date(*current_wm),
                             lookback_days,
                             format_date(start),
                             format_date(bootstrap_start));
            } else {
                start = bootstrap_start;
                spdlog::info("Incremental {} bootstrap from start {} (no watermark)",
                             symbol,
                             format_date(start));
            }

            if (start > end) {
                std::cout << "Already up to date for " << symbol
                          << " (last target: " << format_date(end) << ")" << std::endl;
                ++skipped_symbols;
                continue;
            }
        } else {
            if (start < min_start) start = min_start;
        }

        const bool cloud_mode = config.storage.enabled &&
            (config.storage.backend == "baidu_netdisk" || config.storage.backend == "baidu");
        bool has_local_raw_partition = false;
        for (int y = date_year(start); y <= date_year(end); ++y) {
            if (std::filesystem::exists(paths.raw_daily(symbol, y))) {
                has_local_raw_partition = true;
                break;
            }
        }
        const int dedup_hours = std::max(0, config.ingestion.request_dedup_hours);
        const bool explicit_window_request = request.start.has_value() || request.end.has_value();
        const bool watermark_covers_end = current_wm.has_value() && (*current_wm >= end);
        const bool checkpoint_covers_end = has_stream_checkpoint &&
            (*current_cp->last_event_date >= end);
        const bool dedup_eligible = explicit_window_request || watermark_covers_end || checkpoint_covers_end;
        if (incremental_mode && !stream_resume_mode && dedup_hours > 0 &&
            dedup_eligible &&
            (has_local_raw_partition || cloud_mode) &&
            metadata.has_recent_successful_request(request.provider,
                                                   dataset,
                                                   symbol,
                                                   start,
                                                   end,
                                                   dedup_hours)) {
            std::cout << "Skip duplicate request for " << symbol
                      << " [" << format_date(start) << ", " << format_date(end)
                      << "] within " << dedup_hours << "h window." << std::endl;
            ++skipped_symbols;
            continue;
        }

        std::string run_id = request.provider + "_dl_" + symbol + "_" +
            std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        std::string mode = full_refresh ? "full" : (stream_resume_mode ? "stream_incremental" : "incremental");
        metadata.begin_ingestion_run(run_id, request.provider, dataset, symbol, mode);

        try {
            auto report = collector.collect_symbol(symbol, start, end);
            metadata.finish_ingestion_run(run_id, true,
                                          static_cast<int64_t>(report.total_bars),
                                          static_cast<int64_t>(report.valid_bars));
            metadata.record_request_fingerprint(request.provider,
                                                dataset,
                                                symbol,
                                                start,
                                                end,
                                                "success",
                                                run_id,
                                                static_cast<int64_t>(report.total_bars));
            if (request.use_checkpoint) {
                auto committed_wm = metadata.last_watermark_date(request.provider, dataset, symbol);
                if (committed_wm.has_value()) {
                    metadata.upsert_stream_checkpoint(request.provider,
                                                      dataset,
                                                      symbol,
                                                      build_stream_checkpoint_payload(*committed_wm),
                                                      *committed_wm);
                }
            }

            std::cout << "Downloaded " << report.total_bars << " bars for " << symbol
                      << " (quality: " << std::fixed << std::setprecision(1)
                      << (report.quality_score() * 100) << "%)" << std::endl;
            ++success_symbols;
        } catch (const std::exception& e) {
            metadata.finish_ingestion_run(run_id, false, 0, 0, e.what());
            metadata.record_request_fingerprint(request.provider,
                                                dataset,
                                                symbol,
                                                start,
                                                end,
                                                "failed",
                                                run_id,
                                                0);
            throw;
        }
    }

    if (symbols.size() > 1) {
        std::cout << "Download summary: success=" << success_symbols
                  << ", skipped=" << skipped_symbols
                  << ", total=" << symbols.size() << std::endl;
    }

    return 0;
}

} // namespace trade::app
