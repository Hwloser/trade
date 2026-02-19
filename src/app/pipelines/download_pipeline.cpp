#include "trade/app/pipelines/download_pipeline.h"

#include "trade/collector/collector.h"
#include "trade/common/time_utils.h"
#include "trade/provider/provider_factory.h"
#include "trade/storage/metadata_store.h"
#include "trade/storage/storage_path.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
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

    if (!request.symbol.empty()) {
        Date start;
        Date end = request.end.value_or(std::chrono::floor<std::chrono::days>(now_tp));
        if (!request.refresh && !request.start) {
            int lookback_days = std::max(0, config.ingestion.incremental_lookback_days);
            auto wm = metadata.last_watermark_date(request.provider, dataset, request.symbol);
            if (wm) {
                start = *wm - std::chrono::days{lookback_days};
                if (start < min_start) start = min_start;
                spdlog::info("Incremental from watermark {} (lookback {}d => {})",
                             format_date(*wm), lookback_days, format_date(start));
            } else {
                auto last = metadata.last_download_date(request.symbol);
                if (last) {
                    start = next_trading_day(*last);
                    if (start < min_start) start = min_start;
                    spdlog::info("Incremental from last download {} (start: {})",
                                 format_date(*last), format_date(start));
                } else {
                    start = default_start;
                }
            }

            if (start > end) {
                std::cout << "Already up to date (last target: "
                          << format_date(end) << ")" << std::endl;
                return 0;
            }
        } else {
            start = request.start.value_or(default_start);
        }

        const bool cloud_mode = config.storage.enabled &&
            (config.storage.backend == "baidu_netdisk" || config.storage.backend == "baidu");
        bool has_local_raw_partition = false;
        for (int y = date_year(start); y <= date_year(end); ++y) {
            if (std::filesystem::exists(paths.raw_daily(request.symbol, y))) {
                has_local_raw_partition = true;
                break;
            }
        }
        const int dedup_hours = std::max(0, config.ingestion.request_dedup_hours);
        if (!request.refresh && dedup_hours > 0 &&
            (has_local_raw_partition || cloud_mode) &&
            metadata.has_recent_successful_request(request.provider,
                                                   dataset,
                                                   request.symbol,
                                                   start,
                                                   end,
                                                   dedup_hours)) {
            std::cout << "Skip duplicate request for " << request.symbol
                      << " [" << format_date(start) << ", " << format_date(end)
                      << "] within " << dedup_hours << "h window." << std::endl;
            return 0;
        }

        std::string run_id = request.provider + "_dl_" + request.symbol + "_" +
            std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                now_tp.time_since_epoch()).count());
        std::string mode = request.refresh ? "full" : "incremental";
        metadata.begin_ingestion_run(run_id, request.provider, dataset, request.symbol, mode);

        try {
            auto report = collector.collect_symbol(request.symbol, start, end);
            metadata.finish_ingestion_run(run_id, true,
                                          static_cast<int64_t>(report.total_bars),
                                          static_cast<int64_t>(report.valid_bars));
            metadata.record_request_fingerprint(request.provider,
                                                dataset,
                                                request.symbol,
                                                start,
                                                end,
                                                "success",
                                                run_id,
                                                static_cast<int64_t>(report.total_bars));

            std::cout << "Downloaded " << report.total_bars << " bars for " << request.symbol
                      << " (quality: " << std::fixed << std::setprecision(1)
                      << (report.quality_score() * 100) << "%)" << std::endl;
        } catch (const std::exception& e) {
            metadata.finish_ingestion_run(run_id, false, 0, 0, e.what());
            metadata.record_request_fingerprint(request.provider,
                                                dataset,
                                                request.symbol,
                                                start,
                                                end,
                                                "failed",
                                                run_id,
                                                0);
            throw;
        }
    } else {
        std::string run_id = request.provider + "_dl_all_" +
            std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                now_tp.time_since_epoch()).count());
        std::string mode = request.refresh ? "full" : "incremental";
        metadata.begin_ingestion_run(run_id, request.provider, dataset, "*", mode);

        try {
            if (request.refresh) {
                auto [start_all, end_all] = resolve_request_dates(request, default_start_str);
                collector.collect_all(start_all, end_all,
                    [](const Symbol& sym, int cur, int total) {
                        std::cout << "\r[" << cur << "/" << total << "] " << sym
                                 << "                " << std::flush;
                    });
            } else {
                collector.update_all(
                    [](const Symbol& sym, int cur, int total) {
                        std::cout << "\r[" << cur << "/" << total << "] " << sym
                                 << "                " << std::flush;
                    });
            }
            metadata.finish_ingestion_run(run_id, true, 0, 0);
            std::cout << "\nDownload complete." << std::endl;
        } catch (const std::exception& e) {
            metadata.finish_ingestion_run(run_id, false, 0, 0, e.what());
            throw;
        }
    }

    return 0;
}

} // namespace trade::app
