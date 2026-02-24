#include "trade/cli/commands.h"

#include "trade/app/pipelines/download_pipeline.h"
#include "trade/app/pipelines/sentiment_pipeline.h"
#include "trade/cli/shared.h"
#include "trade/common/time_utils.h"
#include "trade/collector/collector.h"
#include "trade/storage/google_drive_sync.h"
#include "trade/storage/metadata_store.h"
#include "trade/storage/parquet_reader.h"
#include "trade/storage/parquet_writer.h"
#include "trade/storage/storage_path.h"
#include "trade/validator/data_validator.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <spdlog/spdlog.h>

namespace trade::cli {
namespace {

bool cloud_mode_enabled(const trade::Config& config) {
    return config.storage.enabled && config.storage.backend == "google_drive";
}

int days_old(Date d) {
    auto now = std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now());
    return static_cast<int>(std::chrono::duration_cast<std::chrono::days>(now - d).count());
}


bool is_valid_raw_bar(const Bar& b) {
    if (b.open <= 0 || b.high <= 0 || b.low <= 0 || b.close <= 0) return false;
    if (b.high < b.low) return false;
    if (b.high < b.open || b.high < b.close) return false;
    if (b.low > b.open || b.low > b.close) return false;
    if (b.volume < 0 || b.amount < 0) return false;
    if (b.volume > 0 && b.amount <= 0) return false;
    if (b.volume == 0 && b.amount > 0) return false;
    return true;
}

std::vector<Bar> sanitize_raw_bars(const std::vector<Bar>& bars,
                                   int64_t* invalid_dropped,
                                   int64_t* duplicate_dropped) {
    int64_t bad = 0;
    std::map<Date, Bar> by_date;
    int64_t valid_rows = 0;
    for (const auto& b : bars) {
        if (!is_valid_raw_bar(b)) {
            ++bad;
            continue;
        }
        ++valid_rows;
        by_date[b.date] = b;  // latest wins on duplicate dates
    }

    std::vector<Bar> cleaned;
    cleaned.reserve(by_date.size());
    for (const auto& [_, b] : by_date) {
        cleaned.push_back(b);
    }

    if (invalid_dropped) *invalid_dropped = bad;
    if (duplicate_dropped) *duplicate_dropped = std::max<int64_t>(0, valid_rows - cleaned.size());
    return cleaned;
}

std::string symbol_from_file_path(const std::string& rel_path) {
    return std::filesystem::path(rel_path).stem().string();
}

class RawOnlySilverProvider final : public IDataProvider {
public:
    std::string name() const override { return "raw_to_silver"; }

    std::vector<Bar> fetch_daily(const Symbol& /*symbol*/,
                                 Date /*start*/,
                                 Date /*end*/) override {
        throw std::runtime_error("RawOnlySilverProvider does not support fetch_daily");
    }

    std::vector<Instrument> fetch_instruments() override {
        return {};
    }

    bool ping() override { return true; }
};

std::atomic<bool> g_stream_stop{false};

void on_stream_signal(int) {
    g_stream_stop.store(true);
}


} // namespace

int cmd_verify(const CliArgs& args, const trade::Config& config) {
    bool ok_local = false;
    bool ok_cloud = false;
    bool ok_meta = false;
    bool ok_sql = false;

    trade::StoragePath paths(config.data.data_root);
    trade::MetadataStore metadata(paths.metadata_db());

    std::cout << "=== Verify Data Pipeline ===\n";

    // 1) Local check: count instruments
    auto instruments = metadata.get_all_instruments();
    ok_local = !instruments.empty() || !args.symbol.empty();
    if (!args.symbol.empty()) {
        auto bars = load_bars(args.symbol, config);
        ok_local = !bars.empty();
        std::cout << "[Local] symbol=" << args.symbol
                  << " rows=" << bars.size()
                  << " -> " << (ok_local ? "OK" : "FAIL") << "\n";
    } else {
        std::cout << "[Local] instruments=" << instruments.size()
                  << " -> " << (ok_local ? "OK" : "FAIL") << "\n";
    }

    // 2) Cloud check (optional)
    const bool cloud_mode = config.storage.enabled &&
        config.storage.backend == "google_drive";
    if (cloud_mode) {
        trade::GoogleDriveSync client({
            .service_account_json_path = config.storage.google_drive_key_file,
            .root_folder_id = config.storage.google_drive_folder_id,
            .timeout_ms = config.storage.google_drive_timeout_ms,
            .retry_count = config.storage.google_drive_retry_count,
        });
        auto ts = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        std::string probe = "_health/verify_" + std::to_string(ts) + ".txt";
        const std::string msg = "trade-cloud-verify";
        std::vector<uint8_t> up(msg.begin(), msg.end());
        std::vector<uint8_t> down;
        bool uploaded = client.upload_bytes(probe, up);
        bool downloaded = uploaded && client.download_bytes(probe, &down);
        ok_cloud = downloaded && (std::string(down.begin(), down.end()) == msg);
        std::cout << "[Cloud] probe=" << probe
                  << " uploaded=" << (uploaded ? "yes" : "no")
                  << " downloaded=" << (downloaded ? "yes" : "no")
                  << " -> " << (ok_cloud ? "OK" : "FAIL") << "\n";
    } else {
        ok_cloud = true;
        std::cout << "[Cloud] skipped (storage backend is local/disabled)\n";
    }

    // 3) Metadata check
    auto mh = assess_metadata_health(metadata);
    ok_meta = mh.ok;
    std::cout << "[Meta] instruments=" << mh.instrument_count
              << " -> " << (ok_meta ? "OK" : "FAIL") << "\n";

    // 4) SQL check
    if (std::system("which duckdb > /dev/null 2>&1") != 0) {
        std::cout << "[SQL] duckdb not found -> FAIL\n";
        ok_sql = false;
    } else {
        auto views = discover_sql_views(config);
        if (!views.empty()) {
            std::string init_sql = build_sql_init(views) + build_metadata_views_sql(config);
            std::string sql = init_sql + "SELECT count(*) FROM " + views.front().view_name + ";";
            std::string cmd = "duckdb -batch -init /dev/null -cmd \"" + sql + "\" :memory: > /dev/null 2>&1";
            ok_sql = (std::system(cmd.c_str()) == 0);
            std::cout << "[SQL] view=" << views.front().view_name
                      << " -> " << (ok_sql ? "OK" : "FAIL") << "\n";
        } else {
            ok_sql = false;
            std::cout << "[SQL] no dataset views found -> FAIL\n";
        }
    }

    bool pass = ok_local && ok_cloud && ok_meta && ok_sql;
    std::cout << "Result: " << (pass ? "PASS" : "FAIL") << "\n";
    return pass ? 0 : 1;
}

// ============================================================================
// collect
// ============================================================================
int cmd_collect(const CliArgs& args, const trade::Config& config) {
    const std::string action = args.action.empty() ? "raw" : args.action;
    if (action == "sentiment") {
        app::SentimentRequest request;
        request.symbol = args.symbol;
        request.source = args.source;
        if (!args.start_date.empty()) request.start = parse_date(args.start_date);
        if (!args.end_date.empty()) request.end = parse_date(args.end_date);
        return app::run_sentiment(request, config);
    }

    if (action == "all") {
        trade::Config stage_cfg = config;
        stage_cfg.ingestion.write_silver_layer = false;
        stage_cfg.ingestion.write_raw_layer = true;

        app::DownloadRequest market_request;
        market_request.symbol = args.symbol;
        market_request.provider = args.provider;
        market_request.refresh = args.refresh;
        if (!args.start_date.empty()) market_request.start = parse_date(args.start_date);
        if (!args.end_date.empty()) market_request.end = parse_date(args.end_date);
        int rc = app::run_download(market_request, stage_cfg);
        if (rc != 0) return rc;

        app::SentimentRequest sentiment_request;
        sentiment_request.symbol = args.symbol;
        sentiment_request.source = args.source;
        if (!args.start_date.empty()) sentiment_request.start = parse_date(args.start_date);
        if (!args.end_date.empty()) sentiment_request.end = parse_date(args.end_date);
        rc = app::run_sentiment(sentiment_request, config);
        return rc;
    }

    if (action == "stream") {
        if (args.symbol.empty()) {
            spdlog::error("collect --action stream requires --symbol list (comma-separated).");
            return 1;
        }
        if (args.start_date.empty()) {
            spdlog::error("collect --action stream requires --start.");
            return 1;
        }
        if (!args.end_date.empty()) {
            spdlog::warn("--end is ignored in stream mode; end is always current time.");
        }

        trade::Config stage_cfg = config;
        stage_cfg.ingestion.write_silver_layer = false;
        stage_cfg.ingestion.write_raw_layer = true;

        app::DownloadRequest request;
        request.symbol = args.symbol;
        request.provider = args.provider;
        request.refresh = false;
        request.start = parse_date(args.start_date);

        const int interval_sec = std::max(1, config.ingestion.stream_poll_interval_sec);
        g_stream_stop.store(false);
        auto prev_handler = std::signal(SIGINT, on_stream_signal);

        std::cout << "Stream collection started (provider=" << request.provider
                  << ", symbols=" << request.symbol
                  << ", poll=" << interval_sec << "s). Press Ctrl+C to stop."
                  << std::endl;

        int cycle = 0;
        int rc = 0;
        while (!g_stream_stop.load()) {
            ++cycle;
            spdlog::info("stream cycle {}", cycle);
            rc = app::run_download(request, stage_cfg);
            if (rc != 0) {
                std::signal(SIGINT, prev_handler);
                return rc;
            }

            for (int i = 0; i < interval_sec && !g_stream_stop.load(); ++i) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }

        std::signal(SIGINT, prev_handler);
        std::cout << "Stream collection stopped." << std::endl;
        return rc;
    }

    if (action != "raw") {
        spdlog::error("Unsupported collect action '{}'. Use raw|sentiment|all|stream", action);
        return 1;
    }

    const bool no_pull_args = args.symbol.empty() &&
                              args.start_date.empty() &&
                              args.end_date.empty() &&
                              !args.refresh;
    if (no_pull_args) {
        spdlog::info("collect without explicit args -> full refresh all symbols from min_start_date");
    }

    trade::Config stage_cfg = config;
    stage_cfg.ingestion.write_silver_layer = false;
    stage_cfg.ingestion.write_raw_layer = true;

    app::DownloadRequest request;
    request.symbol = args.symbol;
    request.provider = args.provider;
    request.refresh = args.refresh || no_pull_args;
    if (!args.start_date.empty()) {
        request.start = parse_date(args.start_date);
    } else if (no_pull_args) {
        request.start = parse_date(config.ingestion.min_start_date);
    }
    if (!args.end_date.empty()) request.end = parse_date(args.end_date);
    return app::run_download(request, stage_cfg);
}

// ============================================================================
// silver
// ============================================================================
int cmd_silver(const CliArgs& args, const trade::Config& config) {
    if (!args.action.empty()) {
        spdlog::error("silver command does not use --action. Use --symbol/--start/--end.");
        return 1;
    }

    trade::Config stage_cfg = config;
    stage_cfg.ingestion.write_raw_layer = false;
    stage_cfg.ingestion.write_silver_layer = true;
    Collector collector(std::make_unique<RawOnlySilverProvider>(), stage_cfg);

    auto today = std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now());
    Date start = args.start_date.empty()
        ? parse_date(stage_cfg.ingestion.min_start_date)
        : parse_date(args.start_date);
    Date end = args.end_date.empty() ? today : parse_date(args.end_date);
    if (start > end) {
        spdlog::error("Invalid date range: start > end");
        return 1;
    }

    if (!args.symbol.empty()) {
        auto report = collector.build_silver_symbol(args.symbol, start, end);
        std::cout << "Built silver for " << args.symbol
                  << " (rows=" << report.total_bars
                  << ", quality=" << std::fixed << std::setprecision(1)
                  << (report.quality_score() * 100) << "%)"
                  << std::endl;
    } else {
        collector.build_silver_all(start, end,
            [](const Symbol& sym, int cur, int total) {
                std::cout << "\r[silver " << cur << "/" << total << "] " << sym
                          << "                " << std::flush;
            });
        std::cout << "\nSilver build complete." << std::endl;
    }
    return 0;
}

// ============================================================================
// cleanup
// ============================================================================
int cmd_cleanup(const CliArgs& args, const trade::Config& config) {
    const std::string action = args.action.empty() ? "audit" : args.action;

    if (action != "audit" && action != "apply") {
        spdlog::error("Unsupported cleanup action '{}'. Use audit|apply", action);
        return 1;
    }

    const bool apply = (action == "apply");
    const std::string mode = apply ? "apply" : "audit";

    trade::StoragePath paths(config.data.data_root);
    trade::MetadataStore metadata(paths.metadata_db());

    auto instruments = metadata.get_all_instruments();
    std::cout << "=== Data Cleanup (" << mode << ") ===\n"
              << "Data root: " << config.data.data_root << "\n"
              << "Instruments: " << instruments.size() << "\n";

    if (!apply) {
        std::cout << "Dry run only. Use: trade_cli cleanup --action apply --config <path>\n";
    }
    return 0;
}


// ============================================================================
// info
// ============================================================================
int cmd_info(const CliArgs& args, const trade::Config& config) {
    if (!args.symbol.empty()) {
        auto bars = load_bars(args.symbol, config);
        std::cout << "Symbol: " << args.symbol << "\nBars: " << bars.size() << std::endl;
        if (!bars.empty()) {
            std::cout << "Range: " << trade::format_date(bars.front().date)
                     << " to " << trade::format_date(bars.back().date) << std::endl;
            std::cout << "Last close: " << std::fixed << std::setprecision(2)
                     << bars.back().close << std::endl;
        }
    } else {
        std::cout << "Data root: " << config.data.data_root << "\nProvider: "
                 << args.provider << std::endl;
    }
    return 0;
}

// ============================================================================
// sql — launch DuckDB CLI with data directory pre-configured
// ============================================================================
int cmd_sql(const CliArgs& args, const trade::Config& config) {
    // Check if duckdb is available
    if (std::system("which duckdb > /dev/null 2>&1") != 0) {
        spdlog::error("duckdb not found. Install with: brew install duckdb");
        return 1;
    }

    const bool cloud_mode = config.storage.enabled &&
        config.storage.backend == "google_drive";

    // Cloud backflow: hydrate requested file/symbol into local cache before DuckDB starts.
    bool symbol_hydrated = false;
    if (cloud_mode) {
        if (!args.file.empty() && !std::filesystem::exists(args.file)) {
            auto t = trade::ParquetReader::read_table(args.file);
            if (!t) {
                spdlog::warn("Failed to hydrate --file from cloud: {}", args.file);
            }
        }
        if (!args.symbol.empty()) {
            auto hydrated = load_bars(args.symbol, config);
            symbol_hydrated = !hydrated.empty();
            if (!symbol_hydrated) {
                spdlog::warn("No local/cloud data found for symbol {}", args.symbol);
            }
        }
    }

    // Build init SQL: create views from dataset catalog.
    auto views = discover_sql_views(config);
    std::string init_sql = build_sql_init(views);
    init_sql += build_metadata_views_sql(config);

    auto has_dataset = [&](const std::string& dataset_id) {
        return std::any_of(views.begin(), views.end(), [&](const SqlViewDef& v) {
            return v.dataset_id == dataset_id;
        });
    };

    // If a specific file is given, also create a 'data' view
    bool data_view_ready = false;
    if (!args.file.empty()) {
        if (std::filesystem::exists(args.file)) {
            init_sql += "CREATE OR REPLACE VIEW data AS SELECT * FROM read_parquet('" +
                        sql_escape(args.file) + "', union_by_name=true);";
            data_view_ready = true;
        }
    } else if (!args.symbol.empty()) {
        if (!cloud_mode || symbol_hydrated) {
            if (has_dataset("kline") || has_dataset("daily")) {
                init_sql += "CREATE OR REPLACE VIEW data AS "
                            "SELECT * FROM kline WHERE symbol='" +
                            sql_escape(args.symbol) + "';";
                data_view_ready = true;
            }
        }
    }

    std::cout << "Starting DuckDB SQL shell...\n"
              << "Pre-configured views from catalog:\n";
    for (const auto& v : views) {
        std::cout << "  " << v.view_name << "  (" << v.dataset_id << ")\n";
    }
    if (data_view_ready) {
        std::cout << "  data   - specific file/symbol data\n";
    }
    if (views.empty() && !data_view_ready) {
        std::cout << "  (no local parquet found yet; run collect first)\n";
    }
    std::cout << "\nExample queries:\n";
    if (has_dataset("kline") || has_dataset("daily")) {
        std::cout << "  SELECT * FROM kline WHERE symbol='600000.SH' ORDER BY date;\n"
                  << "  SELECT symbol, count(*) FROM kline GROUP BY symbol;\n";
    } else if (!views.empty()) {
        std::cout << "  SELECT * FROM " << views.front().view_name << " LIMIT 20;\n";
    } else {
        std::cout << "  -- no dataset views yet; run collect first\n";
    }
    if (data_view_ready) {
        std::cout << "  SELECT * FROM data LIMIT 20;\n";
    }
    std::cout << "  SELECT * FROM meta_instruments;\n";
    std::cout << std::endl;

    if (cloud_mode) {
        std::cout << "Cloud mode enabled: DuckDB sees local + hydrated cache partitions.\n"
                  << "Tip: use --symbol to pre-hydrate one symbol from Google Drive cloud.\n"
                  << std::endl;
    }

    // Launch duckdb with init commands
    std::string cmd = "duckdb -init /dev/null -cmd \"" + init_sql + "\"";
    return std::system(cmd.c_str());
}

} // namespace trade::cli
