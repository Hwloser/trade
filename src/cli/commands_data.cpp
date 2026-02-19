#include "trade/cli/commands.h"

#include "trade/app/pipelines/download_pipeline.h"
#include "trade/cli/shared.h"
#include "trade/common/time_utils.h"
#include "trade/collector/collector.h"
#include "trade/storage/baidu_netdisk_client.h"
#include "trade/storage/metadata_store.h"
#include "trade/storage/parquet_reader.h"
#include "trade/storage/parquet_writer.h"
#include "trade/storage/storage_path.h"
#include "trade/validator/data_validator.h"
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <spdlog/spdlog.h>

namespace trade::cli {
namespace {

bool cloud_mode_enabled(const trade::Config& config) {
    return config.storage.enabled &&
        (config.storage.backend == "baidu_netdisk" || config.storage.backend == "baidu");
}

int days_old(Date d) {
    auto now = std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now());
    return static_cast<int>(std::chrono::duration_cast<std::chrono::days>(now - d).count());
}

bool is_legacy_curated_dataset(const std::string& dataset_id, const std::string& layer) {
    return layer == "curated" || dataset_id.rfind("curated.", 0) == 0;
}

int ttl_days_for_dataset(const trade::Config& config,
                         const trade::MetadataStore::DatasetRecord& ds) {
    if (ds.dataset_id.rfind("raw.sentiment.", 0) == 0 &&
        config.storage.ttl_sentiment_raw_days > 0) {
        return config.storage.ttl_sentiment_raw_days;
    }
    if (ds.layer == "raw" && config.storage.ttl_raw_days > 0) {
        return config.storage.ttl_raw_days;
    }
    if (ds.layer == "silver" && config.storage.ttl_silver_days > 0) {
        return config.storage.ttl_silver_days;
    }
    if (ds.layer == "gold" && config.storage.ttl_gold_days > 0) {
        return config.storage.ttl_gold_days;
    }
    if (config.storage.ttl_global_days > 0) {
        return config.storage.ttl_global_days;
    }
    return 0;
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

} // namespace

int cmd_verify(const CliArgs& args, const trade::Config& config) {
    bool ok_local = false;
    bool ok_cloud = false;
    bool ok_meta = false;
    bool ok_sql = false;

    trade::StoragePath paths(config.data.data_root);
    trade::MetadataStore metadata(paths.metadata_db());

    std::cout << "=== Verify Data Pipeline ===\n";

    // 1) Local check
    if (!args.symbol.empty()) {
        auto bars = load_bars(args.symbol, config);
        ok_local = !bars.empty();
        std::cout << "[Local] symbol=" << args.symbol
                  << " rows=" << bars.size()
                  << " -> " << (ok_local ? "OK" : "FAIL") << "\n";
    } else {
        auto datasets = metadata.list_datasets();
        ok_local = !datasets.empty();
        std::cout << "[Local] catalog datasets=" << datasets.size()
                  << " -> " << (ok_local ? "OK" : "FAIL") << "\n";
    }

    // 2) Cloud check (optional)
    const bool cloud_mode = config.storage.enabled &&
        (config.storage.backend == "baidu_netdisk" || config.storage.backend == "baidu");
    if (cloud_mode) {
        trade::BaiduNetdiskClient client({
            .access_token = config.storage.baidu_access_token,
            .refresh_token = config.storage.baidu_refresh_token,
            .app_key = config.storage.baidu_app_key,
            .app_secret = config.storage.baidu_app_secret,
            .app_id = config.storage.baidu_app_id,
            .sign_key = config.storage.baidu_sign_key,
            .root_path = config.storage.baidu_root,
            .timeout_ms = config.storage.baidu_timeout_ms,
            .retry_count = config.storage.baidu_retry_count,
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
    std::cout << "[Meta] datasets=" << mh.dataset_count
              << " with_files=" << mh.dataset_with_files
              << " schema_match=" << mh.dataset_schema_match
              << " file_versions=" << mh.file_version_covered << "/" << mh.file_total
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
    if (action != "raw") {
        spdlog::error("collect only supports raw ingestion. Use 'silver' command for raw->silver build.");
        return 1;
    }

    trade::Config stage_cfg = config;
    stage_cfg.ingestion.write_silver_layer = false;
    stage_cfg.ingestion.write_raw_layer = true;

    app::DownloadRequest request;
    request.symbol = args.symbol;
    request.provider = args.provider;
    request.refresh = args.refresh;
    if (!args.start_date.empty()) request.start = parse_date(args.start_date);
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
    const bool apply = (args.action == "apply");
    const bool cloud_mode = cloud_mode_enabled(config);
    const std::string mode = apply ? "apply" : "audit";

    trade::StoragePath paths(config.data.data_root);
    trade::MetadataStore metadata(paths.metadata_db());
    auto datasets = metadata.list_datasets();

    int files_total = 0;
    int files_missing_local = 0;
    int files_ttl_expired = 0;
    int files_legacy_curated = 0;
    int files_removed = 0;
    int raw_files_dirty = 0;
    int raw_files_cleaned = 0;
    int64_t raw_rows_removed = 0;

    std::cout << "=== Data Cleanup (" << mode << ") ===\n"
              << "Data root: " << config.data.data_root << "\n"
              << "Cloud mode: " << (cloud_mode ? "enabled" : "disabled") << "\n"
              << "TTL policy (days): global=" << config.storage.ttl_global_days
              << ", raw=" << config.storage.ttl_raw_days
              << ", silver=" << config.storage.ttl_silver_days
              << ", gold=" << config.storage.ttl_gold_days
              << ", raw.sentiment=" << config.storage.ttl_sentiment_raw_days
              << "\n";

    for (const auto& ds : datasets) {
        auto files = metadata.list_dataset_files(ds.dataset_id);
        for (const auto& f : files) {
            if (!args.symbol.empty()) {
                auto file_symbol = symbol_from_file_path(f.file_path);
                if (file_symbol != args.symbol) continue;
            }
            ++files_total;

            const std::filesystem::path abs_path =
                std::filesystem::path(config.data.data_root) / f.file_path;
            const bool exists_local = std::filesystem::exists(abs_path);

            bool removed = false;
            if (!exists_local && !cloud_mode) {
                ++files_missing_local;
                if (apply) {
                    metadata.delete_dataset_file(ds.dataset_id, f.file_path, "missing_local_file");
                    ++files_removed;
                    removed = true;
                }
            }

            if (!removed) {
                const int ttl_days = ttl_days_for_dataset(config, ds);
                if (ttl_days > 0 && f.max_event_date &&
                    days_old(*f.max_event_date) > ttl_days) {
                    ++files_ttl_expired;
                    if (apply) {
                        if (cloud_mode && !exists_local) {
                            spdlog::warn("Skip TTL delete for cloud-only file not present locally: {}",
                                         f.file_path);
                            continue;
                        }
                        if (exists_local) {
                            std::error_code ec;
                            std::filesystem::remove(abs_path, ec);
                            if (ec) {
                                spdlog::warn("Failed to remove expired file {}: {}",
                                             abs_path.string(), ec.message());
                            }
                        }
                        metadata.delete_dataset_file(ds.dataset_id,
                                                     f.file_path,
                                                     "ttl_expired_" + std::to_string(ttl_days) + "d");
                        ++files_removed;
                        removed = true;
                    }
                }
            }

            if (!removed && is_legacy_curated_dataset(ds.dataset_id, ds.layer)) {
                ++files_legacy_curated;
                if (apply) {
                    if (exists_local) {
                        std::error_code ec;
                        std::filesystem::remove(abs_path, ec);
                        if (ec) {
                            spdlog::warn("Failed to remove legacy curated file {}: {}",
                                         abs_path.string(), ec.message());
                        }
                    }
                    metadata.delete_dataset_file(ds.dataset_id, f.file_path, "legacy_curated_cleanup");
                    ++files_removed;
                    removed = true;
                }
            }

            if (removed) continue;

            // Raw cleaning layer: sanitize only market raw bars
            if (ds.dataset_id == "raw.cn_a.daily" && exists_local) {
                std::vector<Bar> bars;
                try {
                    bars = trade::ParquetReader::read_bars(abs_path.string());
                } catch (const std::exception& e) {
                    spdlog::warn("Failed to read raw bars {}: {}", abs_path.string(), e.what());
                    continue;
                }
                auto report = trade::DataValidator::validate(bars);
                const bool dirty =
                    report.duplicate_dates > 0 ||
                    report.price_anomalies > 0 ||
                    report.volume_anomalies > 0;
                if (!dirty) continue;

                ++raw_files_dirty;
                if (!apply) continue;

                int64_t invalid_dropped = 0;
                int64_t duplicate_dropped = 0;
                auto cleaned = sanitize_raw_bars(
                    bars, &invalid_dropped, &duplicate_dropped);
                raw_rows_removed += invalid_dropped + duplicate_dropped;

                if (cleaned.empty()) {
                    std::error_code ec;
                    std::filesystem::remove(abs_path, ec);
                    metadata.delete_dataset_file(ds.dataset_id, f.file_path, "cleaned_to_empty");
                    ++files_removed;
                    continue;
                }

                Date max_date = cleaned.front().date;
                for (const auto& b : cleaned) {
                    if (b.date > max_date) max_date = b.date;
                }
                trade::ParquetStore::write_bars(abs_path.string(),
                                                cleaned,
                                                trade::ParquetStore::MergeMode::kReplace,
                                                max_date);
                ++raw_files_cleaned;
            }
        }
    }

    int datasets_pruned = 0;
    if (apply) {
        datasets_pruned = metadata.prune_empty_datasets();
    }

    std::cout << "[Summary] files_total=" << files_total
              << " missing_local=" << files_missing_local
              << " ttl_expired=" << files_ttl_expired
              << " legacy_curated=" << files_legacy_curated
              << " raw_dirty=" << raw_files_dirty
              << "\n";
    std::cout << "[Action] files_removed=" << files_removed
              << " raw_cleaned=" << raw_files_cleaned
              << " raw_rows_removed=" << raw_rows_removed
              << " datasets_pruned=" << datasets_pruned
              << "\n";

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
// view — display parquet file contents as a table
// ============================================================================
int cmd_view(const CliArgs& args, const trade::Config& config) {
    std::string path = args.file;

    // If no --file, try to find from --symbol
    if (path.empty() && !args.symbol.empty()) {
        trade::StoragePath paths(config.data.data_root);
        auto now = std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now());
        int end_year = trade::date_year(now);
        int start_year = trade::date_year(trade::parse_date(config.ingestion.min_start_date));
        for (int year = end_year; year >= start_year; --year) {
            std::string raw_path = paths.raw_daily(args.symbol, year);
            std::string silver_path = paths.silver_daily(args.symbol, year);
            std::string legacy_curated_path =
                (std::filesystem::path(config.data.data_root) / "curated" /
                 config.data.market_daily_subpath / std::to_string(year) /
                 (args.symbol + ".parquet"))
                    .string();

            if (std::filesystem::exists(raw_path)) {
                path = raw_path;
                break;
            }
            if (std::filesystem::exists(silver_path)) {
                path = silver_path;
                break;
            }
            if (std::filesystem::exists(legacy_curated_path)) {
                path = legacy_curated_path;
                break;
            }
        }
        if (path.empty()) {
            path = paths.raw_daily(args.symbol, end_year);
        }
    }

    if (path.empty()) {
        spdlog::error("Specify --file <path.parquet> or --symbol <symbol>");
        return 1;
    }

    std::shared_ptr<arrow::Table> table;
    try {
        table = trade::ParquetReader::read_table(path);
    } catch (const std::exception& e) {
        spdlog::error("Failed to read {}: {}", path, e.what());
        return 1;
    }

    int64_t num_rows = table->num_rows();
    int num_cols = table->num_columns();
    int display_rows = (args.limit > 0 && args.limit < num_rows)
        ? args.limit : static_cast<int>(num_rows);

    std::cout << "File: " << path << "\n"
              << "Rows: " << num_rows << "  Columns: " << num_cols << "\n\n";

    // Compute column widths
    std::vector<int> widths(num_cols);
    std::vector<std::string> col_names(num_cols);
    for (int c = 0; c < num_cols; ++c) {
        col_names[c] = table->schema()->field(c)->name();
        widths[c] = static_cast<int>(col_names[c].size());
    }

    // Get string representations for each cell
    std::vector<std::vector<std::string>> cells(display_rows, std::vector<std::string>(num_cols));
    for (int c = 0; c < num_cols; ++c) {
        auto col = table->column(c);
        for (int r = 0; r < display_rows; ++r) {
            int chunk_idx = 0;
            int64_t offset = r;
            while (chunk_idx < col->num_chunks() &&
                   offset >= col->chunk(chunk_idx)->length()) {
                offset -= col->chunk(chunk_idx)->length();
                ++chunk_idx;
            }
            if (chunk_idx < col->num_chunks()) {
                auto arr = col->chunk(chunk_idx);
                if (arr->IsNull(offset)) {
                    cells[r][c] = "null";
                } else {
                    auto scalar = arr->GetScalar(offset);
                    if (scalar.ok()) {
                        cells[r][c] = (*scalar)->ToString();
                    } else {
                        cells[r][c] = "?";
                    }
                }
            }
            widths[c] = std::max(widths[c], static_cast<int>(cells[r][c].size()));
        }
    }

    // Print header
    for (int c = 0; c < num_cols; ++c) {
        std::cout << std::left << std::setw(widths[c] + 2) << col_names[c];
    }
    std::cout << "\n";
    for (int c = 0; c < num_cols; ++c) {
        std::cout << std::string(widths[c], '-') << "  ";
    }
    std::cout << "\n";

    // Print rows
    for (int r = 0; r < display_rows; ++r) {
        for (int c = 0; c < num_cols; ++c) {
            std::cout << std::left << std::setw(widths[c] + 2) << cells[r][c];
        }
        std::cout << "\n";
    }

    if (display_rows < num_rows) {
        std::cout << "... (" << (num_rows - display_rows)
                  << " more rows)" << std::endl;
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
        (config.storage.backend == "baidu_netdisk" || config.storage.backend == "baidu");

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
            std::string layer_dir = has_dataset("raw.cn_a.daily")
                ? config.data.raw_dir
                : config.data.silver_dir;
            std::string pattern = config.data.data_root + "/" + layer_dir + "/" +
                                  config.data.market_daily_subpath + "/**/" +
                                  args.symbol + ".parquet";
            init_sql += "CREATE OR REPLACE VIEW data AS SELECT * FROM read_parquet('" +
                        sql_escape(pattern) + "', union_by_name=true);";
            data_view_ready = true;
        }
    }

    std::cout << "Starting DuckDB SQL shell...\n"
              << "Pre-configured views from catalog:\n";
    for (const auto& v : views) {
        std::cout << "  " << v.view_name << "  (" << v.dataset_id << ")\n";
    }
    if (has_dataset("raw.cn_a.daily")) {
        std::cout << "  daily  (alias of raw_cn_a_daily)\n";
    } else if (has_dataset("silver.cn_a.daily")) {
        std::cout << "  daily  (alias of silver_cn_a_daily)\n";
    }
    if (has_dataset("raw.cn_a.daily")) {
        std::cout << "  raw    (alias of raw_cn_a_daily)\n";
    }
    if (data_view_ready) {
        std::cout << "  data   - specific file/symbol data\n";
    }
    if (views.empty() && !data_view_ready) {
        std::cout << "  (no local parquet found yet; run collect/sentiment first)\n";
    }
    std::cout << "\nExample queries:\n";
    if (has_dataset("raw.cn_a.daily")) {
        std::cout << "  SELECT * FROM raw_cn_a_daily WHERE symbol='600000.SH' ORDER BY date;\n"
                  << "  SELECT symbol, count(*) FROM raw_cn_a_daily GROUP BY symbol;\n";
    } else if (has_dataset("silver.cn_a.daily")) {
        std::cout << "  SELECT * FROM silver_cn_a_daily WHERE symbol='600000.SH' ORDER BY date;\n"
                  << "  SELECT symbol, count(*) FROM silver_cn_a_daily GROUP BY symbol;\n";
    } else if (!views.empty()) {
        std::cout << "  SELECT * FROM " << views.front().view_name << " LIMIT 20;\n";
    } else {
        std::cout << "  -- no dataset views yet; run collect/sentiment first\n";
    }
    if (data_view_ready) {
        std::cout << "  SELECT * FROM data LIMIT 20;\n";
    }
    std::cout << "  SELECT * FROM meta_dataset_catalog;\n"
              << "  SELECT * FROM meta_dataset_files ORDER BY dataset_id, file_path LIMIT 50;\n"
              << "  SELECT * FROM meta_dataset_tombstones_recent LIMIT 50;\n"
              << "  SELECT * FROM meta_quality_checks_recent WHERE status <> 'pass';\n"
              << "  SELECT * FROM meta_accounts;\n"
              << "  SELECT * FROM meta_account_positions;\n";
    std::cout << std::endl;

    if (cloud_mode) {
        std::cout << "Cloud mode enabled: DuckDB sees local + hydrated cache partitions.\n"
                  << "Tip: use --symbol to pre-hydrate one symbol from Baidu cloud.\n"
                  << std::endl;
    }

    // Launch duckdb with init commands
    std::string cmd = "duckdb -init /dev/null -cmd \"" + init_sql + "\"";
    return std::system(cmd.c_str());
}

} // namespace trade::cli
