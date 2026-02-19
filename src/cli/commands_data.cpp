#include "trade/cli/commands.h"

#include "trade/app/pipelines/download_pipeline.h"
#include "trade/cli/shared.h"
#include "trade/common/time_utils.h"
#include "trade/storage/baidu_netdisk_client.h"
#include "trade/storage/metadata_store.h"
#include "trade/storage/parquet_reader.h"
#include "trade/storage/parquet_writer.h"
#include "trade/storage/storage_path.h"
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <spdlog/spdlog.h>

namespace trade::cli {
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
// download
// ============================================================================
int cmd_download(const CliArgs& args, const trade::Config& config) {
    app::DownloadRequest request;
    request.symbol = args.symbol;
    request.provider = args.provider;
    request.refresh = args.refresh;
    if (!args.start_date.empty()) {
        request.start = parse_date(args.start_date);
    }
    if (!args.end_date.empty()) {
        request.end = parse_date(args.end_date);
    }
    return app::run_download(request, config);
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
        std::cout << "  (no local parquet found yet; run download/sentiment first)\n";
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
        std::cout << "  -- no dataset views yet; run download/sentiment first\n";
    }
    if (data_view_ready) {
        std::cout << "  SELECT * FROM data LIMIT 20;\n";
    }
    std::cout << "  SELECT * FROM meta_dataset_catalog;\n"
              << "  SELECT * FROM meta_dataset_files ORDER BY dataset_id, file_path LIMIT 50;\n"
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
