#include "trade/common/config.h"
#include "trade/common/time_utils.h"
#include "trade/common/types.h"
#include "trade/collector/collector.h"
#include "trade/features/feature_engine.h"
#include "trade/features/momentum.h"
#include "trade/features/volatility.h"
#include "trade/features/liquidity.h"
#include "trade/features/feature_monitor.h"
#include "trade/features/preprocessor.h"
#include "trade/stats/descriptive.h"
#include "trade/stats/correlation.h"
#include "trade/risk/var.h"
#include "trade/risk/kelly.h"
#include "trade/risk/covariance.h"
#include "trade/risk/position_sizer.h"
#include "trade/risk/risk_monitor.h"
#include "trade/regime/regime_detector.h"
#include "trade/backtest/backtest_engine.h"
#include "trade/backtest/performance.h"
#include "trade/sentiment/rss_source.h"
#include "trade/sentiment/rule_sentiment.h"
#include "trade/sentiment/sentiment_factor.h"
#include "trade/sentiment/symbol_linker.h"
#include "trade/sentiment/text_cleaner.h"
#include "trade/decision/decision_report.h"
#include "trade/provider/provider_factory.h"
#include "trade/storage/metadata_store.h"
#include "trade/storage/parquet_reader.h"
#include "trade/storage/parquet_writer.h"
#include "trade/storage/baidu_netdisk_client.h"
#include "trade/storage/storage_path.h"
#include <arrow/api.h>
#include <arrow/io/api.h>

#ifdef HAVE_LIGHTGBM
#include "trade/ml/lgbm_model.h"
#include "trade/ml/model_trainer.h"
#include "trade/ml/model_evaluator.h"
#endif

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

void print_usage() {
    std::cout << R"(
trade_cli - Quantitative Trading Decision Support System

Usage:
  trade_cli <command> [options]

Commands:
  download    Download market data (incremental by default)
  verify      Verify local/cloud/sql data pipeline
  view        (Paused) Use sql for querying data
  sql         Open DuckDB SQL shell with data pre-loaded
  features    Compute features for a symbol
  train       Train ML model
  predict     Generate predictions
  risk        Assess risk for a position
  backtest    Run backtest
  sentiment   Analyze sentiment from RSS feeds
  report      Generate decision report
  info        (Paused) Use sql for querying metadata

Options:
  --config <path>       Config path (file or dir, default: config)
  --symbol <symbol>     Stock symbol (e.g., 600000.SH)
  --start <date>        Start date (YYYY-MM-DD)
  --end <date>          End date (YYYY-MM-DD)
  --provider <name>     Data provider (default: eastmoney)
  --refresh             Force full re-download (overwrite existing data)
  --file <path>         Parquet file path (for view command)
  --limit <n>           Max rows to display (for view command, default: all)
  --model <name>        Model name (e.g., lgbm)
  --strategy <name>     Strategy name
  --source <name>       Sentiment source (rss, xueqiu, jin10)
  --output <path>       Output file path
  --verbose             Enable verbose logging
  --help                Show this help
)" << std::endl;
}

struct CliArgs {
    std::string command;
    std::string config_path = "config";
    std::string symbol;
    std::string start_date;
    std::string end_date;
    std::string provider = "eastmoney";
    std::string model;
    std::string strategy;
    std::string source;
    std::string output;
    std::string file;  // for view command
    bool verbose = false;
    bool refresh = false;  // force full re-download
    int limit = 0;  // max rows for view command
};

CliArgs parse_args(int argc, char* argv[]) {
    CliArgs args;
    if (argc < 2) return args;

    args.command = argv[1];

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) args.config_path = argv[++i];
        else if (arg == "--symbol" && i + 1 < argc) args.symbol = argv[++i];
        else if (arg == "--start" && i + 1 < argc) args.start_date = argv[++i];
        else if (arg == "--end" && i + 1 < argc) args.end_date = argv[++i];
        else if (arg == "--provider" && i + 1 < argc) args.provider = argv[++i];
        else if (arg == "--model" && i + 1 < argc) args.model = argv[++i];
        else if (arg == "--strategy" && i + 1 < argc) args.strategy = argv[++i];
        else if (arg == "--source" && i + 1 < argc) args.source = argv[++i];
        else if (arg == "--output" && i + 1 < argc) args.output = argv[++i];
        else if (arg == "--file" && i + 1 < argc) args.file = argv[++i];
        else if (arg == "--limit" && i + 1 < argc) args.limit = std::stoi(argv[++i]);
        else if (arg == "--verbose") args.verbose = true;
        else if (arg == "--refresh") args.refresh = true;
        else if (arg == "--help") { print_usage(); std::exit(0); }
    }
    return args;
}

std::pair<trade::Date, trade::Date> resolve_dates(const CliArgs& args,
                                                    const std::string& default_start) {
    auto start = args.start_date.empty()
        ? trade::parse_date(default_start)
        : trade::parse_date(args.start_date);
    auto end = args.end_date.empty()
        ? std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now())
        : trade::parse_date(args.end_date);
    return {start, end};
}

std::vector<trade::Bar> load_bars(const std::string& symbol,
                                   const trade::Config& config) {
    trade::StoragePath paths(config.data.data_root);
    std::vector<trade::Bar> all_bars;
    const bool cloud_mode = config.storage.enabled &&
        (config.storage.backend == "baidu_netdisk" || config.storage.backend == "baidu");
    auto now = std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now());
    auto min_date = trade::parse_date(config.ingestion.min_start_date);
    int start_year = trade::date_year(min_date);
    int end_year = trade::date_year(now);
    for (int year = start_year; year <= end_year; ++year) {
        auto path = paths.curated_daily(symbol, year);
        if (!std::filesystem::exists(path) && !cloud_mode) {
            continue;
        }
        try {
            auto bars = trade::ParquetReader::read_bars(path);
            all_bars.insert(all_bars.end(), bars.begin(), bars.end());
        } catch (...) {}
    }
    return all_bars;
}

struct SqlViewDef {
    std::string dataset_id;
    std::string view_name;
    std::string glob_path;
};

std::string sql_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        out.push_back(c);
        if (c == '\'') out.push_back('\'');
    }
    return out;
}

std::string sanitize_view_name(std::string s) {
    for (char& c : s) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) {
            c = '_';
        }
    }
    if (s.empty()) s = "dataset";
    if (std::isdigit(static_cast<unsigned char>(s[0]))) {
        s = "d_" + s;
    }
    return s;
}

bool has_local_parquet(const std::filesystem::path& dir) {
    if (!std::filesystem::exists(dir)) return false;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() == ".parquet") return true;
    }
    return false;
}

std::vector<SqlViewDef> discover_sql_views(const trade::Config& config) {
    trade::StoragePath paths(config.data.data_root);
    trade::MetadataStore metadata(paths.metadata_db());
    auto datasets = metadata.list_datasets();

    std::vector<SqlViewDef> views;
    views.reserve(datasets.size());
    for (const auto& ds : datasets) {
        auto prefix = std::filesystem::path(config.data.data_root) / ds.path_prefix;
        if (!has_local_parquet(prefix)) continue;

        SqlViewDef def;
        def.dataset_id = ds.dataset_id;
        def.view_name = sanitize_view_name(ds.dataset_id);
        def.glob_path = (prefix / "**/*.parquet").string();
        views.push_back(std::move(def));
    }

    auto add_fallback = [&](const std::string& dataset_id, const std::filesystem::path& prefix) {
        if (!has_local_parquet(prefix)) return;
        auto it = std::find_if(views.begin(), views.end(),
                               [&](const SqlViewDef& v) { return v.dataset_id == dataset_id; });
        if (it != views.end()) return;
        views.push_back(SqlViewDef{
            .dataset_id = dataset_id,
            .view_name = sanitize_view_name(dataset_id),
            .glob_path = (prefix / "**/*.parquet").string(),
        });
    };

    add_fallback("curated.cn_a.daily",
                 std::filesystem::path(config.data.data_root) /
                 config.data.curated_dir / config.data.market_daily_subpath);
    add_fallback("raw.cn_a.daily",
                 std::filesystem::path(config.data.data_root) /
                 config.data.raw_dir / config.data.market_daily_subpath);

    std::sort(views.begin(), views.end(),
              [](const SqlViewDef& a, const SqlViewDef& b) { return a.view_name < b.view_name; });
    return views;
}

std::string build_sql_init(const std::vector<SqlViewDef>& views) {
    std::string init_sql;
    for (const auto& v : views) {
        init_sql += "CREATE OR REPLACE VIEW " + v.view_name +
                    " AS SELECT * FROM read_parquet('" + sql_escape(v.glob_path) +
                    "', union_by_name=true);";
    }
    auto has_dataset = [&](const std::string& dataset_id) {
        return std::any_of(views.begin(), views.end(), [&](const SqlViewDef& v) {
            return v.dataset_id == dataset_id;
        });
    };
    if (has_dataset("curated.cn_a.daily")) {
        init_sql += "CREATE OR REPLACE VIEW daily AS SELECT * FROM curated_cn_a_daily;";
    }
    if (has_dataset("raw.cn_a.daily")) {
        init_sql += "CREATE OR REPLACE VIEW raw AS SELECT * FROM raw_cn_a_daily;";
    }
    return init_sql;
}

int cmd_verify(const CliArgs& args, const trade::Config& config) {
    bool ok_local = false;
    bool ok_cloud = false;
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

    // 3) SQL check
    if (std::system("which duckdb > /dev/null 2>&1") != 0) {
        std::cout << "[SQL] duckdb not found -> FAIL\n";
        ok_sql = false;
    } else {
        auto views = discover_sql_views(config);
        if (!views.empty()) {
            std::string init_sql = build_sql_init(views);
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

    bool pass = ok_local && ok_cloud && ok_sql;
    std::cout << "Result: " << (pass ? "PASS" : "FAIL") << "\n";
    return pass ? 0 : 1;
}

// ============================================================================
// download
// ============================================================================
int cmd_download(const CliArgs& args, const trade::Config& config) {
    auto provider = trade::ProviderFactory::create(args.provider, config);
    if (!provider->ping()) {
        spdlog::error("Cannot connect to {} provider", args.provider);
        return 1;
    }
    trade::Collector collector(std::move(provider), config);

    trade::StoragePath paths(config.data.data_root);
    trade::MetadataStore metadata(paths.metadata_db());
    const std::string& dataset = config.ingestion.daily_bar_dataset;

    // Default start from config (last N days)
    auto now_tp = std::chrono::system_clock::now();
    auto now_day = std::chrono::floor<std::chrono::days>(now_tp);
    int history_days = std::max(1, config.ingestion.default_history_days);
    auto default_start = now_day - std::chrono::days{history_days};
    auto min_start = trade::parse_date(config.ingestion.min_start_date);
    if (default_start < min_start) default_start = min_start;
    std::string default_start_str = trade::format_date(default_start);

    if (!args.symbol.empty()) {
        trade::Date start, end;
        end = args.end_date.empty()
            ? std::chrono::floor<std::chrono::days>(now_tp)
            : trade::parse_date(args.end_date);

        std::string run_id = args.provider + "_dl_" + args.symbol + "_" +
            std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                now_tp.time_since_epoch()).count());
        std::string mode = args.refresh ? "full" : "incremental";
        metadata.begin_ingestion_run(run_id, args.provider, dataset, args.symbol, mode);

        try {
            // Incremental download: prefer watermark, fallback to last download
            if (!args.refresh && args.start_date.empty()) {
                int lookback_days = std::max(0, config.ingestion.incremental_lookback_days);
                auto wm = metadata.last_watermark_date(args.provider, dataset, args.symbol);
                if (wm) {
                    start = *wm - std::chrono::days{lookback_days};
                    if (start < min_start) start = min_start;
                    spdlog::info("Incremental from watermark {} (lookback {}d => {})",
                                 trade::format_date(*wm), lookback_days, trade::format_date(start));
                } else {
                    auto last = metadata.last_download_date(args.symbol);
                    if (last) {
                        start = trade::next_trading_day(*last);
                        if (start < min_start) start = min_start;
                        spdlog::info("Incremental from last download {} (start: {})",
                                     trade::format_date(*last), trade::format_date(start));
                    } else {
                        start = default_start;
                    }
                }

                if (start > end) {
                    metadata.finish_ingestion_run(run_id, true, 0, 0);
                    std::cout << "Already up to date (last target: "
                              << trade::format_date(end) << ")" << std::endl;
                    return 0;
                }
            } else {
                start = args.start_date.empty()
                    ? default_start
                    : trade::parse_date(args.start_date);
            }

            auto report = collector.collect_symbol(args.symbol, start, end);
            metadata.finish_ingestion_run(run_id, true,
                                          static_cast<int64_t>(report.total_bars),
                                          static_cast<int64_t>(report.valid_bars));

            std::cout << "Downloaded " << report.total_bars << " bars for " << args.symbol
                      << " (quality: " << std::fixed << std::setprecision(1)
                      << (report.quality_score() * 100) << "%)" << std::endl;
        } catch (const std::exception& e) {
            metadata.finish_ingestion_run(run_id, false, 0, 0, e.what());
            throw;
        }
    } else {
        std::string run_id = args.provider + "_dl_all_" +
            std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                now_tp.time_since_epoch()).count());
        std::string mode = args.refresh ? "full" : "incremental";
        metadata.begin_ingestion_run(run_id, args.provider, dataset, "*", mode);

        try {
            if (args.refresh) {
                auto [start_all, end_all] = resolve_dates(args, default_start_str);
                collector.collect_all(start_all, end_all,
                    [](const trade::Symbol& sym, int cur, int total) {
                        std::cout << "\r[" << cur << "/" << total << "] " << sym
                                 << "                " << std::flush;
                    });
            } else {
                collector.update_all(
                    [](const trade::Symbol& sym, int cur, int total) {
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
        int year = trade::date_year(now);
        path = paths.curated_daily(args.symbol, year);
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
    std::string init_sql;
    for (const auto& v : views) {
        init_sql += "CREATE OR REPLACE VIEW " + v.view_name +
                    " AS SELECT * FROM read_parquet('" + sql_escape(v.glob_path) +
                    "', union_by_name=true);";
    }

    auto has_dataset = [&](const std::string& dataset_id) {
        return std::any_of(views.begin(), views.end(), [&](const SqlViewDef& v) {
            return v.dataset_id == dataset_id;
        });
    };
    if (has_dataset("curated.cn_a.daily")) {
        init_sql += "CREATE OR REPLACE VIEW daily AS SELECT * FROM curated_cn_a_daily;";
    }
    if (has_dataset("raw.cn_a.daily")) {
        init_sql += "CREATE OR REPLACE VIEW raw AS SELECT * FROM raw_cn_a_daily;";
    }

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
            std::string pattern = config.data.data_root + "/" + config.data.curated_dir + "/" +
                                  config.data.market_daily_subpath + "/**/" + args.symbol + ".parquet";
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
    if (has_dataset("curated.cn_a.daily")) {
        std::cout << "  daily  (alias of curated_cn_a_daily)\n";
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
    if (has_dataset("curated.cn_a.daily")) {
        std::cout << "  SELECT * FROM curated_cn_a_daily WHERE symbol='600000.SH' ORDER BY date;\n"
                  << "  SELECT symbol, count(*) FROM curated_cn_a_daily GROUP BY symbol;\n";
    } else if (!views.empty()) {
        std::cout << "  SELECT * FROM " << views.front().view_name << " LIMIT 20;\n";
    } else {
        std::cout << "  -- no dataset views yet; run download/sentiment first\n";
    }
    if (data_view_ready) {
        std::cout << "  SELECT * FROM data LIMIT 20;\n";
    }
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

// ============================================================================
// features
// ============================================================================
int cmd_features(const CliArgs& args, const trade::Config& config) {
    if (args.symbol.empty()) {
        spdlog::error("--symbol required"); return 1;
    }
    auto bars = load_bars(args.symbol, config);
    if (bars.empty()) {
        spdlog::error("No data for {}", args.symbol); return 1;
    }
    auto [start, end] = resolve_dates(args, config.ingestion.min_start_date);
    std::vector<trade::Bar> filtered;
    for (const auto& b : bars)
        if (b.date >= start && b.date <= end) filtered.push_back(b);

    spdlog::info("Computing features for {} ({} bars)", args.symbol, filtered.size());

    trade::BarSeries series;
    series.symbol = args.symbol;
    series.bars = std::move(filtered);

    std::unordered_map<trade::Symbol, trade::Instrument> instruments;
    instruments[args.symbol] = trade::Instrument{args.symbol};

    trade::FeatureEngine engine;
    engine.register_calculator(std::make_unique<trade::MomentumCalculator>());
    engine.register_calculator(std::make_unique<trade::VolatilityCalculator>());
    engine.register_calculator(std::make_unique<trade::LiquidityCalculator>());

    auto features = engine.build({series}, instruments);

    std::cout << "Features: " << features.num_features() << " cols x "
              << features.num_observations() << " rows\n\nFeature names:" << std::endl;
    for (int i = 0; i < features.num_features(); ++i)
        std::cout << "  " << features.names[i] << std::endl;

    if (features.num_observations() > 0) {
        int last = features.num_observations() - 1;
        std::cout << "\nLast observation:" << std::endl;
        for (int i = 0; i < features.num_features(); ++i) {
            double v = features.matrix(last, i);
            if (!std::isnan(v))
                std::cout << "  " << std::left << std::setw(30) << features.names[i]
                         << std::fixed << std::setprecision(4) << v << std::endl;
        }
    }
    return 0;
}

// ============================================================================
// train
// ============================================================================
int cmd_train(const CliArgs& args, const trade::Config& config) {
#ifdef HAVE_LIGHTGBM
    if (args.symbol.empty()) { spdlog::error("--symbol required"); return 1; }
    auto bars = load_bars(args.symbol, config);
    if (bars.size() < 252) {
        spdlog::error("Need >=252 bars, got {}", bars.size()); return 1;
    }

    trade::BarSeries series{args.symbol, bars};
    std::unordered_map<trade::Symbol, trade::Instrument> instruments;
    instruments[args.symbol] = trade::Instrument{args.symbol};

    trade::FeatureEngine engine;
    engine.register_calculator(std::make_unique<trade::MomentumCalculator>());
    engine.register_calculator(std::make_unique<trade::VolatilityCalculator>());
    engine.register_calculator(std::make_unique<trade::LiquidityCalculator>());
    auto features = engine.build({series}, instruments);
    spdlog::info("Features: {} x {}", features.num_observations(), features.num_features());

    int n = features.num_observations();
    Eigen::VectorXd labels(n);
    labels.setZero();
    for (int i = 0; i + 5 < static_cast<int>(bars.size()); ++i)
        if (bars[i].close > 0)
            labels(i) = (bars[i + 5].close - bars[i].close) / bars[i].close;

    int split = static_cast<int>(n * 0.8);
    trade::LGBMModel model;
    trade::LGBMParams params;
    params.n_estimators = 300;
    params.learning_rate = 0.05;
    params.num_leaves = 31;

    auto result = model.train(
        features.matrix.topRows(split), labels.head(split), params,
        features.matrix.bottomRows(n - split), labels.tail(n - split));

    spdlog::info("Done: best_iter={} score={:.6f}", result.best_iteration, result.best_score);

    trade::StoragePath paths(config.data.data_root);
    std::string mpath = paths.models_dir() + "/lgbm_factor_v1.model";
    model.save(mpath);
    spdlog::info("Saved to {}", mpath);

    auto imp = model.feature_importance_named(features.names, "gain");
    std::cout << "\nTop features:" << std::endl;
    for (int i = 0; i < std::min(20, static_cast<int>(imp.size())); ++i)
        std::cout << "  " << std::left << std::setw(30) << imp[i].first
                 << std::fixed << std::setprecision(1) << imp[i].second << std::endl;
    return 0;
#else
    spdlog::error("LightGBM not available"); return 1;
#endif
}

// ============================================================================
// predict
// ============================================================================
int cmd_predict(const CliArgs& args, const trade::Config& config) {
#ifdef HAVE_LIGHTGBM
    if (args.symbol.empty()) { spdlog::error("--symbol required"); return 1; }
    auto bars = load_bars(args.symbol, config);
    if (bars.empty()) { spdlog::error("No data for {}", args.symbol); return 1; }

    trade::StoragePath paths(config.data.data_root);
    trade::LGBMModel model;
    try { model.load(paths.models_dir() + "/lgbm_factor_v1.model"); }
    catch (const std::exception& e) {
        spdlog::error("Load model failed: {} (run 'train' first)", e.what()); return 1;
    }

    trade::BarSeries series{args.symbol, bars};
    std::unordered_map<trade::Symbol, trade::Instrument> instruments;
    instruments[args.symbol] = trade::Instrument{args.symbol};

    trade::FeatureEngine engine;
    engine.register_calculator(std::make_unique<trade::MomentumCalculator>());
    engine.register_calculator(std::make_unique<trade::VolatilityCalculator>());
    engine.register_calculator(std::make_unique<trade::LiquidityCalculator>());
    auto features = engine.build({series}, instruments);
    if (features.num_observations() == 0) { spdlog::error("No features"); return 1; }

    int last = features.num_observations() - 1;
    double pred = model.predict_one(features.matrix.row(last));

    std::cout << "Symbol:    " << args.symbol << "\n"
              << "Date:      " << trade::format_date(bars.back().date) << "\n"
              << "Close:     " << std::fixed << std::setprecision(2)
              << bars.back().close << "\n"
              << "Pred 5d:   " << std::showpos << std::setprecision(4)
              << (pred * 100) << "%\n"
              << "Direction: " << (pred > 0 ? "UP" : "DOWN") << std::endl;
    return 0;
#else
    spdlog::error("LightGBM not available"); return 1;
#endif
}

// ============================================================================
// risk
// ============================================================================
int cmd_risk(const CliArgs& args, const trade::Config& config) {
    if (args.symbol.empty()) { spdlog::error("--symbol required"); return 1; }
    auto bars = load_bars(args.symbol, config);
    if (bars.size() < 60) {
        spdlog::error("Need >=60 bars, got {}", bars.size()); return 1;
    }

    std::vector<double> returns;
    for (size_t i = 1; i < bars.size(); ++i)
        if (bars[i - 1].close > 0)
            returns.push_back((bars[i].close - bars[i - 1].close) / bars[i - 1].close);

    Eigen::VectorXd w(1); w(0) = 1.0;
    Eigen::MatrixXd ret_mat(returns.size(), 1);
    for (size_t i = 0; i < returns.size(); ++i) ret_mat(i, 0) = returns[i];

    trade::CovarianceEstimator cov_est;
    auto cov = cov_est.estimate(ret_mat);

    trade::VaRCalculator var_calc;
    auto var = var_calc.compute(w, cov, ret_mat);

    double mean_r = 0;
    for (double r : returns) mean_r += r;
    mean_r /= returns.size();
    double vol = std::sqrt(cov(0, 0));

    Eigen::VectorXd mu_vec(1); mu_vec(0) = mean_r;
    Eigen::VectorXd sigma_vec(1); sigma_vec(0) = vol;
    Eigen::VectorXd conf_vec(1); conf_vec(0) = 1.0;

    trade::KellyCalculator kelly;
    auto k = kelly.compute_diagnostics(mu_vec, sigma_vec, conf_vec);

    double peak = bars[0].close, max_dd = 0;
    for (const auto& b : bars) {
        peak = std::max(peak, b.close);
        max_dd = std::max(max_dd, (peak - b.close) / peak);
    }

    std::cout << "=== Risk: " << args.symbol << " ===\n"
              << "Period: " << trade::format_date(bars.front().date) << " to "
              << trade::format_date(bars.back().date) << "\n\n"
              << "Daily vol:       " << std::fixed << std::setprecision(2)
              << (vol * 100) << "%\n"
              << "Annual vol:      " << (vol * std::sqrt(252) * 100) << "%\n"
              << "VaR 99% param:   " << (var.parametric.var * 100) << "%\n"
              << "VaR 99% hist:    " << (var.historical.var * 100) << "%\n"
              << "VaR 99% MC:      " << (var.monte_carlo.var * 100) << "%\n"
              << "VaR 99% prod:    " << (var.var_1d_99 * 100) << "%\n"
              << "CVaR 99%:        " << (var.parametric.cvar * 100) << "%\n"
              << "Quarter Kelly:   " << std::setprecision(1)
              << (k.final_weights.size() > 0 ? k.final_weights(0) * 100 : 0.0) << "%\n"
              << "Max drawdown:    " << std::setprecision(2) << (max_dd * 100) << "%\n";
    return 0;
}

// ============================================================================
// backtest
// ============================================================================
int cmd_backtest(const CliArgs& args, const trade::Config& config) {
    if (args.symbol.empty()) { spdlog::error("--symbol required"); return 1; }
    auto [start, end] = resolve_dates(args, "2022-01-01");
    auto bars = load_bars(args.symbol, config);
    if (bars.size() < 120) {
        spdlog::error("Need >=120 bars, got {}", bars.size()); return 1;
    }

    spdlog::info("Backtest {} [{} to {}]", args.symbol,
                 trade::format_date(start), trade::format_date(end));

    // Simple buy-and-hold backtest from price series
    trade::BacktestResult result;
    result.strategy_name = args.strategy.empty() ? "buy_and_hold" : args.strategy;
    result.start_date = start;
    result.end_date = end;
    result.initial_capital = config.backtest.initial_capital;

    double capital = result.initial_capital, shares = 0, peak_nav = capital;
    for (const auto& b : bars) {
        if (b.date < start || b.date > end) continue;
        if (shares == 0 && b.open > 0) {
            shares = std::floor(capital / b.open / 100) * 100;
            capital -= shares * b.open;
        }
        double nav = capital + shares * b.close;
        peak_nav = std::max(peak_nav, nav);

        trade::DailyRecord rec;
        rec.date = b.date;
        rec.nav = nav;
        rec.cash = capital;
        rec.drawdown = (peak_nav - nav) / peak_nav;
        if (!result.daily_records.empty()) {
            double prev = result.daily_records.back().nav;
            rec.daily_return = prev > 0 ? (nav - prev) / prev : 0;
        }
        rec.cumulative_return = result.initial_capital > 0
            ? (nav / result.initial_capital - 1.0) : 0;
        result.daily_records.push_back(rec);
    }
    if (!result.daily_records.empty()) {
        result.final_nav = result.daily_records.back().nav;
        result.trading_days = static_cast<int>(result.daily_records.size());
    }

    trade::PerformanceCalculator calc;
    auto p = calc.compute(result);

    std::cout << "=== Backtest: " << result.strategy_name << " ===\n"
              << "Period:      " << trade::format_date(start) << " to "
              << trade::format_date(end) << " (" << result.trading_days << "d)\n"
              << "Capital:     " << std::fixed << std::setprecision(0)
              << result.initial_capital << " -> " << result.final_nav << "\n"
              << "Return:      " << std::showpos << std::setprecision(2)
              << (p.cumulative_return * 100) << "%\n"
              << "Ann return:  " << (p.annualised_return * 100) << "%\n"
              << std::noshowpos
              << "Sharpe:      " << std::setprecision(3) << p.sharpe_ratio << "\n"
              << "Sortino:     " << p.sortino_ratio << "\n"
              << "Calmar:      " << p.calmar_ratio << "\n"
              << "Max DD:      " << std::setprecision(2)
              << (p.max_drawdown * 100) << "%\n"
              << "DD duration: " << p.max_drawdown_duration << "d\n"
              << "VaR 95%:     " << (p.var_95 * 100) << "%\n"
              << "VaR 99%:     " << (p.var_99 * 100) << "%\n"
              << "Sharpe t:    " << std::setprecision(3) << p.sharpe_t_statistic << "\n"
              << "DSR:         " << p.deflated_sharpe_ratio << "\n";
    return 0;
}

// ============================================================================
// sentiment
// ============================================================================
int cmd_sentiment(const CliArgs& args, const trade::Config& config) {
    std::string src = args.source.empty() ? config.sentiment.default_source : args.source;
    spdlog::info("Sentiment (source: {})", src);

    if (src != "rss") {
        spdlog::error("CLI supports 'rss' source; '{}' requires API credentials", src);
        return 1;
    }

    trade::StoragePath paths(config.data.data_root);
    trade::MetadataStore metadata(paths.metadata_db());

    auto now = std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now());
    auto sentiment_default_start =
        now - std::chrono::days{std::max(1, config.sentiment.default_history_days)};
    auto [start, end] = resolve_dates(args, trade::format_date(sentiment_default_start));
    if (args.start_date.empty()) {
        auto wm = metadata.last_watermark_date(src, "sentiment_text", src);
        if (wm) {
            auto next = *wm - std::chrono::days{std::max(0, config.sentiment.incremental_lookback_days)};
            if (next > start) start = next;
        }
    }

    auto run_id = src + "_sent_" +
        std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    metadata.begin_ingestion_run(run_id, src, "sentiment_pipeline", src, "incremental");

    try {
        trade::RssSource rss;
        if (config.sentiment.rss_feeds.empty()) {
            spdlog::error("sentiment.rss_feeds is empty in config");
            return 1;
        }
        for (const auto& feed : config.sentiment.rss_feeds) {
            rss.add_feed(feed.url, feed.name);
        }

        auto events = rss.fetch_range(start, end);
        if (events.empty()) {
            metadata.finish_ingestion_run(run_id, true, 0, 0);
            std::cout << "No new sentiment events in range "
                      << trade::format_date(start) << " to " << trade::format_date(end)
                      << std::endl;
            return 0;
        }

        trade::TextCleaner cleaner;
        for (auto& ev : events) {
            ev.clean_text = cleaner.clean(ev.raw_text);
        }

        // Bronze write by date partition
        std::map<trade::Date, std::vector<trade::TextEvent>> bronze_by_date;
        for (const auto& ev : events) {
            auto d = std::chrono::floor<std::chrono::days>(ev.timestamp);
            bronze_by_date[d].push_back(ev);
        }
        for (auto& [d, day_events] : bronze_by_date) {
            auto ymd = std::chrono::year_month_day{d};
            int y = static_cast<int>(ymd.year());
            int m = static_cast<unsigned>(ymd.month());
            auto path = paths.sentiment_bronze(y, m, src, d);
            trade::ParquetStore::write_text_events(path, day_events,
                trade::ParquetStore::MergeMode::kMergeByKey, d);
        }

        trade::RuleSentiment model;
        if (!config.sentiment.dict_path.empty()) {
            model.load_dict(config.sentiment.dict_path);
        }

        trade::SymbolLinker linker;
        auto instruments = metadata.get_all_instruments();
        linker.build_index(instruments);

        struct Agg {
            double pos = 0.0;
            double neu = 0.0;
            double neg = 0.0;
            int count = 0;
        };
        std::unordered_map<std::string, Agg> agg;

        int pos_cnt = 0, neg_cnt = 0, neu_cnt = 0;
        for (const auto& ev : events) {
            const auto txt = ev.clean_text.empty() ? ev.raw_text : ev.clean_text;
            auto score = model.predict(txt);

            if (score.positive > score.neutral && score.positive > score.negative) ++pos_cnt;
            else if (score.negative > score.neutral && score.negative > score.positive) ++neg_cnt;
            else ++neu_cnt;

            auto day = std::chrono::floor<std::chrono::days>(ev.timestamp);
            std::vector<trade::Symbol> syms;
            if (!args.symbol.empty()) {
                syms.push_back(args.symbol);
            } else {
                syms = linker.link_symbols(ev.title + " " + txt);
            }
            if (syms.empty()) continue;

            for (const auto& sym : syms) {
                auto key = sym + "|" + trade::format_date(day) + "|" + src;
                auto& a = agg[key];
                a.pos += score.positive;
                a.neu += score.neutral;
                a.neg += score.negative;
                a.count += 1;
            }
        }

        std::vector<trade::NlpResult> nlp_results;
        nlp_results.reserve(agg.size());
        for (const auto& [key, a] : agg) {
            auto p1 = key.find('|');
            auto p2 = key.find('|', p1 + 1);
            if (p1 == std::string::npos || p2 == std::string::npos || a.count <= 0) continue;

            trade::NlpResult r;
            r.symbol = key.substr(0, p1);
            r.date = trade::parse_date(key.substr(p1 + 1, p2 - p1 - 1));
            r.source = key.substr(p2 + 1);
            r.sentiment.positive = a.pos / a.count;
            r.sentiment.neutral = a.neu / a.count;
            r.sentiment.negative = a.neg / a.count;
            r.article_count = a.count;
            nlp_results.push_back(std::move(r));
        }

        // Silver write by date partition
        std::map<trade::Date, std::vector<trade::NlpResult>> silver_by_date;
        for (const auto& r : nlp_results) {
            silver_by_date[r.date].push_back(r);
        }
        for (auto& [d, day_results] : silver_by_date) {
            auto ymd = std::chrono::year_month_day{d};
            int y = static_cast<int>(ymd.year());
            int m = static_cast<unsigned>(ymd.month());
            auto path = paths.sentiment_silver(y, m, d);
            trade::ParquetStore::write_nlp_results(path, day_results,
                trade::ParquetStore::MergeMode::kMergeByKey, d);
        }

        // Gold factors
        std::unordered_map<trade::Symbol, trade::BarSeries> bar_map;
        for (const auto& r : nlp_results) {
            if (!bar_map.count(r.symbol)) {
                trade::BarSeries series;
                series.symbol = r.symbol;
                series.bars = load_bars(r.symbol, config);
                bar_map[r.symbol] = std::move(series);
            }
        }

        trade::SentimentFactorCalculator calc;
        auto factors = calc.compute(nlp_results, bar_map);

        std::map<trade::Date, std::vector<trade::SentimentFactors>> gold_by_date;
        for (const auto& f : factors) {
            gold_by_date[f.date].push_back(f);
        }
        for (auto& [d, day_factors] : gold_by_date) {
            auto ymd = std::chrono::year_month_day{d};
            int y = static_cast<int>(ymd.year());
            int m = static_cast<unsigned>(ymd.month());
            auto path = paths.sentiment_gold(y, m, d);
            trade::ParquetStore::write_sentiment_factors(path, day_factors,
                trade::ParquetStore::MergeMode::kMergeByKey, d);
        }

        trade::Date max_date = start;
        for (const auto& [d, _] : bronze_by_date) {
            if (d > max_date) max_date = d;
        }
        metadata.upsert_watermark(src, "sentiment_text", src, max_date);

        int total = pos_cnt + neg_cnt + neu_cnt;
        std::cout << "=== Sentiment Incremental ===\n"
                  << "Range: " << trade::format_date(start) << " to " << trade::format_date(end) << "\n"
                  << "Events: " << events.size() << "\n"
                  << "NLP rows: " << nlp_results.size() << "\n"
                  << "Factor rows: " << factors.size() << "\n"
                  << "Summary: " << pos_cnt << " pos / " << neu_cnt << " neu / " << neg_cnt << " neg\n";
        if (total > 0) {
            std::cout << "Net sentiment: " << std::showpos << std::fixed << std::setprecision(3)
                      << (double(pos_cnt - neg_cnt) / total) << std::noshowpos << "\n";
        }

        metadata.finish_ingestion_run(run_id, true,
                                      static_cast<int64_t>(events.size()),
                                      static_cast<int64_t>(nlp_results.size() + factors.size()));
        return 0;
    } catch (const std::exception& e) {
        metadata.finish_ingestion_run(run_id, false, 0, 0, e.what());
        spdlog::error("Sentiment pipeline failed: {}", e.what());
        return 1;
    }
}


// ============================================================================
// report
// ============================================================================
int cmd_report(const CliArgs& args, const trade::Config& config) {
    if (args.symbol.empty()) { spdlog::error("--symbol required"); return 1; }
    auto bars = load_bars(args.symbol, config);
    if (bars.size() < 60) { spdlog::error("Need >=60 bars"); return 1; }

    std::vector<double> rets;
    for (size_t i = 1; i < bars.size(); ++i)
        if (bars[i - 1].close > 0)
            rets.push_back((bars[i].close - bars[i - 1].close) / bars[i - 1].close);

    double mr = 0, vol = 0;
    for (double r : rets) mr += r;
    mr /= rets.size();
    for (double r : rets) vol += (r - mr) * (r - mr);
    vol = std::sqrt(vol / rets.size());

    Eigen::VectorXd mu_vec(1); mu_vec(0) = mr;
    Eigen::VectorXd sigma_vec(1); sigma_vec(0) = vol;
    Eigen::VectorXd conf_vec(1); conf_vec(0) = 1.0;

    trade::KellyCalculator kelly;
    auto k = kelly.compute_diagnostics(mu_vec, sigma_vec, conf_vec);

    trade::RegimeDetector detector;
    std::vector<double> prices;
    for (const auto& b : bars) prices.push_back(b.close);
    trade::RegimeDetector::MarketBreadth breadth;
    breadth.total_stocks = 2500;
    breadth.up_stocks = 1500;
    auto regime = detector.update(prices, breadth);

    std::string regime_str = regime.regime_name();

    double full_kelly_val = k.raw_kelly.size() > 0 ? k.raw_kelly(0) : 0.0;
    double quarter_kelly_val = k.final_weights.size() > 0 ? k.final_weights(0) : 0.0;

    nlohmann::json rpt;
    rpt["ticker"]    = args.symbol;
    rpt["date"]      = trade::format_date(bars.back().date);
    rpt["close"]     = bars.back().close;
    rpt["regime"]    = regime_str;
    rpt["risk"] = {
        {"daily_vol", vol}, {"annual_vol", vol * std::sqrt(252)},
        {"mean_daily_return", mr},
        {"full_kelly", full_kelly_val}, {"quarter_kelly", quarter_kelly_val}
    };
    rpt["recommendation"] = {
        {"suggested_weight", quarter_kelly_val},
        {"confidence", 0.6}, {"regime", regime_str}
    };

    std::cout << "=== Report: " << args.symbol << " ===\n\n"
              << rpt.dump(2) << std::endl;

    if (!args.output.empty()) {
        std::ofstream ofs(args.output);
        ofs << rpt.dump(2) << std::endl;
        spdlog::info("Saved to {}", args.output);
    }
    return 0;
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    auto args = parse_args(argc, argv);
    if (args.command.empty() || args.command == "--help") { print_usage(); return 0; }

    auto console = spdlog::stdout_color_mt("console");
    spdlog::set_default_logger(console);
    spdlog::set_level(args.verbose ? spdlog::level::debug : spdlog::level::info);

    trade::Config config;
    // Try config relative to executable, then source dir, then CWD
    std::vector<std::string> config_search = {args.config_path};
#ifdef TRADE_SOURCE_DIR
    config_search.push_back(std::string(TRADE_SOURCE_DIR) + "/config/config.yaml");
#endif
    bool loaded = false;
    for (const auto& cp : config_search) {
        try {
            config = trade::Config::load(cp);
            loaded = true;
            break;
        } catch (...) {}
    }
    if (!loaded) {
        spdlog::debug("Config not found, using defaults");
        config = trade::Config::defaults();
    }

    // Resolve relative data_root to project source dir
    if (!config.data.data_root.empty() && config.data.data_root[0] != '/') {
#ifdef TRADE_SOURCE_DIR
        config.data.data_root = std::string(TRADE_SOURCE_DIR) + "/" + config.data.data_root;
#endif
    }
    trade::ParquetStore::configure_runtime(config.data, config.storage);
    trade::ParquetReader::configure_runtime(config.data, config.storage);

    try {
        if (args.command == "download")  return cmd_download(args, config);
        if (args.command == "verify")    return cmd_verify(args, config);
        if (args.command == "view") {
            spdlog::error("Command 'view' is paused. Use 'sql' for querying data.");
            return 1;
        }
        if (args.command == "sql")       return cmd_sql(args, config);
        if (args.command == "info") {
            spdlog::error("Command 'info' is paused. Use 'sql' for querying data.");
            return 1;
        }
        if (args.command == "features")  return cmd_features(args, config);
        if (args.command == "train")     return cmd_train(args, config);
        if (args.command == "predict")   return cmd_predict(args, config);
        if (args.command == "risk")      return cmd_risk(args, config);
        if (args.command == "backtest")  return cmd_backtest(args, config);
        if (args.command == "sentiment") return cmd_sentiment(args, config);
        if (args.command == "report")    return cmd_report(args, config);
        spdlog::error("Unknown command: {}", args.command);
        print_usage();
        return 1;
    } catch (const std::exception& e) {
        spdlog::error("Error: {}", e.what());
        return 1;
    }
}
