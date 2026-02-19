#include "trade/cli/shared.h"

#include "trade/common/time_utils.h"
#include "trade/storage/parquet_reader.h"
#include "trade/storage/storage_path.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>

namespace trade::cli {

std::pair<Date, Date> resolve_dates(const CliArgs& args,
                                    const std::string& default_start) {
    auto start = args.start_date.empty()
        ? parse_date(default_start)
        : parse_date(args.start_date);
    auto end = args.end_date.empty()
        ? std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now())
        : parse_date(args.end_date);
    return {start, end};
}

std::vector<Bar> load_bars(const std::string& symbol,
                           const Config& config) {
    StoragePath paths(config.data.data_root);
    std::vector<Bar> all_bars;
    const bool cloud_mode = config.storage.enabled &&
        (config.storage.backend == "baidu_netdisk" || config.storage.backend == "baidu");
    auto now = std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now());
    auto min_date = parse_date(config.ingestion.min_start_date);
    int start_year = date_year(min_date);
    int end_year = date_year(now);
    for (int year = start_year; year <= end_year; ++year) {
        const std::string raw_path = paths.raw_daily(symbol, year);
        const std::string silver_path = paths.silver_daily(symbol, year);
        std::string legacy_curated_path =
            (std::filesystem::path(config.data.data_root) / "curated" /
             config.data.market_daily_subpath / std::to_string(year) /
             (symbol + ".parquet"))
                .string();
        const std::vector<std::string> candidates = {
            raw_path,
            silver_path,
            legacy_curated_path,
        };
        for (const auto& path : candidates) {
            if (!cloud_mode && !std::filesystem::exists(path)) continue;
            try {
                auto bars = ParquetReader::read_bars(path);
                if (!bars.empty()) {
                    all_bars.insert(all_bars.end(), bars.begin(), bars.end());
                    break;
                }
            } catch (...) {
            }
        }
    }
    return all_bars;
}

std::string sql_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        out.push_back(c);
        if (c == '\'') out.push_back('\'');
    }
    return out;
}

namespace {

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

std::string sql_string(const std::string& s) {
    return "'" + sql_escape(s) + "'";
}

std::string sql_date_or_null(const std::optional<Date>& d) {
    if (!d) return "NULL";
    return sql_string(format_date(*d));
}

std::string build_values_view_sql(const std::string& view_name,
                                  const std::vector<std::string>& columns,
                                  const std::vector<std::vector<std::string>>& rows) {
    if (columns.empty()) return "";
    std::string out = "CREATE OR REPLACE VIEW " + view_name + "(";
    for (size_t i = 0; i < columns.size(); ++i) {
        if (i > 0) out += ", ";
        out += columns[i];
    }
    out += ") AS SELECT * FROM (VALUES ";

    if (rows.empty()) {
        out += "(";
        for (size_t i = 0; i < columns.size(); ++i) {
            if (i > 0) out += ", ";
            out += "NULL";
        }
        out += ")";
    } else {
        for (size_t i = 0; i < rows.size(); ++i) {
            if (i > 0) out += ", ";
            out += "(";
            const auto& row = rows[i];
            for (size_t j = 0; j < columns.size(); ++j) {
                if (j > 0) out += ", ";
                if (j < row.size()) out += row[j];
                else out += "NULL";
            }
            out += ")";
        }
    }

    out += ") AS t(";
    for (size_t i = 0; i < columns.size(); ++i) {
        if (i > 0) out += ", ";
        out += columns[i];
    }
    out += ")";
    if (rows.empty()) out += " WHERE 1=0";
    out += ";";
    return out;
}

} // namespace

std::vector<SqlViewDef> discover_sql_views(const Config& config) {
    StoragePath paths(config.data.data_root);
    MetadataStore metadata(paths.metadata_db());
    auto datasets = metadata.list_datasets();

    std::vector<SqlViewDef> views;
    views.reserve(datasets.size());
    for (const auto& ds : datasets) {
        if (ds.layer == "curated" || ds.dataset_id.rfind("curated.", 0) == 0) {
            continue;
        }
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

    add_fallback("raw.cn_a.daily",
                 std::filesystem::path(config.data.data_root) /
                 config.data.raw_dir / config.data.market_daily_subpath);
    add_fallback("silver.cn_a.daily",
                 std::filesystem::path(config.data.data_root) /
                 config.data.silver_dir / config.data.market_daily_subpath);
    add_fallback("silver.cn_a.daily",
                 std::filesystem::path(config.data.data_root) /
                 "curated" / config.data.market_daily_subpath);

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
    if (has_dataset("raw.cn_a.daily")) {
        init_sql += "CREATE OR REPLACE VIEW daily AS SELECT * FROM raw_cn_a_daily;";
    } else if (has_dataset("silver.cn_a.daily")) {
        init_sql += "CREATE OR REPLACE VIEW daily AS SELECT * FROM silver_cn_a_daily;";
    }
    if (has_dataset("raw.cn_a.daily")) {
        init_sql += "CREATE OR REPLACE VIEW raw AS SELECT * FROM raw_cn_a_daily;";
    }
    return init_sql;
}

std::string build_metadata_views_sql(const Config& config) {
    StoragePath paths(config.data.data_root);
    MetadataStore metadata(paths.metadata_db());
    auto datasets = metadata.list_datasets();

    std::vector<std::vector<std::string>> dataset_rows;
    dataset_rows.reserve(datasets.size());
    for (const auto& ds : datasets) {
        dataset_rows.push_back({
            sql_string(ds.dataset_id),
            sql_string(ds.layer),
            sql_string(ds.domain),
            sql_string(ds.data_type),
            sql_string(ds.path_prefix),
            std::to_string(ds.schema_version),
            sql_date_or_null(ds.latest_event_date),
        });
    }

    std::vector<std::vector<std::string>> file_rows;
    std::vector<std::vector<std::string>> schema_rows;
    std::vector<std::vector<std::string>> quality_rows;
    std::vector<std::vector<std::string>> tombstone_rows;
    std::vector<std::vector<std::string>> account_rows;
    std::vector<std::vector<std::string>> account_cash_rows;
    std::vector<std::vector<std::string>> account_position_rows;
    std::vector<std::vector<std::string>> account_trade_rows;
    for (const auto& ds : datasets) {
        auto files = metadata.list_dataset_files(ds.dataset_id);
        for (const auto& f : files) {
            file_rows.push_back({
                sql_string(f.dataset_id),
                sql_string(f.file_path),
                std::to_string(f.row_count),
                sql_date_or_null(f.max_event_date),
                std::to_string(f.current_version),
            });
        }

        auto active_schema = metadata.get_active_schema(ds.dataset_id);
        if (active_schema) {
            schema_rows.push_back({
                sql_string(active_schema->dataset_id),
                std::to_string(active_schema->schema_version),
                sql_string(active_schema->schema_hash),
                sql_string(active_schema->schema_json),
            });
        }
    }

    auto accounts = metadata.list_broker_accounts(false);
    for (const auto& a : accounts) {
        account_rows.push_back({
            sql_string(a.account_id),
            sql_string(a.broker),
            sql_string(a.account_name),
            std::to_string(a.is_active ? 1 : 0),
        });
        if (auto latest_cash = metadata.latest_account_cash(a.account_id)) {
            account_cash_rows.push_back({
                sql_string(latest_cash->account_id),
                sql_string(format_date(latest_cash->as_of_date)),
                std::to_string(latest_cash->total_asset),
                std::to_string(latest_cash->cash),
                std::to_string(latest_cash->available_cash),
                std::to_string(latest_cash->frozen_cash),
                std::to_string(latest_cash->market_value),
            });
        }
        for (const auto& p : metadata.latest_account_positions(a.account_id)) {
            account_position_rows.push_back({
                sql_string(p.account_id),
                sql_string(format_date(p.as_of_date)),
                sql_string(p.symbol),
                std::to_string(p.quantity),
                std::to_string(p.available_quantity),
                std::to_string(p.cost_price),
                std::to_string(p.last_price),
                std::to_string(p.market_value),
                std::to_string(p.unrealized_pnl),
                std::to_string(p.unrealized_pnl_ratio),
            });
        }
        for (const auto& t : metadata.list_account_trades(a.account_id, 200)) {
            account_trade_rows.push_back({
                sql_string(t.account_id),
                sql_string(t.trade_id),
                sql_string(format_date(t.trade_date)),
                sql_string(t.symbol),
                sql_string(t.side == Side::kSell ? "sell" : "buy"),
                std::to_string(t.price),
                std::to_string(t.quantity),
                std::to_string(t.amount),
                std::to_string(t.fee),
            });
        }
    }

    auto checks = metadata.list_quality_checks("", 200);
    for (const auto& c : checks) {
        quality_rows.push_back({
            sql_string(c.dataset_id),
            sql_string(c.check_name),
            sql_string(c.status),
            sql_string(c.severity),
            std::to_string(c.metric_value),
            std::to_string(c.threshold_value),
            sql_string(c.message),
            sql_date_or_null(c.event_date),
            sql_string(c.run_id),
        });
    }
    auto tombstones = metadata.list_dataset_tombstones("", 200);
    for (const auto& t : tombstones) {
        tombstone_rows.push_back({
            sql_string(t.dataset_id),
            sql_string(t.file_path),
            std::to_string(t.version),
            sql_string(t.reason),
            sql_date_or_null(t.max_event_date),
        });
    }

    std::string sql;
    sql += build_values_view_sql(
        "meta_dataset_catalog",
        {"dataset_id", "layer", "domain", "data_type", "path_prefix", "schema_version",
         "latest_event_date"},
        dataset_rows);
    sql += build_values_view_sql(
        "meta_dataset_files",
        {"dataset_id", "file_path", "row_count", "max_event_date", "current_version"},
        file_rows);
    sql += build_values_view_sql(
        "meta_schema_registry",
        {"dataset_id", "schema_version", "schema_hash", "schema_json"},
        schema_rows);
    sql += build_values_view_sql(
        "meta_quality_checks_recent",
        {"dataset_id", "check_name", "status", "severity", "metric_value", "threshold_value",
         "message", "event_date", "run_id"},
        quality_rows);
    sql += build_values_view_sql(
        "meta_dataset_tombstones_recent",
        {"dataset_id", "file_path", "version", "reason", "max_event_date"},
        tombstone_rows);
    sql += build_values_view_sql(
        "meta_accounts",
        {"account_id", "broker", "account_name", "is_active"},
        account_rows);
    sql += build_values_view_sql(
        "meta_account_cash",
        {"account_id", "as_of_date", "total_asset", "cash", "available_cash", "frozen_cash",
         "market_value"},
        account_cash_rows);
    sql += build_values_view_sql(
        "meta_account_positions",
        {"account_id", "as_of_date", "symbol", "quantity", "available_quantity", "cost_price",
         "last_price", "market_value", "unrealized_pnl", "unrealized_pnl_ratio"},
        account_position_rows);
    sql += build_values_view_sql(
        "meta_account_trades",
        {"account_id", "trade_id", "trade_date", "symbol", "side", "price", "quantity",
         "amount", "fee"},
        account_trade_rows);
    return sql;
}

MetadataHealth assess_metadata_health(MetadataStore& metadata) {
    MetadataHealth h;
    auto datasets = metadata.list_datasets();
    h.dataset_count = datasets.size();
    for (const auto& ds : datasets) {
        auto files = metadata.list_dataset_files(ds.dataset_id);
        if (!files.empty()) ++h.dataset_with_files;

        auto schema = metadata.get_active_schema(ds.dataset_id);
        if (schema && schema->schema_version == ds.schema_version) {
            ++h.dataset_schema_match;
        }

        for (const auto& f : files) {
            ++h.file_total;
            auto versions = metadata.list_dataset_file_versions(ds.dataset_id, f.file_path);
            if (!versions.empty()) {
                if (versions.front().version == f.current_version) {
                    ++h.file_version_covered;
                }
            }
        }
    }

    h.ok = h.dataset_count > 0 &&
           h.dataset_with_files == h.dataset_count &&
           h.dataset_schema_match == h.dataset_count;
    return h;
}

} // namespace trade::cli
