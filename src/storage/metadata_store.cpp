#include "trade/storage/metadata_store.h"
#include "trade/common/time_utils.h"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <sqlite3.h>
#include <stdexcept>

namespace trade {

namespace {

std::optional<Date> read_date_column(sqlite3_stmt* stmt, int col) {
    auto txt = sqlite3_column_text(stmt, col);
    if (!txt) return std::nullopt;
    return parse_date(reinterpret_cast<const char*>(txt));
}

void bind_date_or_null(sqlite3_stmt* stmt, int col, std::optional<Date> d) {
    if (d) {
        std::string v = format_date(*d);
        sqlite3_bind_text(stmt, col, v.c_str(), -1, SQLITE_TRANSIENT);
    } else {
        sqlite3_bind_null(stmt, col);
    }
}

int clamp_limit(int limit) {
    if (limit <= 0) return 100;
    return std::min(limit, 10000);
}

std::string normalize_side(std::string v) {
    std::transform(v.begin(), v.end(), v.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return v;
}

const char* side_to_text(Side side) {
    return side == Side::kSell ? "sell" : "buy";
}

Side side_from_text(const char* txt) {
    if (!txt) return Side::kBuy;
    const std::string s = normalize_side(txt);
    return s == "sell" ? Side::kSell : Side::kBuy;
}

} // namespace

struct MetadataStore::Impl {
    sqlite3* db = nullptr;

    ~Impl() {
        if (db) sqlite3_close(db);
    }

    void exec(const std::string& sql) {
        char* err = nullptr;
        int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &err);
        if (rc != SQLITE_OK) {
            std::string msg = err ? err : "unknown error";
            sqlite3_free(err);
            throw std::runtime_error("SQL error: " + msg);
        }
    }
};

MetadataStore::MetadataStore(const std::string& db_path) : impl_(std::make_unique<Impl>()) {
    auto parent = std::filesystem::path(db_path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    int rc = sqlite3_open(db_path.c_str(), &impl_->db);
    if (rc != SQLITE_OK) {
        throw std::runtime_error("Failed to open database: " + db_path);
    }

    impl_->exec(R"(
        CREATE TABLE IF NOT EXISTS instruments (
            symbol TEXT PRIMARY KEY,
            name TEXT,
            market INTEGER,
            board INTEGER,
            industry INTEGER,
            list_date TEXT,
            delist_date TEXT,
            status INTEGER,
            total_shares INTEGER DEFAULT 0,
            float_shares INTEGER DEFAULT 0
        )
    )");

    // Schema migration: add columns if missing (for existing DBs)
    sqlite3_exec(impl_->db,
                 "ALTER TABLE instruments ADD COLUMN total_shares INTEGER DEFAULT 0",
                 nullptr, nullptr, nullptr);
    sqlite3_exec(impl_->db,
                 "ALTER TABLE instruments ADD COLUMN float_shares INTEGER DEFAULT 0",
                 nullptr, nullptr, nullptr);

    impl_->exec(R"(
        CREATE TABLE IF NOT EXISTS downloads (
            symbol TEXT,
            start_date TEXT,
            end_date TEXT,
            row_count INTEGER,
            downloaded_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, end_date)
        )
    )");

    impl_->exec(R"(
        CREATE TABLE IF NOT EXISTS watermarks (
            source TEXT NOT NULL,
            dataset TEXT NOT NULL,
            symbol TEXT NOT NULL,
            last_event_date TEXT NOT NULL,
            cursor_payload TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (source, dataset, symbol)
        )
    )");

    impl_->exec(R"(
        CREATE TABLE IF NOT EXISTS ingestion_runs (
            run_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            dataset TEXT NOT NULL,
            symbol TEXT NOT NULL,
            mode TEXT NOT NULL,
            start_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            end_time TEXT,
            status TEXT NOT NULL,
            rows_in INTEGER NOT NULL DEFAULT 0,
            rows_out INTEGER NOT NULL DEFAULT 0,
            error TEXT NOT NULL DEFAULT ''
        )
    )");

    impl_->exec(R"(
        CREATE TABLE IF NOT EXISTS request_fingerprints (
            source TEXT NOT NULL,
            dataset TEXT NOT NULL,
            symbol TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT '',
            run_id TEXT NOT NULL DEFAULT '',
            rows_out INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (source, dataset, symbol, start_date, end_date)
        )
    )");

    impl_->exec(R"(
        CREATE TABLE IF NOT EXISTS holidays (
            date TEXT PRIMARY KEY,
            year INTEGER
        )
    )");

    impl_->exec(R"(
        CREATE TABLE IF NOT EXISTS dataset_catalog (
            dataset_id TEXT PRIMARY KEY,
            layer TEXT NOT NULL,
            domain TEXT NOT NULL,
            data_type TEXT NOT NULL,
            path_prefix TEXT NOT NULL,
            schema_version INTEGER NOT NULL DEFAULT 1,
            latest_event_date TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    )");

    impl_->exec(R"(
        CREATE TABLE IF NOT EXISTS dataset_files (
            dataset_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            row_count INTEGER NOT NULL DEFAULT 0,
            max_event_date TEXT,
            current_version INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (dataset_id, file_path)
        )
    )");

    // Schema registry for dataset-level schema versioning.
    impl_->exec(R"(
        CREATE TABLE IF NOT EXISTS schema_registry (
            dataset_id TEXT NOT NULL,
            schema_version INTEGER NOT NULL,
            schema_json TEXT NOT NULL DEFAULT '',
            schema_hash TEXT NOT NULL DEFAULT '',
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (dataset_id, schema_version)
        )
    )");

    // Per-write file versions (MVCC-style history for replay/audit).
    impl_->exec(R"(
        CREATE TABLE IF NOT EXISTS dataset_file_versions (
            dataset_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            version INTEGER NOT NULL,
            row_count INTEGER NOT NULL DEFAULT 0,
            max_event_date TEXT,
            run_id TEXT NOT NULL DEFAULT '',
            content_hash TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (dataset_id, file_path, version)
        )
    )");

    // Soft-delete records (tombstones) for lifecycle / recovery workflow.
    impl_->exec(R"(
        CREATE TABLE IF NOT EXISTS dataset_tombstones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 0,
            reason TEXT NOT NULL DEFAULT '',
            max_event_date TEXT,
            deleted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    )");

    // Data-quality events (checks, thresholds, pass/fail).
    impl_->exec(R"(
        CREATE TABLE IF NOT EXISTS quality_checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            check_name TEXT NOT NULL,
            status TEXT NOT NULL,
            severity TEXT NOT NULL,
            metric_value REAL NOT NULL DEFAULT 0.0,
            threshold_value REAL NOT NULL DEFAULT 0.0,
            message TEXT NOT NULL DEFAULT '',
            event_date TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    )");

    // Stream/hf checkpoints to support realtime incremental pulls.
    impl_->exec(R"(
        CREATE TABLE IF NOT EXISTS stream_checkpoints (
            source TEXT NOT NULL,
            stream TEXT NOT NULL,
            shard TEXT NOT NULL,
            cursor_payload TEXT NOT NULL DEFAULT '{}',
            last_event_date TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (source, stream, shard)
        )
    )");

    // Training snapshot metadata for reproducibility.
    impl_->exec(R"(
        CREATE TABLE IF NOT EXISTS training_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            query_spec TEXT NOT NULL DEFAULT '',
            snapshot_path TEXT NOT NULL DEFAULT '',
            start_date TEXT,
            end_date TEXT,
            row_count INTEGER NOT NULL DEFAULT 0,
            schema_version INTEGER NOT NULL DEFAULT 1,
            model_name TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    )");

    // Broker account registry + snapshots.
    impl_->exec(R"(
        CREATE TABLE IF NOT EXISTS broker_accounts (
            account_id TEXT PRIMARY KEY,
            broker TEXT NOT NULL,
            account_name TEXT NOT NULL DEFAULT '',
            auth_payload TEXT NOT NULL DEFAULT '',
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    )");

    impl_->exec(R"(
        CREATE TABLE IF NOT EXISTS account_cash_snapshots (
            account_id TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            total_asset REAL NOT NULL DEFAULT 0.0,
            cash REAL NOT NULL DEFAULT 0.0,
            available_cash REAL NOT NULL DEFAULT 0.0,
            frozen_cash REAL NOT NULL DEFAULT 0.0,
            market_value REAL NOT NULL DEFAULT 0.0,
            source TEXT NOT NULL DEFAULT 'manual',
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (account_id, as_of_date)
        )
    )");

    impl_->exec(R"(
        CREATE TABLE IF NOT EXISTS account_position_snapshots (
            account_id TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            quantity INTEGER NOT NULL DEFAULT 0,
            available_quantity INTEGER NOT NULL DEFAULT 0,
            cost_price REAL NOT NULL DEFAULT 0.0,
            last_price REAL NOT NULL DEFAULT 0.0,
            market_value REAL NOT NULL DEFAULT 0.0,
            unrealized_pnl REAL NOT NULL DEFAULT 0.0,
            unrealized_pnl_ratio REAL NOT NULL DEFAULT 0.0,
            source TEXT NOT NULL DEFAULT 'manual',
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (account_id, as_of_date, symbol)
        )
    )");

    impl_->exec(R"(
        CREATE TABLE IF NOT EXISTS account_trades (
            account_id TEXT NOT NULL,
            trade_id TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            price REAL NOT NULL DEFAULT 0.0,
            quantity INTEGER NOT NULL DEFAULT 0,
            amount REAL NOT NULL DEFAULT 0.0,
            fee REAL NOT NULL DEFAULT 0.0,
            source TEXT NOT NULL DEFAULT 'manual',
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (account_id, trade_id)
        )
    )");

    // Existing DB migration: add current_version if missing.
    sqlite3_exec(impl_->db,
                 "ALTER TABLE dataset_files ADD COLUMN current_version INTEGER NOT NULL DEFAULT 1",
                 nullptr, nullptr, nullptr);

    impl_->exec("CREATE INDEX IF NOT EXISTS idx_downloads_symbol_end ON downloads(symbol, end_date)");
    impl_->exec("CREATE INDEX IF NOT EXISTS idx_watermarks_lookup ON watermarks(source, dataset, symbol)");
    impl_->exec("CREATE INDEX IF NOT EXISTS idx_request_fingerprints_lookup ON request_fingerprints(source, dataset, symbol, updated_at)");
    impl_->exec("CREATE INDEX IF NOT EXISTS idx_dataset_files_dataset ON dataset_files(dataset_id)");
    impl_->exec("CREATE INDEX IF NOT EXISTS idx_schema_registry_active ON schema_registry(dataset_id, is_active)");
    impl_->exec("CREATE INDEX IF NOT EXISTS idx_quality_checks_dataset_time ON quality_checks(dataset_id, created_at)");
    impl_->exec("CREATE INDEX IF NOT EXISTS idx_stream_checkpoints_lookup ON stream_checkpoints(source, stream, shard)");
    impl_->exec("CREATE INDEX IF NOT EXISTS idx_training_snapshots_dataset ON training_snapshots(dataset_id, created_at)");
    impl_->exec("CREATE INDEX IF NOT EXISTS idx_dataset_file_versions_lookup ON dataset_file_versions(dataset_id, file_path, version)");
    impl_->exec("CREATE INDEX IF NOT EXISTS idx_dataset_tombstones_lookup ON dataset_tombstones(dataset_id, deleted_at)");
    impl_->exec("CREATE INDEX IF NOT EXISTS idx_broker_accounts_active ON broker_accounts(is_active, broker)");
    impl_->exec("CREATE INDEX IF NOT EXISTS idx_account_cash_lookup ON account_cash_snapshots(account_id, as_of_date)");
    impl_->exec("CREATE INDEX IF NOT EXISTS idx_account_position_lookup ON account_position_snapshots(account_id, as_of_date, symbol)");
    impl_->exec("CREATE INDEX IF NOT EXISTS idx_account_trades_lookup ON account_trades(account_id, trade_date)");

    // Backfill schema registry for legacy records.
    impl_->exec(R"(
        INSERT OR IGNORE INTO schema_registry
        (dataset_id, schema_version, schema_json, schema_hash, is_active, created_at)
        SELECT dataset_id,
               schema_version,
               '',
               'legacy_v' || schema_version,
               1,
               CURRENT_TIMESTAMP
          FROM dataset_catalog
    )");

    spdlog::debug("MetadataStore initialized at {}", db_path);
}

MetadataStore::~MetadataStore() = default;

void MetadataStore::upsert_instrument(const Instrument& inst) {
    const char* sql = R"(
        INSERT OR REPLACE INTO instruments (symbol, name, market, board, industry, list_date, delist_date, status, total_shares, float_shares)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    )";
    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        spdlog::error("Failed to prepare upsert_instrument: {}", sqlite3_errmsg(impl_->db));
        return;
    }

    sqlite3_bind_text(stmt, 1, inst.symbol.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, inst.name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 3, static_cast<int>(inst.market));
    sqlite3_bind_int(stmt, 4, static_cast<int>(inst.board));
    sqlite3_bind_int(stmt, 5, static_cast<int>(inst.industry));

    std::string list_str = format_date(inst.list_date);
    sqlite3_bind_text(stmt, 6, list_str.c_str(), -1, SQLITE_TRANSIENT);

    if (inst.delist_date) {
        std::string delist_str = format_date(*inst.delist_date);
        sqlite3_bind_text(stmt, 7, delist_str.c_str(), -1, SQLITE_TRANSIENT);
    } else {
        sqlite3_bind_null(stmt, 7);
    }
    sqlite3_bind_int(stmt, 8, static_cast<int>(inst.status));
    sqlite3_bind_int64(stmt, 9, inst.total_shares);
    sqlite3_bind_int64(stmt, 10, inst.float_shares);

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        spdlog::error("Failed to upsert instrument {}: {}", inst.symbol, sqlite3_errmsg(impl_->db));
    }
    sqlite3_finalize(stmt);
}

std::optional<Instrument> MetadataStore::get_instrument(const Symbol& symbol) {
    const char* sql = "SELECT * FROM instruments WHERE symbol = ?";
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, symbol.c_str(), -1, SQLITE_TRANSIENT);

    if (sqlite3_step(stmt) == SQLITE_ROW) {
        Instrument inst;
        inst.symbol = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        inst.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        inst.market = static_cast<Market>(sqlite3_column_int(stmt, 2));
        inst.board = static_cast<Board>(sqlite3_column_int(stmt, 3));
        inst.industry = static_cast<SWIndustry>(sqlite3_column_int(stmt, 4));
        inst.list_date = parse_date(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5)));
        auto delist = sqlite3_column_text(stmt, 6);
        if (delist) inst.delist_date = parse_date(reinterpret_cast<const char*>(delist));
        inst.status = static_cast<TradingStatus>(sqlite3_column_int(stmt, 7));
        inst.total_shares = sqlite3_column_int64(stmt, 8);
        inst.float_shares = sqlite3_column_int64(stmt, 9);
        sqlite3_finalize(stmt);
        return inst;
    }
    sqlite3_finalize(stmt);
    return std::nullopt;
}

std::vector<Instrument> MetadataStore::get_all_instruments() {
    std::vector<Instrument> result;
    const char* sql = "SELECT * FROM instruments";
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Instrument inst;
        inst.symbol = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        inst.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        inst.market = static_cast<Market>(sqlite3_column_int(stmt, 2));
        inst.board = static_cast<Board>(sqlite3_column_int(stmt, 3));
        inst.industry = static_cast<SWIndustry>(sqlite3_column_int(stmt, 4));
        inst.list_date = parse_date(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5)));
        auto delist = sqlite3_column_text(stmt, 6);
        if (delist) inst.delist_date = parse_date(reinterpret_cast<const char*>(delist));
        inst.status = static_cast<TradingStatus>(sqlite3_column_int(stmt, 7));
        inst.total_shares = sqlite3_column_int64(stmt, 8);
        inst.float_shares = sqlite3_column_int64(stmt, 9);
        result.push_back(std::move(inst));
    }
    sqlite3_finalize(stmt);
    return result;
}

std::vector<Instrument> MetadataStore::get_instruments_by_market(Market market) {
    std::vector<Instrument> all = get_all_instruments();
    std::vector<Instrument> filtered;
    for (auto& inst : all) {
        if (inst.market == market) filtered.push_back(std::move(inst));
    }
    return filtered;
}

std::vector<Instrument> MetadataStore::get_instruments_by_industry(SWIndustry industry) {
    std::vector<Instrument> all = get_all_instruments();
    std::vector<Instrument> filtered;
    for (auto& inst : all) {
        if (inst.industry == industry) filtered.push_back(std::move(inst));
    }
    return filtered;
}

void MetadataStore::record_download(const Symbol& symbol, Date start, Date end,
                                    int64_t row_count) {
    const char* sql = R"(
        INSERT OR REPLACE INTO downloads (symbol, start_date, end_date, row_count)
        VALUES (?, ?, ?, ?)
    )";
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    std::string start_s = format_date(start);
    std::string end_s = format_date(end);
    sqlite3_bind_text(stmt, 1, symbol.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, start_s.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, end_s.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 4, row_count);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

std::optional<Date> MetadataStore::last_download_date(const Symbol& symbol) {
    const char* sql = "SELECT MAX(end_date) FROM downloads WHERE symbol = ?";
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, symbol.c_str(), -1, SQLITE_TRANSIENT);

    if (sqlite3_step(stmt) == SQLITE_ROW && sqlite3_column_text(stmt, 0)) {
        auto date = parse_date(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
        sqlite3_finalize(stmt);
        return date;
    }
    sqlite3_finalize(stmt);
    return std::nullopt;
}

std::vector<Symbol> MetadataStore::symbols_needing_update(Date cutoff) {
    std::vector<Symbol> result;
    std::string cutoff_str = format_date(cutoff);

    const char* sql = R"(
        SELECT i.symbol FROM instruments i
        LEFT JOIN (
            SELECT symbol, MAX(end_date) as last_date
            FROM downloads GROUP BY symbol
        ) d ON i.symbol = d.symbol
        WHERE d.last_date IS NULL OR d.last_date < ?
    )";
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, cutoff_str.c_str(), -1, SQLITE_TRANSIENT);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        result.emplace_back(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
    }
    sqlite3_finalize(stmt);
    return result;
}

void MetadataStore::upsert_watermark(const std::string& source,
                                     const std::string& dataset,
                                     const Symbol& symbol,
                                     Date last_event_date,
                                     const std::string& cursor_payload) {
    const char* sql = R"(
        INSERT INTO watermarks (source, dataset, symbol, last_event_date, cursor_payload, updated_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(source, dataset, symbol)
        DO UPDATE SET
            last_event_date = excluded.last_event_date,
            cursor_payload = excluded.cursor_payload,
            updated_at = CURRENT_TIMESTAMP
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare upsert_watermark: {}", sqlite3_errmsg(impl_->db));
        return;
    }

    std::string date_s = format_date(last_event_date);
    sqlite3_bind_text(stmt, 1, source.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, dataset.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, symbol.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, date_s.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 5, cursor_payload.c_str(), -1, SQLITE_TRANSIENT);

    if (sqlite3_step(stmt) != SQLITE_DONE) {
        spdlog::error("Failed to upsert watermark {}/{}/{}: {}",
                      source, dataset, symbol, sqlite3_errmsg(impl_->db));
    }
    sqlite3_finalize(stmt);
}

std::optional<Date> MetadataStore::last_watermark_date(const std::string& source,
                                                       const std::string& dataset,
                                                       const Symbol& symbol) {
    const char* sql = R"(
        SELECT last_event_date FROM watermarks
        WHERE source = ? AND dataset = ? AND symbol = ?
    )";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, source.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, dataset.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, symbol.c_str(), -1, SQLITE_TRANSIENT);

    std::optional<Date> out;
    if (sqlite3_step(stmt) == SQLITE_ROW && sqlite3_column_text(stmt, 0)) {
        out = parse_date(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
    }
    sqlite3_finalize(stmt);
    return out;
}

void MetadataStore::begin_ingestion_run(const std::string& run_id,
                                        const std::string& source,
                                        const std::string& dataset,
                                        const Symbol& symbol,
                                        const std::string& mode) {
    const char* sql = R"(
        INSERT OR REPLACE INTO ingestion_runs
        (run_id, source, dataset, symbol, mode, status, start_time)
        VALUES (?, ?, ?, ?, ?, 'running', CURRENT_TIMESTAMP)
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare begin_ingestion_run: {}", sqlite3_errmsg(impl_->db));
        return;
    }

    sqlite3_bind_text(stmt, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, source.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, dataset.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, symbol.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 5, mode.c_str(), -1, SQLITE_TRANSIENT);

    if (sqlite3_step(stmt) != SQLITE_DONE) {
        spdlog::error("Failed to begin ingestion run {}: {}", run_id, sqlite3_errmsg(impl_->db));
    }
    sqlite3_finalize(stmt);
}

void MetadataStore::finish_ingestion_run(const std::string& run_id,
                                         bool success,
                                         int64_t rows_in,
                                         int64_t rows_out,
                                         const std::string& error) {
    const char* sql = R"(
        UPDATE ingestion_runs
        SET end_time = CURRENT_TIMESTAMP,
            status = ?,
            rows_in = ?,
            rows_out = ?,
            error = ?
        WHERE run_id = ?
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare finish_ingestion_run: {}", sqlite3_errmsg(impl_->db));
        return;
    }

    const char* status = success ? "success" : "failed";
    sqlite3_bind_text(stmt, 1, status, -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 2, rows_in);
    sqlite3_bind_int64(stmt, 3, rows_out);
    sqlite3_bind_text(stmt, 4, error.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 5, run_id.c_str(), -1, SQLITE_TRANSIENT);

    if (sqlite3_step(stmt) != SQLITE_DONE) {
        spdlog::error("Failed to finish ingestion run {}: {}", run_id, sqlite3_errmsg(impl_->db));
    }
    sqlite3_finalize(stmt);
}

bool MetadataStore::has_recent_successful_request(const std::string& source,
                                                  const std::string& dataset,
                                                  const Symbol& symbol,
                                                  Date start_date,
                                                  Date end_date,
                                                  int within_hours) {
    if (within_hours <= 0) return false;

    const char* sql = R"(
        SELECT 1
        FROM request_fingerprints
        WHERE source = ?
          AND dataset = ?
          AND symbol = ?
          AND start_date = ?
          AND end_date = ?
          AND status = 'success'
          AND ((julianday('now') - julianday(updated_at)) * 24.0) <= ?
        LIMIT 1
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare has_recent_successful_request: {}", sqlite3_errmsg(impl_->db));
        return false;
    }

    const std::string start_s = format_date(start_date);
    const std::string end_s = format_date(end_date);
    sqlite3_bind_text(stmt, 1, source.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, dataset.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, symbol.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, start_s.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 5, end_s.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(stmt, 6, static_cast<double>(within_hours));

    const bool hit = (sqlite3_step(stmt) == SQLITE_ROW);
    sqlite3_finalize(stmt);
    return hit;
}

void MetadataStore::record_request_fingerprint(const std::string& source,
                                               const std::string& dataset,
                                               const Symbol& symbol,
                                               Date start_date,
                                               Date end_date,
                                               const std::string& status,
                                               const std::string& run_id,
                                               int64_t rows_out) {
    const char* sql = R"(
        INSERT INTO request_fingerprints
        (source, dataset, symbol, start_date, end_date, status, run_id, rows_out, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(source, dataset, symbol, start_date, end_date)
        DO UPDATE SET
            status = excluded.status,
            run_id = excluded.run_id,
            rows_out = excluded.rows_out,
            updated_at = CURRENT_TIMESTAMP
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare record_request_fingerprint: {}", sqlite3_errmsg(impl_->db));
        return;
    }

    const std::string start_s = format_date(start_date);
    const std::string end_s = format_date(end_date);
    sqlite3_bind_text(stmt, 1, source.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, dataset.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, symbol.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, start_s.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 5, end_s.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 6, status.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 7, run_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 8, rows_out);

    if (sqlite3_step(stmt) != SQLITE_DONE) {
        spdlog::error("Failed to record request fingerprint {}/{}/{} [{}-{}]: {}",
                      source, dataset, symbol, start_s, end_s, sqlite3_errmsg(impl_->db));
    }
    sqlite3_finalize(stmt);
}

void MetadataStore::load_holidays(const std::vector<Date>& holidays) {
    impl_->exec("BEGIN TRANSACTION");
    for (const auto& h : holidays) {
        std::string sql = "INSERT OR IGNORE INTO holidays (date, year) VALUES ('" +
                          format_date(h) + "', " + std::to_string(date_year(h)) + ")";
        impl_->exec(sql);
    }
    impl_->exec("COMMIT");
}

std::vector<Date> MetadataStore::get_holidays(int year) {
    std::vector<Date> result;
    const char* sql = "SELECT date FROM holidays WHERE year = ?";
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    sqlite3_bind_int(stmt, 1, year);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        result.push_back(parse_date(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0))));
    }
    sqlite3_finalize(stmt);
    return result;
}

int MetadataStore::upsert_dataset_file(const std::string& dataset_id,
                                       const std::string& layer,
                                       const std::string& domain,
                                       const std::string& data_type,
                                       const std::string& path_prefix,
                                       const std::string& file_path,
                                       int64_t row_count,
                                       std::optional<Date> max_event_date,
                                       int schema_version,
                                       const std::string& run_id,
                                       std::optional<int> forced_file_version,
                                       const std::string& content_hash) {
    int file_version = forced_file_version.value_or(1);
    if (!forced_file_version) {
        const char* sql_next = R"(
            SELECT COALESCE(MAX(version), 0) + 1
            FROM dataset_file_versions
            WHERE dataset_id = ? AND file_path = ?
        )";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(impl_->db, sql_next, -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, dataset_id.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 2, file_path.c_str(), -1, SQLITE_TRANSIENT);
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                file_version = sqlite3_column_int(stmt, 0);
            }
        }
        sqlite3_finalize(stmt);
    }

    const char* sql_catalog = R"(
        INSERT INTO dataset_catalog
        (dataset_id, layer, domain, data_type, path_prefix, schema_version, latest_event_date, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(dataset_id)
        DO UPDATE SET
            layer = excluded.layer,
            domain = excluded.domain,
            data_type = excluded.data_type,
            path_prefix = excluded.path_prefix,
            schema_version = excluded.schema_version,
            latest_event_date = CASE
                WHEN excluded.latest_event_date IS NULL THEN dataset_catalog.latest_event_date
                WHEN dataset_catalog.latest_event_date IS NULL THEN excluded.latest_event_date
                WHEN excluded.latest_event_date > dataset_catalog.latest_event_date THEN excluded.latest_event_date
                ELSE dataset_catalog.latest_event_date
            END,
            updated_at = CURRENT_TIMESTAMP
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql_catalog, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare upsert dataset_catalog: {}", sqlite3_errmsg(impl_->db));
        return file_version;
    }
    sqlite3_bind_text(stmt, 1, dataset_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, layer.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, domain.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, data_type.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 5, path_prefix.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 6, schema_version);
    bind_date_or_null(stmt, 7, max_event_date);
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        spdlog::error("Failed to upsert dataset catalog {}: {}",
                      dataset_id, sqlite3_errmsg(impl_->db));
    }
    sqlite3_finalize(stmt);

    const char* sql_file = R"(
        INSERT INTO dataset_files
        (dataset_id, file_path, row_count, max_event_date, current_version, updated_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(dataset_id, file_path)
        DO UPDATE SET
            row_count = excluded.row_count,
            max_event_date = CASE
                WHEN excluded.max_event_date IS NULL THEN dataset_files.max_event_date
                ELSE excluded.max_event_date
            END,
            current_version = excluded.current_version,
            updated_at = CURRENT_TIMESTAMP
    )";
    stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql_file, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare upsert dataset_files: {}", sqlite3_errmsg(impl_->db));
        return file_version;
    }
    sqlite3_bind_text(stmt, 1, dataset_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, file_path.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 3, row_count);
    bind_date_or_null(stmt, 4, max_event_date);
    sqlite3_bind_int(stmt, 5, file_version);
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        spdlog::error("Failed to upsert dataset file {} {}: {}",
                      dataset_id, file_path, sqlite3_errmsg(impl_->db));
    }
    sqlite3_finalize(stmt);

    const char* sql_version = R"(
        INSERT INTO dataset_file_versions
        (dataset_id, file_path, version, row_count, max_event_date, run_id, content_hash, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(dataset_id, file_path, version)
        DO UPDATE SET
            row_count = excluded.row_count,
            max_event_date = excluded.max_event_date,
            run_id = excluded.run_id,
            content_hash = excluded.content_hash,
            created_at = CURRENT_TIMESTAMP
    )";
    stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql_version, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare upsert dataset_file_versions: {}", sqlite3_errmsg(impl_->db));
        return file_version;
    }
    sqlite3_bind_text(stmt, 1, dataset_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, file_path.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 3, file_version);
    sqlite3_bind_int64(stmt, 4, row_count);
    bind_date_or_null(stmt, 5, max_event_date);
    sqlite3_bind_text(stmt, 6, run_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 7, content_hash.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        spdlog::error("Failed to upsert dataset file version {} {} v{}: {}",
                      dataset_id, file_path, file_version, sqlite3_errmsg(impl_->db));
    }
    sqlite3_finalize(stmt);

    return file_version;
}

std::vector<MetadataStore::DatasetRecord> MetadataStore::list_datasets() {
    std::vector<DatasetRecord> out;
    const char* sql = R"(
        SELECT dataset_id, layer, domain, data_type, path_prefix, schema_version, latest_event_date
        FROM dataset_catalog
        ORDER BY dataset_id
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare list_datasets: {}", sqlite3_errmsg(impl_->db));
        return out;
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        DatasetRecord r;
        r.dataset_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        r.layer = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        r.domain = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        r.data_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        r.path_prefix = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        r.schema_version = sqlite3_column_int(stmt, 5);
        auto date_txt = sqlite3_column_text(stmt, 6);
        if (date_txt) {
            r.latest_event_date = parse_date(reinterpret_cast<const char*>(date_txt));
        }
        out.push_back(std::move(r));
    }
    sqlite3_finalize(stmt);
    return out;
}

std::vector<MetadataStore::DatasetFileRecord> MetadataStore::list_dataset_files(
    const std::string& dataset_id) {
    std::vector<DatasetFileRecord> out;
    const char* sql = R"(
        SELECT dataset_id, file_path, row_count, max_event_date, current_version
        FROM dataset_files
        WHERE dataset_id = ?
        ORDER BY file_path
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare list_dataset_files: {}", sqlite3_errmsg(impl_->db));
        return out;
    }
    sqlite3_bind_text(stmt, 1, dataset_id.c_str(), -1, SQLITE_TRANSIENT);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        DatasetFileRecord r;
        r.dataset_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        r.file_path = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        r.row_count = sqlite3_column_int64(stmt, 2);
        r.max_event_date = read_date_column(stmt, 3);
        r.current_version = sqlite3_column_int(stmt, 4);
        out.push_back(std::move(r));
    }
    sqlite3_finalize(stmt);
    return out;
}

std::vector<MetadataStore::DatasetFileVersionRecord>
MetadataStore::list_dataset_file_versions(const std::string& dataset_id,
                                          const std::string& file_path) {
    std::vector<DatasetFileVersionRecord> out;
    const char* sql = R"(
        SELECT dataset_id, file_path, version, row_count, max_event_date, run_id, content_hash
        FROM dataset_file_versions
        WHERE dataset_id = ? AND file_path = ?
        ORDER BY version DESC
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare list_dataset_file_versions: {}", sqlite3_errmsg(impl_->db));
        return out;
    }
    sqlite3_bind_text(stmt, 1, dataset_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, file_path.c_str(), -1, SQLITE_TRANSIENT);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        DatasetFileVersionRecord r;
        r.dataset_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        r.file_path = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        r.version = sqlite3_column_int(stmt, 2);
        r.row_count = sqlite3_column_int64(stmt, 3);
        r.max_event_date = read_date_column(stmt, 4);
        r.run_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        r.content_hash = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));
        out.push_back(std::move(r));
    }
    sqlite3_finalize(stmt);
    return out;
}

void MetadataStore::delete_dataset_file(const std::string& dataset_id,
                                        const std::string& file_path,
                                        const std::string& reason) {
    impl_->exec("BEGIN TRANSACTION");
    try {
        int cur_version = 0;
        std::optional<Date> max_event_date;

        const char* sql_get = R"(
            SELECT current_version, max_event_date
            FROM dataset_files
            WHERE dataset_id = ? AND file_path = ?
        )";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(impl_->db, sql_get, -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, dataset_id.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 2, file_path.c_str(), -1, SQLITE_TRANSIENT);
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                cur_version = sqlite3_column_int(stmt, 0);
                max_event_date = read_date_column(stmt, 1);
            }
        }
        sqlite3_finalize(stmt);

        const char* sql_tomb = R"(
            INSERT INTO dataset_tombstones
            (dataset_id, file_path, version, reason, max_event_date, deleted_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        )";
        stmt = nullptr;
        if (sqlite3_prepare_v2(impl_->db, sql_tomb, -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, dataset_id.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 2, file_path.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_int(stmt, 3, cur_version);
            sqlite3_bind_text(stmt, 4, reason.c_str(), -1, SQLITE_TRANSIENT);
            bind_date_or_null(stmt, 5, max_event_date);
            (void)sqlite3_step(stmt);
        } else {
            spdlog::warn("Failed to prepare dataset_tombstones insert: {}",
                         sqlite3_errmsg(impl_->db));
        }
        sqlite3_finalize(stmt);

        const char* sql_del = R"(
            DELETE FROM dataset_files
            WHERE dataset_id = ? AND file_path = ?
        )";
        stmt = nullptr;
        if (sqlite3_prepare_v2(impl_->db, sql_del, -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, dataset_id.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 2, file_path.c_str(), -1, SQLITE_TRANSIENT);
            if (sqlite3_step(stmt) != SQLITE_DONE) {
                spdlog::error("Failed to delete dataset file {} {}: {}",
                              dataset_id, file_path, sqlite3_errmsg(impl_->db));
            }
        } else {
            spdlog::error("Failed to prepare delete dataset file: {}",
                          sqlite3_errmsg(impl_->db));
        }
        sqlite3_finalize(stmt);
        impl_->exec("COMMIT");
    } catch (...) {
        impl_->exec("ROLLBACK");
        throw;
    }
}

int MetadataStore::prune_empty_datasets() {
    const char* sql = R"(
        DELETE FROM dataset_catalog
        WHERE dataset_id NOT IN (
            SELECT DISTINCT dataset_id FROM dataset_files
        )
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare prune_empty_datasets: {}", sqlite3_errmsg(impl_->db));
        return 0;
    }
    int before = sqlite3_total_changes(impl_->db);
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        spdlog::error("Failed to prune empty datasets: {}", sqlite3_errmsg(impl_->db));
    }
    sqlite3_finalize(stmt);
    int after = sqlite3_total_changes(impl_->db);
    return std::max(0, after - before);
}

std::vector<MetadataStore::DatasetTombstoneRecord>
MetadataStore::list_dataset_tombstones(const std::string& dataset_id, int limit) {
    std::vector<DatasetTombstoneRecord> out;
    const char* sql = R"(
        SELECT dataset_id, file_path, version, reason, max_event_date
        FROM dataset_tombstones
        WHERE (? = '' OR dataset_id = ?)
        ORDER BY id DESC
        LIMIT ?
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare list_dataset_tombstones: {}", sqlite3_errmsg(impl_->db));
        return out;
    }
    sqlite3_bind_text(stmt, 1, dataset_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, dataset_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 3, clamp_limit(limit));

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        DatasetTombstoneRecord r;
        r.dataset_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        r.file_path = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        r.version = sqlite3_column_int(stmt, 2);
        r.reason = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        r.max_event_date = read_date_column(stmt, 4);
        out.push_back(std::move(r));
    }
    sqlite3_finalize(stmt);
    return out;
}

void MetadataStore::upsert_schema(const std::string& dataset_id,
                                  int schema_version,
                                  const std::string& schema_json,
                                  const std::string& schema_hash,
                                  bool set_active) {
    const char* sql = R"(
        INSERT INTO schema_registry
        (dataset_id, schema_version, schema_json, schema_hash, is_active, created_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(dataset_id, schema_version)
        DO UPDATE SET
            schema_json = excluded.schema_json,
            schema_hash = excluded.schema_hash,
            is_active = excluded.is_active
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare upsert_schema: {}", sqlite3_errmsg(impl_->db));
        return;
    }
    sqlite3_bind_text(stmt, 1, dataset_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 2, schema_version);
    sqlite3_bind_text(stmt, 3, schema_json.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, schema_hash.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 5, set_active ? 1 : 0);
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        spdlog::error("Failed to upsert schema {} v{}: {}",
                      dataset_id, schema_version, sqlite3_errmsg(impl_->db));
    }
    sqlite3_finalize(stmt);

    if (!set_active) return;
    const char* sql_deactivate = R"(
        UPDATE schema_registry
        SET is_active = 0
        WHERE dataset_id = ? AND schema_version <> ?
    )";
    stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql_deactivate, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, dataset_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 2, schema_version);
        (void)sqlite3_step(stmt);
    }
    sqlite3_finalize(stmt);
}

std::optional<MetadataStore::SchemaRecord>
MetadataStore::get_active_schema(const std::string& dataset_id) {
    const char* sql = R"(
        SELECT dataset_id, schema_version, schema_json, schema_hash, is_active
        FROM schema_registry
        WHERE dataset_id = ? AND is_active = 1
        ORDER BY schema_version DESC
        LIMIT 1
    )";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, dataset_id.c_str(), -1, SQLITE_TRANSIENT);

    std::optional<SchemaRecord> out;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        SchemaRecord r;
        r.dataset_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        r.schema_version = sqlite3_column_int(stmt, 1);
        r.schema_json = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        r.schema_hash = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        r.is_active = sqlite3_column_int(stmt, 4) != 0;
        out = std::move(r);
    }
    sqlite3_finalize(stmt);
    return out;
}

std::optional<int> MetadataStore::find_schema_version_by_hash(const std::string& dataset_id,
                                                              const std::string& schema_hash) {
    const char* sql = R"(
        SELECT schema_version
        FROM schema_registry
        WHERE dataset_id = ? AND schema_hash = ?
        ORDER BY schema_version DESC
        LIMIT 1
    )";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, dataset_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, schema_hash.c_str(), -1, SQLITE_TRANSIENT);

    std::optional<int> out;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        out = sqlite3_column_int(stmt, 0);
    }
    sqlite3_finalize(stmt);
    return out;
}

void MetadataStore::record_quality_check(const QualityCheckRecord& check) {
    const char* sql = R"(
        INSERT INTO quality_checks
        (run_id, dataset_id, check_name, status, severity, metric_value, threshold_value, message, event_date, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare record_quality_check: {}", sqlite3_errmsg(impl_->db));
        return;
    }
    sqlite3_bind_text(stmt, 1, check.run_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, check.dataset_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, check.check_name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, check.status.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 5, check.severity.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(stmt, 6, check.metric_value);
    sqlite3_bind_double(stmt, 7, check.threshold_value);
    sqlite3_bind_text(stmt, 8, check.message.c_str(), -1, SQLITE_TRANSIENT);
    bind_date_or_null(stmt, 9, check.event_date);
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        spdlog::error("Failed to record quality check {} {}: {}",
                      check.dataset_id, check.check_name, sqlite3_errmsg(impl_->db));
    }
    sqlite3_finalize(stmt);
}

std::vector<MetadataStore::QualityCheckRecord>
MetadataStore::list_quality_checks(const std::string& dataset_id, int limit) {
    std::vector<QualityCheckRecord> out;
    const char* sql = R"(
        SELECT run_id, dataset_id, check_name, status, severity,
               metric_value, threshold_value, message, event_date
        FROM quality_checks
        WHERE (? = '' OR dataset_id = ?)
        ORDER BY id DESC
        LIMIT ?
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare list_quality_checks: {}", sqlite3_errmsg(impl_->db));
        return out;
    }
    sqlite3_bind_text(stmt, 1, dataset_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, dataset_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 3, clamp_limit(limit));

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        QualityCheckRecord r;
        r.run_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        r.dataset_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        r.check_name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        r.status = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        r.severity = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        r.metric_value = sqlite3_column_double(stmt, 5);
        r.threshold_value = sqlite3_column_double(stmt, 6);
        r.message = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7));
        r.event_date = read_date_column(stmt, 8);
        out.push_back(std::move(r));
    }
    sqlite3_finalize(stmt);
    return out;
}

void MetadataStore::upsert_stream_checkpoint(const std::string& source,
                                             const std::string& stream,
                                             const std::string& shard,
                                             const std::string& cursor_payload,
                                             std::optional<Date> last_event_date) {
    const char* sql = R"(
        INSERT INTO stream_checkpoints
        (source, stream, shard, cursor_payload, last_event_date, updated_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(source, stream, shard)
        DO UPDATE SET
            cursor_payload = excluded.cursor_payload,
            last_event_date = excluded.last_event_date,
            updated_at = CURRENT_TIMESTAMP
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare upsert_stream_checkpoint: {}", sqlite3_errmsg(impl_->db));
        return;
    }
    sqlite3_bind_text(stmt, 1, source.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, stream.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, shard.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, cursor_payload.c_str(), -1, SQLITE_TRANSIENT);
    bind_date_or_null(stmt, 5, last_event_date);

    if (sqlite3_step(stmt) != SQLITE_DONE) {
        spdlog::error("Failed to upsert stream checkpoint {}/{}/{}: {}",
                      source, stream, shard, sqlite3_errmsg(impl_->db));
    }
    sqlite3_finalize(stmt);
}

std::optional<MetadataStore::StreamCheckpointRecord>
MetadataStore::get_stream_checkpoint(const std::string& source,
                                     const std::string& stream,
                                     const std::string& shard) {
    const char* sql = R"(
        SELECT source, stream, shard, cursor_payload, last_event_date
        FROM stream_checkpoints
        WHERE source = ? AND stream = ? AND shard = ?
    )";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, source.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, stream.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, shard.c_str(), -1, SQLITE_TRANSIENT);

    std::optional<StreamCheckpointRecord> out;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        StreamCheckpointRecord r;
        r.source = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        r.stream = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        r.shard = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        r.cursor_payload = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        r.last_event_date = read_date_column(stmt, 4);
        out = std::move(r);
    }
    sqlite3_finalize(stmt);
    return out;
}

void MetadataStore::record_training_snapshot(const TrainingSnapshotRecord& snapshot) {
    const char* sql = R"(
        INSERT OR REPLACE INTO training_snapshots
        (snapshot_id, dataset_id, query_spec, snapshot_path, start_date, end_date,
         row_count, schema_version, model_name, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare record_training_snapshot: {}", sqlite3_errmsg(impl_->db));
        return;
    }
    sqlite3_bind_text(stmt, 1, snapshot.snapshot_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, snapshot.dataset_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, snapshot.query_spec.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, snapshot.snapshot_path.c_str(), -1, SQLITE_TRANSIENT);
    bind_date_or_null(stmt, 5, snapshot.start_date);
    bind_date_or_null(stmt, 6, snapshot.end_date);
    sqlite3_bind_int64(stmt, 7, snapshot.row_count);
    sqlite3_bind_int(stmt, 8, snapshot.schema_version);
    sqlite3_bind_text(stmt, 9, snapshot.model_name.c_str(), -1, SQLITE_TRANSIENT);

    if (sqlite3_step(stmt) != SQLITE_DONE) {
        spdlog::error("Failed to record training snapshot {}: {}",
                      snapshot.snapshot_id, sqlite3_errmsg(impl_->db));
    }
    sqlite3_finalize(stmt);
}

std::vector<MetadataStore::TrainingSnapshotRecord>
MetadataStore::list_training_snapshots(const std::string& dataset_id, int limit) {
    std::vector<TrainingSnapshotRecord> out;
    const char* sql = R"(
        SELECT snapshot_id, dataset_id, query_spec, snapshot_path,
               start_date, end_date, row_count, schema_version, model_name
        FROM training_snapshots
        WHERE dataset_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare list_training_snapshots: {}", sqlite3_errmsg(impl_->db));
        return out;
    }
    sqlite3_bind_text(stmt, 1, dataset_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 2, clamp_limit(limit));

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        TrainingSnapshotRecord r;
        r.snapshot_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        r.dataset_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        r.query_spec = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        r.snapshot_path = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        r.start_date = read_date_column(stmt, 4);
        r.end_date = read_date_column(stmt, 5);
        r.row_count = sqlite3_column_int64(stmt, 6);
        r.schema_version = sqlite3_column_int(stmt, 7);
        r.model_name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 8));
        out.push_back(std::move(r));
    }
    sqlite3_finalize(stmt);
    return out;
}

void MetadataStore::upsert_broker_account(const BrokerAccount& account) {
    const char* sql = R"(
        INSERT INTO broker_accounts
        (account_id, broker, account_name, auth_payload, is_active, updated_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(account_id)
        DO UPDATE SET
            broker = excluded.broker,
            account_name = excluded.account_name,
            auth_payload = excluded.auth_payload,
            is_active = excluded.is_active,
            updated_at = CURRENT_TIMESTAMP
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare upsert_broker_account: {}", sqlite3_errmsg(impl_->db));
        return;
    }

    sqlite3_bind_text(stmt, 1, account.account_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, account.broker.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, account.account_name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, account.auth_payload.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 5, account.is_active ? 1 : 0);

    if (sqlite3_step(stmt) != SQLITE_DONE) {
        spdlog::error("Failed to upsert broker account {}: {}",
                      account.account_id, sqlite3_errmsg(impl_->db));
    }
    sqlite3_finalize(stmt);
}

std::optional<BrokerAccount> MetadataStore::get_broker_account(const std::string& account_id) {
    const char* sql = R"(
        SELECT account_id, broker, account_name, auth_payload, is_active
        FROM broker_accounts
        WHERE account_id = ?
    )";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, account_id.c_str(), -1, SQLITE_TRANSIENT);

    std::optional<BrokerAccount> out;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        BrokerAccount a;
        a.account_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        a.broker = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        a.account_name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        a.auth_payload = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        a.is_active = sqlite3_column_int(stmt, 4) != 0;
        out = std::move(a);
    }
    sqlite3_finalize(stmt);
    return out;
}

std::vector<BrokerAccount> MetadataStore::list_broker_accounts(bool active_only) {
    std::vector<BrokerAccount> out;
    const char* sql = R"(
        SELECT account_id, broker, account_name, auth_payload, is_active
        FROM broker_accounts
        WHERE (? = 0 OR is_active = 1)
        ORDER BY broker, account_id
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare list_broker_accounts: {}", sqlite3_errmsg(impl_->db));
        return out;
    }
    sqlite3_bind_int(stmt, 1, active_only ? 1 : 0);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        BrokerAccount a;
        a.account_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        a.broker = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        a.account_name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        a.auth_payload = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        a.is_active = sqlite3_column_int(stmt, 4) != 0;
        out.push_back(std::move(a));
    }
    sqlite3_finalize(stmt);
    return out;
}

void MetadataStore::upsert_account_cash(const AccountCashSnapshot& cash,
                                        const std::string& source) {
    const char* sql = R"(
        INSERT INTO account_cash_snapshots
        (account_id, as_of_date, total_asset, cash, available_cash, frozen_cash, market_value, source, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(account_id, as_of_date)
        DO UPDATE SET
            total_asset = excluded.total_asset,
            cash = excluded.cash,
            available_cash = excluded.available_cash,
            frozen_cash = excluded.frozen_cash,
            market_value = excluded.market_value,
            source = excluded.source,
            updated_at = CURRENT_TIMESTAMP
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare upsert_account_cash: {}", sqlite3_errmsg(impl_->db));
        return;
    }

    const std::string as_of = format_date(cash.as_of_date);
    sqlite3_bind_text(stmt, 1, cash.account_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, as_of.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(stmt, 3, cash.total_asset);
    sqlite3_bind_double(stmt, 4, cash.cash);
    sqlite3_bind_double(stmt, 5, cash.available_cash);
    sqlite3_bind_double(stmt, 6, cash.frozen_cash);
    sqlite3_bind_double(stmt, 7, cash.market_value);
    sqlite3_bind_text(stmt, 8, source.c_str(), -1, SQLITE_TRANSIENT);

    if (sqlite3_step(stmt) != SQLITE_DONE) {
        spdlog::error("Failed to upsert account cash {}: {}",
                      cash.account_id, sqlite3_errmsg(impl_->db));
    }
    sqlite3_finalize(stmt);
}

std::optional<AccountCashSnapshot> MetadataStore::latest_account_cash(const std::string& account_id) {
    const char* sql = R"(
        SELECT account_id, as_of_date, total_asset, cash, available_cash, frozen_cash, market_value
        FROM account_cash_snapshots
        WHERE account_id = ?
        ORDER BY as_of_date DESC
        LIMIT 1
    )";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, account_id.c_str(), -1, SQLITE_TRANSIENT);

    std::optional<AccountCashSnapshot> out;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        AccountCashSnapshot c;
        c.account_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        c.as_of_date = parse_date(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
        c.total_asset = sqlite3_column_double(stmt, 2);
        c.cash = sqlite3_column_double(stmt, 3);
        c.available_cash = sqlite3_column_double(stmt, 4);
        c.frozen_cash = sqlite3_column_double(stmt, 5);
        c.market_value = sqlite3_column_double(stmt, 6);
        out = std::move(c);
    }
    sqlite3_finalize(stmt);
    return out;
}

std::vector<AccountCashSnapshot> MetadataStore::list_account_cash(const std::string& account_id,
                                                                  int limit) {
    std::vector<AccountCashSnapshot> out;
    const char* sql = R"(
        SELECT account_id, as_of_date, total_asset, cash, available_cash, frozen_cash, market_value
        FROM account_cash_snapshots
        WHERE account_id = ?
        ORDER BY as_of_date DESC
        LIMIT ?
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare list_account_cash: {}", sqlite3_errmsg(impl_->db));
        return out;
    }
    sqlite3_bind_text(stmt, 1, account_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 2, clamp_limit(limit));

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        AccountCashSnapshot c;
        c.account_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        c.as_of_date = parse_date(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
        c.total_asset = sqlite3_column_double(stmt, 2);
        c.cash = sqlite3_column_double(stmt, 3);
        c.available_cash = sqlite3_column_double(stmt, 4);
        c.frozen_cash = sqlite3_column_double(stmt, 5);
        c.market_value = sqlite3_column_double(stmt, 6);
        out.push_back(std::move(c));
    }
    sqlite3_finalize(stmt);
    return out;
}

void MetadataStore::upsert_account_position(const AccountPositionSnapshot& position,
                                            const std::string& source) {
    const char* sql = R"(
        INSERT INTO account_position_snapshots
        (account_id, as_of_date, symbol, quantity, available_quantity, cost_price, last_price,
         market_value, unrealized_pnl, unrealized_pnl_ratio, source, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(account_id, as_of_date, symbol)
        DO UPDATE SET
            quantity = excluded.quantity,
            available_quantity = excluded.available_quantity,
            cost_price = excluded.cost_price,
            last_price = excluded.last_price,
            market_value = excluded.market_value,
            unrealized_pnl = excluded.unrealized_pnl,
            unrealized_pnl_ratio = excluded.unrealized_pnl_ratio,
            source = excluded.source,
            updated_at = CURRENT_TIMESTAMP
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare upsert_account_position: {}", sqlite3_errmsg(impl_->db));
        return;
    }

    const std::string as_of = format_date(position.as_of_date);
    sqlite3_bind_text(stmt, 1, position.account_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, as_of.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, position.symbol.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 4, position.quantity);
    sqlite3_bind_int64(stmt, 5, position.available_quantity);
    sqlite3_bind_double(stmt, 6, position.cost_price);
    sqlite3_bind_double(stmt, 7, position.last_price);
    sqlite3_bind_double(stmt, 8, position.market_value);
    sqlite3_bind_double(stmt, 9, position.unrealized_pnl);
    sqlite3_bind_double(stmt, 10, position.unrealized_pnl_ratio);
    sqlite3_bind_text(stmt, 11, source.c_str(), -1, SQLITE_TRANSIENT);

    if (sqlite3_step(stmt) != SQLITE_DONE) {
        spdlog::error("Failed to upsert account position {} {}: {}",
                      position.account_id, position.symbol, sqlite3_errmsg(impl_->db));
    }
    sqlite3_finalize(stmt);
}

std::vector<AccountPositionSnapshot>
MetadataStore::latest_account_positions(const std::string& account_id) {
    const char* sql = R"(
        SELECT MAX(as_of_date) FROM account_position_snapshots WHERE account_id = ?
    )";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, account_id.c_str(), -1, SQLITE_TRANSIENT);

    std::optional<Date> as_of;
    if (sqlite3_step(stmt) == SQLITE_ROW && sqlite3_column_text(stmt, 0)) {
        as_of = parse_date(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
    }
    sqlite3_finalize(stmt);
    if (!as_of) return {};
    return list_account_positions(account_id, as_of);
}

std::vector<AccountPositionSnapshot> MetadataStore::list_account_positions(
    const std::string& account_id,
    std::optional<Date> as_of_date) {
    std::vector<AccountPositionSnapshot> out;

    const char* sql_all = R"(
        SELECT account_id, as_of_date, symbol, quantity, available_quantity,
               cost_price, last_price, market_value, unrealized_pnl, unrealized_pnl_ratio
        FROM account_position_snapshots
        WHERE account_id = ?
        ORDER BY as_of_date DESC, symbol
    )";
    const char* sql_day = R"(
        SELECT account_id, as_of_date, symbol, quantity, available_quantity,
               cost_price, last_price, market_value, unrealized_pnl, unrealized_pnl_ratio
        FROM account_position_snapshots
        WHERE account_id = ? AND as_of_date = ?
        ORDER BY symbol
    )";

    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, as_of_date ? sql_day : sql_all, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare list_account_positions: {}", sqlite3_errmsg(impl_->db));
        return out;
    }
    sqlite3_bind_text(stmt, 1, account_id.c_str(), -1, SQLITE_TRANSIENT);
    if (as_of_date) {
        const std::string as_of = format_date(*as_of_date);
        sqlite3_bind_text(stmt, 2, as_of.c_str(), -1, SQLITE_TRANSIENT);
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        AccountPositionSnapshot p;
        p.account_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        p.as_of_date = parse_date(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
        p.symbol = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        p.quantity = sqlite3_column_int64(stmt, 3);
        p.available_quantity = sqlite3_column_int64(stmt, 4);
        p.cost_price = sqlite3_column_double(stmt, 5);
        p.last_price = sqlite3_column_double(stmt, 6);
        p.market_value = sqlite3_column_double(stmt, 7);
        p.unrealized_pnl = sqlite3_column_double(stmt, 8);
        p.unrealized_pnl_ratio = sqlite3_column_double(stmt, 9);
        out.push_back(std::move(p));
    }
    sqlite3_finalize(stmt);
    return out;
}

void MetadataStore::upsert_account_trade(const AccountTradeRecord& trade,
                                         const std::string& source) {
    const char* sql = R"(
        INSERT INTO account_trades
        (account_id, trade_id, trade_date, symbol, side, price, quantity, amount, fee, source, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(account_id, trade_id)
        DO UPDATE SET
            trade_date = excluded.trade_date,
            symbol = excluded.symbol,
            side = excluded.side,
            price = excluded.price,
            quantity = excluded.quantity,
            amount = excluded.amount,
            fee = excluded.fee,
            source = excluded.source,
            updated_at = CURRENT_TIMESTAMP
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare upsert_account_trade: {}", sqlite3_errmsg(impl_->db));
        return;
    }

    const std::string trade_date = format_date(trade.trade_date);
    sqlite3_bind_text(stmt, 1, trade.account_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, trade.trade_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, trade_date.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, trade.symbol.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 5, side_to_text(trade.side), -1, SQLITE_STATIC);
    sqlite3_bind_double(stmt, 6, trade.price);
    sqlite3_bind_int64(stmt, 7, trade.quantity);
    sqlite3_bind_double(stmt, 8, trade.amount);
    sqlite3_bind_double(stmt, 9, trade.fee);
    sqlite3_bind_text(stmt, 10, source.c_str(), -1, SQLITE_TRANSIENT);

    if (sqlite3_step(stmt) != SQLITE_DONE) {
        spdlog::error("Failed to upsert account trade {} {}: {}",
                      trade.account_id, trade.trade_id, sqlite3_errmsg(impl_->db));
    }
    sqlite3_finalize(stmt);
}

std::vector<AccountTradeRecord> MetadataStore::list_account_trades(const std::string& account_id,
                                                                   int limit) {
    std::vector<AccountTradeRecord> out;
    const char* sql = R"(
        SELECT account_id, trade_id, trade_date, symbol, side, price, quantity, amount, fee
        FROM account_trades
        WHERE account_id = ?
        ORDER BY trade_date DESC, trade_id DESC
        LIMIT ?
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare list_account_trades: {}", sqlite3_errmsg(impl_->db));
        return out;
    }
    sqlite3_bind_text(stmt, 1, account_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 2, clamp_limit(limit));

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        AccountTradeRecord t;
        t.account_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        t.trade_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        t.trade_date = parse_date(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2)));
        t.symbol = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        t.side = side_from_text(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4)));
        t.price = sqlite3_column_double(stmt, 5);
        t.quantity = sqlite3_column_int64(stmt, 6);
        t.amount = sqlite3_column_double(stmt, 7);
        t.fee = sqlite3_column_double(stmt, 8);
        out.push_back(std::move(t));
    }
    sqlite3_finalize(stmt);
    return out;
}

} // namespace trade
