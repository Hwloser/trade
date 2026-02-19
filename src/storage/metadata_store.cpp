#include "trade/storage/metadata_store.h"
#include "trade/common/time_utils.h"
#include <filesystem>
#include <spdlog/spdlog.h>
#include <sqlite3.h>
#include <stdexcept>

namespace trade {

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
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (dataset_id, file_path)
        )
    )");

    impl_->exec("CREATE INDEX IF NOT EXISTS idx_downloads_symbol_end ON downloads(symbol, end_date)");
    impl_->exec("CREATE INDEX IF NOT EXISTS idx_watermarks_lookup ON watermarks(source, dataset, symbol)");
    impl_->exec("CREATE INDEX IF NOT EXISTS idx_dataset_files_dataset ON dataset_files(dataset_id)");

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

void MetadataStore::upsert_dataset_file(const std::string& dataset_id,
                                        const std::string& layer,
                                        const std::string& domain,
                                        const std::string& data_type,
                                        const std::string& path_prefix,
                                        const std::string& file_path,
                                        int64_t row_count,
                                        std::optional<Date> max_event_date,
                                        int schema_version) {
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
        return;
    }
    sqlite3_bind_text(stmt, 1, dataset_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, layer.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, domain.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, data_type.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 5, path_prefix.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 6, schema_version);
    if (max_event_date) {
        std::string d = format_date(*max_event_date);
        sqlite3_bind_text(stmt, 7, d.c_str(), -1, SQLITE_TRANSIENT);
    } else {
        sqlite3_bind_null(stmt, 7);
    }
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        spdlog::error("Failed to upsert dataset catalog {}: {}",
                      dataset_id, sqlite3_errmsg(impl_->db));
    }
    sqlite3_finalize(stmt);

    const char* sql_file = R"(
        INSERT INTO dataset_files
        (dataset_id, file_path, row_count, max_event_date, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(dataset_id, file_path)
        DO UPDATE SET
            row_count = excluded.row_count,
            max_event_date = CASE
                WHEN excluded.max_event_date IS NULL THEN dataset_files.max_event_date
                ELSE excluded.max_event_date
            END,
            updated_at = CURRENT_TIMESTAMP
    )";
    stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql_file, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare upsert dataset_files: {}", sqlite3_errmsg(impl_->db));
        return;
    }
    sqlite3_bind_text(stmt, 1, dataset_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, file_path.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 3, row_count);
    if (max_event_date) {
        std::string d = format_date(*max_event_date);
        sqlite3_bind_text(stmt, 4, d.c_str(), -1, SQLITE_TRANSIENT);
    } else {
        sqlite3_bind_null(stmt, 4);
    }
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        spdlog::error("Failed to upsert dataset file {} {}: {}",
                      dataset_id, file_path, sqlite3_errmsg(impl_->db));
    }
    sqlite3_finalize(stmt);
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
        SELECT dataset_id, file_path, row_count, max_event_date
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
        auto date_txt = sqlite3_column_text(stmt, 3);
        if (date_txt) {
            r.max_event_date = parse_date(reinterpret_cast<const char*>(date_txt));
        }
        out.push_back(std::move(r));
    }
    sqlite3_finalize(stmt);
    return out;
}

} // namespace trade
