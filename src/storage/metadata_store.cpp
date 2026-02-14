#include "trade/storage/metadata_store.h"
#include "trade/common/time_utils.h"
#include <sqlite3.h>
#include <spdlog/spdlog.h>
#include <filesystem>
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

    // Create tables
    impl_->exec(R"(
        CREATE TABLE IF NOT EXISTS instruments (
            symbol TEXT PRIMARY KEY,
            name TEXT,
            market INTEGER,
            board INTEGER,
            industry INTEGER,
            list_date TEXT,
            delist_date TEXT,
            status INTEGER
        )
    )");

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
        CREATE TABLE IF NOT EXISTS holidays (
            date TEXT PRIMARY KEY,
            year INTEGER
        )
    )");

    spdlog::debug("MetadataStore initialized at {}", db_path);
}

MetadataStore::~MetadataStore() = default;

void MetadataStore::upsert_instrument(const Instrument& inst) {
    const char* sql = R"(
        INSERT OR REPLACE INTO instruments (symbol, name, market, board, industry, list_date, delist_date, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    )";
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);

    sqlite3_bind_text(stmt, 1, inst.symbol.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, inst.name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 3, static_cast<int>(inst.market));
    sqlite3_bind_int(stmt, 4, static_cast<int>(inst.board));
    sqlite3_bind_int(stmt, 5, static_cast<int>(inst.industry));
    sqlite3_bind_text(stmt, 6, format_date(inst.list_date).c_str(), -1, SQLITE_TRANSIENT);
    if (inst.delist_date) {
        sqlite3_bind_text(stmt, 7, format_date(*inst.delist_date).c_str(), -1, SQLITE_TRANSIENT);
    } else {
        sqlite3_bind_null(stmt, 7);
    }
    sqlite3_bind_int(stmt, 8, static_cast<int>(inst.status));

    sqlite3_step(stmt);
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
    sqlite3_bind_text(stmt, 1, symbol.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, format_date(start).c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, format_date(end).c_str(), -1, SQLITE_TRANSIENT);
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

    // Find symbols with no download or last download before cutoff
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

} // namespace trade
