#include "trade/storage/duck_store.h"
#include <duckdb.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace trade {

struct DuckStore::Impl {
    duckdb_database db = nullptr;
    duckdb_connection con = nullptr;

    ~Impl() {
        if (con) duckdb_disconnect(&con);
        if (db)  duckdb_close(&db);
    }
};

DuckStore::DuckStore() : impl_(new Impl()) {
    if (duckdb_open(nullptr, &impl_->db) != DuckDBSuccess) {
        throw std::runtime_error("DuckStore: failed to open in-memory DuckDB");
    }
    if (duckdb_connect(impl_->db, &impl_->con) != DuckDBSuccess) {
        throw std::runtime_error("DuckStore: failed to connect to DuckDB");
    }
}

DuckStore::~DuckStore() {
    delete impl_;
}

bool DuckStore::available() {
    return true;
}

bool DuckStore::execute(const std::string& sql) {
    duckdb_result result;
    bool ok = (duckdb_query(impl_->con, sql.c_str(), &result) == DuckDBSuccess);
    if (!ok) {
        spdlog::error("DuckStore::execute failed: {}", duckdb_result_error(&result));
    }
    duckdb_destroy_result(&result);
    return ok;
}

std::vector<std::vector<std::string>> DuckStore::query(const std::string& sql) {
    duckdb_result result;
    std::vector<std::vector<std::string>> rows;

    if (duckdb_query(impl_->con, sql.c_str(), &result) != DuckDBSuccess) {
        spdlog::error("DuckStore::query failed: {}", duckdb_result_error(&result));
        duckdb_destroy_result(&result);
        return rows;
    }

    idx_t ncols = duckdb_column_count(&result);
    idx_t nrows = duckdb_row_count(&result);
    rows.reserve(static_cast<size_t>(nrows));

    for (idx_t r = 0; r < nrows; ++r) {
        std::vector<std::string> row;
        row.reserve(static_cast<size_t>(ncols));
        for (idx_t c = 0; c < ncols; ++c) {
            auto* val = duckdb_value_varchar(&result, c, r);
            row.push_back(val ? std::string(val) : "");
            duckdb_free(val);
        }
        rows.push_back(std::move(row));
    }

    duckdb_destroy_result(&result);
    return rows;
}

int64_t DuckStore::count_rows(const std::string& glob_pattern) {
    std::string sql = "SELECT count(*) FROM read_parquet('" + glob_pattern + "')";
    auto rows = query(sql);
    if (rows.empty() || rows[0].empty()) return -1;
    try {
        return std::stoll(rows[0][0]);
    } catch (...) {
        return -1;
    }
}

} // namespace trade
