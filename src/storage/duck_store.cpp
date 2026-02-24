#include "trade/storage/duck_store.h"
#include "trade/common/time_utils.h"
#include "trade/model/bar.h"
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

std::vector<Bar> DuckStore::read_bars(const std::string& glob_pattern,
                                       const std::string& symbol,
                                       const std::string& start_date,
                                       const std::string& end_date) {
    // Escape single quotes in symbol to prevent SQL injection
    std::string safe_symbol;
    for (char c : symbol) {
        safe_symbol.push_back(c);
        if (c == '\'') safe_symbol.push_back('\'');
    }

    std::string sql =
        "SELECT date, open, high, low, close, volume, amount, turnover_rate,"
        " prev_close, vwap, limit_up, limit_down, hit_limit_up, hit_limit_down,"
        " bar_status, board"
        " FROM read_parquet('" + glob_pattern + "', union_by_name=true)"
        " WHERE symbol = '" + safe_symbol + "'";

    if (!start_date.empty()) {
        sql += " AND date >= '" + start_date + "'";
    }
    if (!end_date.empty()) {
        sql += " AND date <= '" + end_date + "'";
    }
    sql += " ORDER BY date";

    auto rows = query(sql);

    // Column indices in SELECT list:
    // 0:date  1:open  2:high  3:low  4:close  5:volume  6:amount
    // 7:turnover_rate  8:prev_close  9:vwap  10:limit_up  11:limit_down
    // 12:hit_limit_up  13:hit_limit_down  14:bar_status  15:board
    auto safe_double = [](const std::string& s) -> double {
        if (s.empty()) return 0.0;
        try { return std::stod(s); } catch (...) { return 0.0; }
    };
    auto safe_int64 = [](const std::string& s) -> int64_t {
        if (s.empty()) return 0;
        try { return std::stoll(s); } catch (...) { return 0; }
    };
    auto safe_bool = [](const std::string& s) -> bool {
        return s == "true" || s == "1" || s == "True" || s == "TRUE";
    };
    auto safe_uint8 = [](const std::string& s) -> uint8_t {
        if (s.empty()) return 0;
        try { return static_cast<uint8_t>(std::stoul(s)); } catch (...) { return 0; }
    };

    std::vector<Bar> bars;
    bars.reserve(rows.size());
    for (const auto& row : rows) {
        if (row.size() < 16) continue;
        Bar bar;
        bar.symbol        = symbol;
        bar.date          = parse_date(row[0]);
        bar.open          = safe_double(row[1]);
        bar.high          = safe_double(row[2]);
        bar.low           = safe_double(row[3]);
        bar.close         = safe_double(row[4]);
        bar.volume        = safe_int64(row[5]);
        bar.amount        = safe_double(row[6]);
        bar.turnover_rate = safe_double(row[7]);
        bar.prev_close    = safe_double(row[8]);
        bar.vwap          = safe_double(row[9]);
        bar.limit_up      = safe_double(row[10]);
        bar.limit_down    = safe_double(row[11]);
        bar.hit_limit_up  = safe_bool(row[12]);
        bar.hit_limit_down = safe_bool(row[13]);
        bar.bar_status    = static_cast<TradingStatus>(safe_uint8(row[14]));
        bar.board         = static_cast<Board>(safe_uint8(row[15]));
        bars.push_back(std::move(bar));
    }
    return bars;
}

} // namespace trade
