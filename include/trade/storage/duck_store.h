#pragma once
#include <string>
#include <vector>

namespace trade {

// DuckDB-backed analytics query engine.
// Wraps the DuckDB C API for in-process SQL execution against Parquet files.
class DuckStore {
public:
    DuckStore();
    ~DuckStore();

    DuckStore(const DuckStore&) = delete;
    DuckStore& operator=(const DuckStore&) = delete;

    // Returns true if DuckDB is compiled in and functional.
    static bool available();

    // Execute a SQL statement (no result needed, e.g. CREATE VIEW).
    bool execute(const std::string& sql);

    // Run a SELECT query; returns rows as vector of string columns.
    std::vector<std::vector<std::string>> query(const std::string& sql);

    // Count rows matching a glob pattern.
    // e.g. count_rows("data/kline/**/*.parquet")
    int64_t count_rows(const std::string& glob_pattern);

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

} // namespace trade
