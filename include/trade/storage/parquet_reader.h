#pragma once

#include "trade/model/bar.h"
#include <string>
#include <vector>
#include <optional>
#include <arrow/api.h>

namespace trade {

class ParquetReader {
public:
    // Read bars from parquet file
    static std::vector<Bar> read_bars(const std::string& path);

    // Read bars with date filter
    static std::vector<Bar> read_bars(const std::string& path,
                                       std::optional<Date> start,
                                       std::optional<Date> end);

    // Read raw Arrow table
    static std::shared_ptr<arrow::Table> read_table(const std::string& path);

    // Read specific columns only
    static std::shared_ptr<arrow::Table> read_columns(
        const std::string& path,
        const std::vector<std::string>& columns);

    // Get row count without reading data
    static int64_t row_count(const std::string& path);
};

} // namespace trade
