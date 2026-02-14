#pragma once

#include "trade/model/bar.h"
#include <string>
#include <vector>
#include <arrow/api.h>
#include <parquet/arrow/writer.h>

namespace trade {

class ParquetWriter {
public:
    // Write bars to a parquet file (full 20-column schema)
    static void write_bars(const std::string& path, const std::vector<Bar>& bars);

    // Write arbitrary Arrow table
    static void write_table(const std::string& path,
                           const std::shared_ptr<arrow::Table>& table);

private:
    static std::shared_ptr<arrow::Schema> bar_schema();
    static std::shared_ptr<arrow::Table> bars_to_table(const std::vector<Bar>& bars);
};

} // namespace trade
