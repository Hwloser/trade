#include "trade/storage/parquet_reader.h"
#include "trade/common/time_utils.h"
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>
#include <spdlog/spdlog.h>

namespace trade {

namespace {

Bar row_to_bar(const std::shared_ptr<arrow::Table>& table, int64_t row) {
    Bar bar;
    auto get_string = [&](const std::string& col) -> std::string {
        auto column = table->GetColumnByName(col);
        if (!column) return "";
        auto arr = std::static_pointer_cast<arrow::StringArray>(column->chunk(0));
        return arr->GetString(row);
    };
    auto get_double = [&](const std::string& col) -> double {
        auto column = table->GetColumnByName(col);
        if (!column) return 0.0;
        auto arr = std::static_pointer_cast<arrow::DoubleArray>(column->chunk(0));
        return arr->Value(row);
    };
    auto get_int64 = [&](const std::string& col) -> int64_t {
        auto column = table->GetColumnByName(col);
        if (!column) return 0;
        auto arr = std::static_pointer_cast<arrow::Int64Array>(column->chunk(0));
        return arr->Value(row);
    };
    auto get_bool = [&](const std::string& col) -> bool {
        auto column = table->GetColumnByName(col);
        if (!column) return false;
        auto arr = std::static_pointer_cast<arrow::BooleanArray>(column->chunk(0));
        return arr->Value(row);
    };
    auto get_uint8 = [&](const std::string& col) -> uint8_t {
        auto column = table->GetColumnByName(col);
        if (!column) return 0;
        auto arr = std::static_pointer_cast<arrow::UInt8Array>(column->chunk(0));
        return arr->Value(row);
    };

    bar.symbol = get_string("symbol");
    bar.date = parse_date(get_string("date"));
    bar.open = get_double("open");
    bar.high = get_double("high");
    bar.low = get_double("low");
    bar.close = get_double("close");
    bar.volume = get_int64("volume");
    bar.amount = get_double("amount");
    bar.turnover_rate = get_double("turnover_rate");
    bar.prev_close = get_double("prev_close");
    bar.vwap = get_double("vwap");

    // Extended fields (schema evolution: missing columns → defaults)
    bar.limit_up = get_double("limit_up");
    bar.limit_down = get_double("limit_down");
    bar.hit_limit_up = get_bool("hit_limit_up");
    bar.hit_limit_down = get_bool("hit_limit_down");
    // Support both old "status" and new "bar_status" column names
    if (table->GetColumnByName("bar_status")) {
        bar.bar_status = static_cast<TradingStatus>(get_uint8("bar_status"));
    } else {
        bar.bar_status = static_cast<TradingStatus>(get_uint8("status"));
    }
    bar.board = static_cast<Board>(get_uint8("board"));

    double north = get_double("north_net_buy");
    if (north != 0.0) bar.north_net_buy = north;
    double margin = get_double("margin_balance");
    if (margin != 0.0) bar.margin_balance = margin;
    double short_v = get_double("short_sell_volume");
    if (short_v != 0.0) bar.short_sell_volume = short_v;

    return bar;
}

} // namespace

std::vector<Bar> ParquetReader::read_bars(const std::string& path) {
    auto table = read_table(path);
    if (!table) return {};

    std::vector<Bar> bars;
    bars.reserve(table->num_rows());
    for (int64_t i = 0; i < table->num_rows(); ++i) {
        bars.push_back(row_to_bar(table, i));
    }
    return bars;
}

std::vector<Bar> ParquetReader::read_bars(const std::string& path,
                                           std::optional<Date> start,
                                           std::optional<Date> end) {
    auto all = read_bars(path);
    if (!start && !end) return all;

    std::vector<Bar> filtered;
    for (auto& bar : all) {
        if (start && bar.date < *start) continue;
        if (end && bar.date > *end) continue;
        filtered.push_back(std::move(bar));
    }
    return filtered;
}

std::shared_ptr<arrow::Table> ParquetReader::read_table(const std::string& path) {
    auto infile = arrow::io::ReadableFile::Open(path);
    if (!infile.ok()) {
        spdlog::error("Failed to open {}: {}", path, infile.status().ToString());
        return nullptr;
    }

    auto reader_result = parquet::arrow::OpenFile(*infile, arrow::default_memory_pool());
    if (!reader_result.ok()) {
        spdlog::error("Failed to open parquet reader for {}: {}", path, reader_result.status().ToString());
        return nullptr;
    }
    auto reader = std::move(*reader_result);

    std::shared_ptr<arrow::Table> table;
    auto status = reader->ReadTable(&table);
    if (!status.ok()) {
        spdlog::error("Failed to read table from {}: {}", path, status.ToString());
        return nullptr;
    }

    return table;
}

std::shared_ptr<arrow::Table> ParquetReader::read_columns(
    const std::string& path,
    const std::vector<std::string>& columns) {
    auto infile = arrow::io::ReadableFile::Open(path);
    if (!infile.ok()) return nullptr;

    auto reader_result = parquet::arrow::OpenFile(*infile, arrow::default_memory_pool());
    if (!reader_result.ok()) return nullptr;
    auto reader = std::move(*reader_result);

    // Get column indices
    auto schema = reader->parquet_reader()->metadata()->schema();
    std::vector<int> indices;
    for (const auto& col : columns) {
        int idx = schema->ColumnIndex(col);
        if (idx >= 0) indices.push_back(idx);
    }

    std::shared_ptr<arrow::Table> table;
    auto status = reader->ReadTable(indices, &table);
    if (!status.ok()) return nullptr;
    return table;
}

int64_t ParquetReader::row_count(const std::string& path) {
    auto infile = arrow::io::ReadableFile::Open(path);
    if (!infile.ok()) return -1;

    auto reader_result = parquet::arrow::OpenFile(*infile, arrow::default_memory_pool());
    if (!reader_result.ok()) return -1;
    auto reader = std::move(*reader_result);

    return reader->parquet_reader()->metadata()->num_rows();
}

} // namespace trade
