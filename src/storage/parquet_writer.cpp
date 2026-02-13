#include "trade/storage/parquet_writer.h"
#include "trade/common/time_utils.h"
#include <arrow/builder.h>
#include <arrow/io/file.h>
#include <arrow/table.h>
#include <parquet/arrow/writer.h>
#include <spdlog/spdlog.h>
#include <filesystem>

namespace trade {

std::shared_ptr<arrow::Schema> ParquetWriter::bar_schema() {
    return arrow::schema({
        arrow::field("symbol", arrow::utf8()),
        arrow::field("date", arrow::utf8()),    // YYYY-MM-DD string
        arrow::field("open", arrow::float64()),
        arrow::field("high", arrow::float64()),
        arrow::field("low", arrow::float64()),
        arrow::field("close", arrow::float64()),
        arrow::field("volume", arrow::int64()),
        arrow::field("amount", arrow::float64()),
        arrow::field("turnover_rate", arrow::float64()),
        arrow::field("prev_close", arrow::float64()),
        arrow::field("vwap", arrow::float64()),
    });
}

std::shared_ptr<arrow::Schema> ParquetWriter::ext_bar_schema() {
    return arrow::schema({
        arrow::field("symbol", arrow::utf8()),
        arrow::field("date", arrow::utf8()),
        arrow::field("open", arrow::float64()),
        arrow::field("high", arrow::float64()),
        arrow::field("low", arrow::float64()),
        arrow::field("close", arrow::float64()),
        arrow::field("volume", arrow::int64()),
        arrow::field("amount", arrow::float64()),
        arrow::field("turnover_rate", arrow::float64()),
        arrow::field("prev_close", arrow::float64()),
        arrow::field("vwap", arrow::float64()),
        arrow::field("limit_up", arrow::float64()),
        arrow::field("limit_down", arrow::float64()),
        arrow::field("hit_limit_up", arrow::boolean()),
        arrow::field("hit_limit_down", arrow::boolean()),
        arrow::field("status", arrow::uint8()),
        arrow::field("board", arrow::uint8()),
        arrow::field("north_net_buy", arrow::float64()),
        arrow::field("margin_balance", arrow::float64()),
        arrow::field("short_sell_volume", arrow::float64()),
    });
}

std::shared_ptr<arrow::Table> ParquetWriter::bars_to_table(const std::vector<Bar>& bars) {
    arrow::StringBuilder symbol_builder;
    arrow::StringBuilder date_builder;
    arrow::DoubleBuilder open_builder, high_builder, low_builder, close_builder;
    arrow::Int64Builder volume_builder;
    arrow::DoubleBuilder amount_builder, turnover_builder, prev_close_builder, vwap_builder;

    for (const auto& bar : bars) {
        (void)symbol_builder.Append(bar.symbol);
        (void)date_builder.Append(format_date(bar.date));
        (void)open_builder.Append(bar.open);
        (void)high_builder.Append(bar.high);
        (void)low_builder.Append(bar.low);
        (void)close_builder.Append(bar.close);
        (void)volume_builder.Append(bar.volume);
        (void)amount_builder.Append(bar.amount);
        (void)turnover_builder.Append(bar.turnover_rate);
        (void)prev_close_builder.Append(bar.prev_close);
        (void)vwap_builder.Append(bar.vwap);
    }

    std::shared_ptr<arrow::Array> arrays[11];
    (void)symbol_builder.Finish(&arrays[0]);
    (void)date_builder.Finish(&arrays[1]);
    (void)open_builder.Finish(&arrays[2]);
    (void)high_builder.Finish(&arrays[3]);
    (void)low_builder.Finish(&arrays[4]);
    (void)close_builder.Finish(&arrays[5]);
    (void)volume_builder.Finish(&arrays[6]);
    (void)amount_builder.Finish(&arrays[7]);
    (void)turnover_builder.Finish(&arrays[8]);
    (void)prev_close_builder.Finish(&arrays[9]);
    (void)vwap_builder.Finish(&arrays[10]);

    return arrow::Table::Make(bar_schema(),
        {arrays[0], arrays[1], arrays[2], arrays[3], arrays[4],
         arrays[5], arrays[6], arrays[7], arrays[8], arrays[9], arrays[10]});
}

std::shared_ptr<arrow::Table> ParquetWriter::ext_bars_to_table(const std::vector<ExtBar>& bars) {
    arrow::StringBuilder symbol_builder, date_builder;
    arrow::DoubleBuilder open_b, high_b, low_b, close_b, amount_b, turnover_b, prev_close_b, vwap_b;
    arrow::Int64Builder volume_b;
    arrow::DoubleBuilder limit_up_b, limit_down_b;
    arrow::BooleanBuilder hit_up_b, hit_down_b;
    arrow::UInt8Builder status_b, board_b;
    arrow::DoubleBuilder north_b, margin_b, short_b;

    for (const auto& bar : bars) {
        (void)symbol_builder.Append(bar.symbol);
        (void)date_builder.Append(format_date(bar.date));
        (void)open_b.Append(bar.open);
        (void)high_b.Append(bar.high);
        (void)low_b.Append(bar.low);
        (void)close_b.Append(bar.close);
        (void)volume_b.Append(bar.volume);
        (void)amount_b.Append(bar.amount);
        (void)turnover_b.Append(bar.turnover_rate);
        (void)prev_close_b.Append(bar.prev_close);
        (void)vwap_b.Append(bar.vwap);
        (void)limit_up_b.Append(bar.limit_up);
        (void)limit_down_b.Append(bar.limit_down);
        (void)hit_up_b.Append(bar.hit_limit_up);
        (void)hit_down_b.Append(bar.hit_limit_down);
        (void)status_b.Append(static_cast<uint8_t>(bar.status));
        (void)board_b.Append(static_cast<uint8_t>(bar.board));
        (void)north_b.Append(bar.north_net_buy.value_or(0.0));
        (void)margin_b.Append(bar.margin_balance.value_or(0.0));
        (void)short_b.Append(bar.short_sell_volume.value_or(0.0));
    }

    std::vector<std::shared_ptr<arrow::Array>> arrays(20);
    (void)symbol_builder.Finish(&arrays[0]);
    (void)date_builder.Finish(&arrays[1]);
    (void)open_b.Finish(&arrays[2]);
    (void)high_b.Finish(&arrays[3]);
    (void)low_b.Finish(&arrays[4]);
    (void)close_b.Finish(&arrays[5]);
    (void)volume_b.Finish(&arrays[6]);
    (void)amount_b.Finish(&arrays[7]);
    (void)turnover_b.Finish(&arrays[8]);
    (void)prev_close_b.Finish(&arrays[9]);
    (void)vwap_b.Finish(&arrays[10]);
    (void)limit_up_b.Finish(&arrays[11]);
    (void)limit_down_b.Finish(&arrays[12]);
    (void)hit_up_b.Finish(&arrays[13]);
    (void)hit_down_b.Finish(&arrays[14]);
    (void)status_b.Finish(&arrays[15]);
    (void)board_b.Finish(&arrays[16]);
    (void)north_b.Finish(&arrays[17]);
    (void)margin_b.Finish(&arrays[18]);
    (void)short_b.Finish(&arrays[19]);

    return arrow::Table::Make(ext_bar_schema(), arrays);
}

void ParquetWriter::write_bars(const std::string& path, const std::vector<Bar>& bars) {
    auto table = bars_to_table(bars);
    write_table(path, table);
}

void ParquetWriter::write_ext_bars(const std::string& path, const std::vector<ExtBar>& bars) {
    auto table = ext_bars_to_table(bars);
    write_table(path, table);
}

void ParquetWriter::write_table(const std::string& path,
                                 const std::shared_ptr<arrow::Table>& table) {
    // Ensure parent directory exists
    auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    auto outfile = arrow::io::FileOutputStream::Open(path);
    if (!outfile.ok()) {
        spdlog::error("Failed to open {} for writing: {}", path, outfile.status().ToString());
        return;
    }

    auto writer_props = parquet::WriterProperties::Builder()
        .compression(parquet::Compression::SNAPPY)
        ->build();

    auto status = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(),
                                              *outfile, /*chunk_size=*/65536,
                                              writer_props);
    if (!status.ok()) {
        spdlog::error("Failed to write parquet {}: {}", path, status.ToString());
    } else {
        spdlog::debug("Wrote {} rows to {}", table->num_rows(), path);
    }
}

} // namespace trade
