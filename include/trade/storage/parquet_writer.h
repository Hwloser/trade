#pragma once

#include "trade/model/bar.h"
#include "trade/sentiment/sentiment_factor.h"
#include "trade/sentiment/text_source.h"
#include "trade/common/config.h"

#include <arrow/api.h>
#include <parquet/arrow/writer.h>
#include <optional>
#include <string>
#include <vector>

namespace trade {

class ParquetStore {
public:
    enum class MergeMode : uint8_t {
        kReplace = 0,
        kMergeByKey = 1,
    };

    // Market bars
    static void write_bars(const std::string& path,
                           const std::vector<Bar>& bars,
                           MergeMode mode = MergeMode::kReplace,
                           std::optional<Date> partition_max_date = std::nullopt);

    // Sentiment bronze/silver/gold
    static void write_text_events(const std::string& path,
                                  const std::vector<TextEvent>& events,
                                  MergeMode mode = MergeMode::kMergeByKey,
                                  std::optional<Date> partition_max_date = std::nullopt);
    static void write_nlp_results(const std::string& path,
                                  const std::vector<NlpResult>& results,
                                  MergeMode mode = MergeMode::kMergeByKey,
                                  std::optional<Date> partition_max_date = std::nullopt);
    static void write_sentiment_factors(const std::string& path,
                                        const std::vector<SentimentFactors>& factors,
                                        MergeMode mode = MergeMode::kMergeByKey,
                                        std::optional<Date> partition_max_date = std::nullopt);

    // Generic Arrow table write
    static void write_table(const std::string& path,
                            const std::shared_ptr<arrow::Table>& table,
                            std::optional<Date> partition_max_date = std::nullopt);

    // Configure runtime storage routing (local / hybrid / cloud)
    static void configure_runtime(const DataConfig& data_cfg,
                                  const StorageConfig& storage_cfg);

private:
    static std::shared_ptr<arrow::Schema> bar_schema();
    static std::shared_ptr<arrow::Table> bars_to_table(const std::vector<Bar>& bars);
};

// Backward-compatible alias: existing code can continue using ParquetWriter.
using ParquetWriter = ParquetStore;

} // namespace trade
