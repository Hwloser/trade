#pragma once

#include "trade/common/types.h"
#include <string>
#include <filesystem>

namespace trade {

// Constructs standardized storage paths
// e.g., data/raw/cn_a/daily/2024/600000.SH.parquet
class StoragePath {
public:
    explicit StoragePath(const std::string& data_root);

    // Market data paths
    std::string raw_daily(const Symbol& symbol, int year) const;
    std::string curated_daily(const Symbol& symbol, int year) const;

    // Sentiment data paths (Bronze/Silver/Gold)
    std::string sentiment_bronze(int year, int month, const std::string& source, Date date) const;
    std::string sentiment_silver(int year, int month, Date date) const;
    std::string sentiment_gold(int year, int month, Date date) const;

    // Model paths
    std::string model_file(const std::string& name) const;
    std::string onnx_model(const std::string& name) const;

    // Future data paths
    std::string raw_minute(const Symbol& symbol, int year, int month) const;
    std::string raw_tick(const Symbol& symbol, Date date) const;

    // Models directory
    std::string models_dir() const;

    // Metadata
    std::string metadata_db() const;

    // Ensure directory exists
    static void ensure_dir(const std::string& path);

private:
    std::filesystem::path root_;
};

} // namespace trade
