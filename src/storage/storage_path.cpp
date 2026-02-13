#include "trade/storage/storage_path.h"
#include "trade/common/time_utils.h"
#include <fmt/format.h>

namespace trade {

StoragePath::StoragePath(const std::string& data_root) : root_(data_root) {}

std::string StoragePath::raw_daily(const Symbol& symbol, int year) const {
    return (root_ / "raw" / "cn_a" / "daily" / std::to_string(year) /
            (symbol + ".parquet")).string();
}

std::string StoragePath::curated_daily(const Symbol& symbol, int year) const {
    return (root_ / "curated" / "cn_a" / "daily" / std::to_string(year) /
            (symbol + ".parquet")).string();
}

std::string StoragePath::sentiment_bronze(int year, int month,
                                           const std::string& source, Date date) const {
    return (root_ / "raw" / "sentiment" / "bronze" /
            std::to_string(year) / fmt::format("{:02d}", month) /
            fmt::format("{}_{}.parquet", source, format_date(date))).string();
}

std::string StoragePath::sentiment_silver(int year, int month, Date date) const {
    return (root_ / "curated" / "sentiment" / "silver" /
            std::to_string(year) / fmt::format("{:02d}", month) /
            fmt::format("nlp_{}.parquet", format_date(date))).string();
}

std::string StoragePath::sentiment_gold(int year, int month, Date date) const {
    return (root_ / "curated" / "sentiment" / "gold" /
            std::to_string(year) / fmt::format("{:02d}", month) /
            fmt::format("factors_{}.parquet", format_date(date))).string();
}

std::string StoragePath::model_file(const std::string& name) const {
    return (root_ / "models" / (name + ".model")).string();
}

std::string StoragePath::onnx_model(const std::string& name) const {
    return (root_ / "models" / "sentiment" / (name + ".onnx")).string();
}

std::string StoragePath::metadata_db() const {
    return (root_ / "metadata.db").string();
}

void StoragePath::ensure_dir(const std::string& path) {
    auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

} // namespace trade
