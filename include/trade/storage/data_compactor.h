#pragma once

#include "trade/common/config.h"

#include <cstdint>
#include <functional>
#include <string>

namespace trade {

enum class CompactionMode : uint8_t {
    kMinor = 0,
    kMajor = 1,
};

struct CompactionOptions {
    int bucket_count = 32;
    int small_file_row_threshold = 4000;
    int tombstone_retention_days = 0;
    bool dry_run = false;
};

struct CompactionStats {
    int groups_total = 0;
    int groups_compacted = 0;
    int source_files_seen = 0;
    int source_files_deleted = 0;
    int target_files_written = 0;
    int target_files_uploaded = 0;
    int tombstones_purged = 0;
    int64_t rows_in = 0;
    int64_t rows_out = 0;
    int64_t rows_deduped = 0;
};

using CompactionProgress = std::function<void(const std::string& phase,
                                              int current,
                                              int total,
                                              const std::string& detail)>;

class DataCompactor {
public:
    explicit DataCompactor(const Config& config);

    CompactionStats compact_daily_dataset(const std::string& dataset_id,
                                          CompactionMode mode,
                                          const CompactionOptions& options = {},
                                          CompactionProgress progress = nullptr);

private:
    Config config_;
};

} // namespace trade
