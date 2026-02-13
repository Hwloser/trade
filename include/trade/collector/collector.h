#pragma once

#include "trade/provider/i_data_provider.h"
#include "trade/storage/parquet_writer.h"
#include "trade/storage/metadata_store.h"
#include "trade/storage/storage_path.h"
#include "trade/normalizer/bar_normalizer.h"
#include "trade/validator/data_validator.h"
#include "trade/common/config.h"
#include <memory>
#include <vector>

namespace trade {

// Orchestrates the data collection pipeline:
// Provider → Normalizer → Validator → Storage
class Collector {
public:
    Collector(std::unique_ptr<IDataProvider> provider,
              const Config& config);

    // Download and store daily bars for a single symbol
    QualityReport collect_symbol(const Symbol& symbol, Date start, Date end);

    // Download and store daily bars for all symbols
    void collect_all(Date start, Date end,
                     ProgressCallback progress = nullptr);

    // Incremental update: only download new data since last download
    void update_all(ProgressCallback progress = nullptr);

private:
    std::unique_ptr<IDataProvider> provider_;
    StoragePath paths_;
    MetadataStore metadata_;
    Config config_;
};

} // namespace trade
