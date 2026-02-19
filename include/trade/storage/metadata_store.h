#pragma once

#include "trade/common/types.h"
#include "trade/model/instrument.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace trade {

// SQLite-backed metadata store for instruments and data status
class MetadataStore {
public:
    struct DatasetRecord {
        std::string dataset_id;
        std::string layer;
        std::string domain;
        std::string data_type;
        std::string path_prefix;
        int schema_version = 1;
        std::optional<Date> latest_event_date;
    };

    struct DatasetFileRecord {
        std::string dataset_id;
        std::string file_path;
        int64_t row_count = 0;
        std::optional<Date> max_event_date;
    };

    explicit MetadataStore(const std::string& db_path);
    ~MetadataStore();

    // Instrument metadata
    void upsert_instrument(const Instrument& inst);
    std::optional<Instrument> get_instrument(const Symbol& symbol);
    std::vector<Instrument> get_all_instruments();
    std::vector<Instrument> get_instruments_by_market(Market market);
    std::vector<Instrument> get_instruments_by_industry(SWIndustry industry);

    // Data download tracking
    void record_download(const Symbol& symbol, Date start, Date end,
                        int64_t row_count);
    std::optional<Date> last_download_date(const Symbol& symbol);
    std::vector<Symbol> symbols_needing_update(Date cutoff);

    // Incremental watermarks (source + dataset + symbol)
    void upsert_watermark(const std::string& source,
                          const std::string& dataset,
                          const Symbol& symbol,
                          Date last_event_date,
                          const std::string& cursor_payload = "{}");
    std::optional<Date> last_watermark_date(const std::string& source,
                                            const std::string& dataset,
                                            const Symbol& symbol);

    // Ingestion run logs
    void begin_ingestion_run(const std::string& run_id,
                             const std::string& source,
                             const std::string& dataset,
                             const Symbol& symbol,
                             const std::string& mode);
    void finish_ingestion_run(const std::string& run_id,
                              bool success,
                              int64_t rows_in,
                              int64_t rows_out,
                              const std::string& error = "");

    // Holiday calendar
    void load_holidays(const std::vector<Date>& holidays);
    std::vector<Date> get_holidays(int year);

    // Dataset catalog (for discovery/query/training reproducibility)
    void upsert_dataset_file(const std::string& dataset_id,
                             const std::string& layer,
                             const std::string& domain,
                             const std::string& data_type,
                             const std::string& path_prefix,
                             const std::string& file_path,
                             int64_t row_count,
                             std::optional<Date> max_event_date = std::nullopt,
                             int schema_version = 1);
    std::vector<DatasetRecord> list_datasets();
    std::vector<DatasetFileRecord> list_dataset_files(const std::string& dataset_id);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace trade
