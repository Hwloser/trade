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

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace trade
