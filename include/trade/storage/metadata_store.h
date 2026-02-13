#pragma once

#include "trade/common/types.h"
#include "trade/model/instrument.h"
#include <string>
#include <vector>
#include <optional>

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

    // Holiday calendar
    void load_holidays(const std::vector<Date>& holidays);
    std::vector<Date> get_holidays(int year);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace trade
