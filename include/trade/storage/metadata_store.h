#pragma once

#include "trade/common/types.h"
#include "trade/model/account.h"
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
        int current_version = 1;
    };

    struct SchemaRecord {
        std::string dataset_id;
        int schema_version = 1;
        std::string schema_json;
        std::string schema_hash;
        bool is_active = true;
    };

    struct QualityCheckRecord {
        std::string run_id;
        std::string dataset_id;
        std::string check_name;
        std::string status;      // pass|warn|fail
        std::string severity;    // info|warning|critical
        double metric_value = 0.0;
        double threshold_value = 0.0;
        std::string message;
        std::optional<Date> event_date;
    };

    struct StreamCheckpointRecord {
        std::string source;
        std::string stream;
        std::string shard;
        std::string cursor_payload;
        std::optional<Date> last_event_date;
    };

    struct TrainingSnapshotRecord {
        std::string snapshot_id;
        std::string dataset_id;
        std::string query_spec;
        std::string snapshot_path;
        std::optional<Date> start_date;
        std::optional<Date> end_date;
        int64_t row_count = 0;
        int schema_version = 1;
        std::string model_name;
    };

    struct DatasetFileVersionRecord {
        std::string dataset_id;
        std::string file_path;
        int version = 1;
        int64_t row_count = 0;
        std::optional<Date> max_event_date;
        std::string run_id;
        std::string content_hash;
    };

    struct DatasetTombstoneRecord {
        std::string dataset_id;
        std::string file_path;
        int version = 0;
        std::string reason;
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
    int upsert_dataset_file(const std::string& dataset_id,
                            const std::string& layer,
                            const std::string& domain,
                            const std::string& data_type,
                            const std::string& path_prefix,
                            const std::string& file_path,
                            int64_t row_count,
                            std::optional<Date> max_event_date = std::nullopt,
                            int schema_version = 1,
                            const std::string& run_id = "",
                            std::optional<int> forced_file_version = std::nullopt,
                            const std::string& content_hash = "");
    std::vector<DatasetRecord> list_datasets();
    std::vector<DatasetFileRecord> list_dataset_files(const std::string& dataset_id);
    std::vector<DatasetFileVersionRecord> list_dataset_file_versions(
        const std::string& dataset_id,
        const std::string& file_path);
    void delete_dataset_file(const std::string& dataset_id,
                             const std::string& file_path,
                             const std::string& reason = "cleanup");
    int prune_empty_datasets();
    std::vector<DatasetTombstoneRecord> list_dataset_tombstones(
        const std::string& dataset_id,
        int limit = 100);

    // Schema registry
    void upsert_schema(const std::string& dataset_id,
                       int schema_version,
                       const std::string& schema_json,
                       const std::string& schema_hash,
                       bool set_active = true);
    std::optional<SchemaRecord> get_active_schema(const std::string& dataset_id);
    std::optional<int> find_schema_version_by_hash(const std::string& dataset_id,
                                                   const std::string& schema_hash);

    // Data quality events
    void record_quality_check(const QualityCheckRecord& check);
    std::vector<QualityCheckRecord> list_quality_checks(const std::string& dataset_id,
                                                        int limit = 100);

    // Streaming checkpoint (for real-time/high-frequency extension)
    void upsert_stream_checkpoint(const std::string& source,
                                  const std::string& stream,
                                  const std::string& shard,
                                  const std::string& cursor_payload,
                                  std::optional<Date> last_event_date = std::nullopt);
    std::optional<StreamCheckpointRecord> get_stream_checkpoint(const std::string& source,
                                                                const std::string& stream,
                                                                const std::string& shard);

    // Model training snapshots for reproducibility
    void record_training_snapshot(const TrainingSnapshotRecord& snapshot);
    std::vector<TrainingSnapshotRecord> list_training_snapshots(const std::string& dataset_id,
                                                                int limit = 100);

    // Broker account metadata and snapshots
    void upsert_broker_account(const BrokerAccount& account);
    std::optional<BrokerAccount> get_broker_account(const std::string& account_id);
    std::vector<BrokerAccount> list_broker_accounts(bool active_only = true);

    void upsert_account_cash(const AccountCashSnapshot& cash,
                             const std::string& source = "manual");
    std::optional<AccountCashSnapshot> latest_account_cash(const std::string& account_id);
    std::vector<AccountCashSnapshot> list_account_cash(const std::string& account_id,
                                                       int limit = 30);

    void upsert_account_position(const AccountPositionSnapshot& position,
                                 const std::string& source = "manual");
    std::vector<AccountPositionSnapshot> latest_account_positions(const std::string& account_id);
    std::vector<AccountPositionSnapshot> list_account_positions(
        const std::string& account_id,
        std::optional<Date> as_of_date = std::nullopt);

    void upsert_account_trade(const AccountTradeRecord& trade,
                              const std::string& source = "manual");
    std::vector<AccountTradeRecord> list_account_trades(const std::string& account_id,
                                                        int limit = 100);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace trade
