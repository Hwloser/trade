#include <gtest/gtest.h>
#include "trade/storage/metadata_store.h"
#include "trade/model/instrument.h"

#include <filesystem>
#include <memory>
#include <sqlite3.h>

using namespace trade;

// =============================================================================
// Helper: create a date from year/month/day
// =============================================================================
static Date make_date(int year, int month, int day) {
    return std::chrono::sys_days{
        std::chrono::year{year} / std::chrono::month{static_cast<unsigned>(month)} /
        std::chrono::day{static_cast<unsigned>(day)}};
}

// =============================================================================
// Helper: create a test instrument
// =============================================================================
static Instrument make_instrument(const std::string& symbol,
                                   const std::string& name,
                                   Market market,
                                   Board board,
                                   SWIndustry industry) {
    Instrument inst;
    inst.symbol = symbol;
    inst.name = name;
    inst.market = market;
    inst.board = board;
    inst.industry = industry;
    inst.list_date = make_date(2000, 1, 1);
    inst.status = TradingStatus::kNormal;
    return inst;
}

// =============================================================================
// Fixture: creates an in-memory SQLite metadata store
// =============================================================================
class MetadataStoreTest : public ::testing::Test {
protected:
    std::unique_ptr<MetadataStore> store_;

    void SetUp() override {
        store_ = std::make_unique<MetadataStore>(":memory:");
    }
};

// =============================================================================
// Instrument CRUD tests
// =============================================================================

TEST_F(MetadataStoreTest, UpsertAndGetInstrument) {
    auto inst = make_instrument("600000.SH", "Pudong Bank", Market::kSH,
                                Board::kMain, SWIndustry::kBanking);
    store_->upsert_instrument(inst);

    auto result = store_->get_instrument("600000.SH");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->symbol, "600000.SH");
    EXPECT_EQ(result->name, "Pudong Bank");
    EXPECT_EQ(result->market, Market::kSH);
    EXPECT_EQ(result->market_name, "Shanghai");
    EXPECT_EQ(result->market_label(), "Shanghai");
    EXPECT_EQ(result->board, Board::kMain);
    EXPECT_EQ(result->industry, SWIndustry::kBanking);
}

TEST_F(MetadataStoreTest, GetNonExistentInstrument) {
    auto result = store_->get_instrument("999999.XX");
    EXPECT_FALSE(result.has_value());
}

TEST_F(MetadataStoreTest, UpsertUpdatesExisting) {
    auto inst = make_instrument("600000.SH", "Pudong Bank", Market::kSH,
                                Board::kMain, SWIndustry::kBanking);
    store_->upsert_instrument(inst);

    // Update the name
    inst.name = "Pudong Development Bank";
    store_->upsert_instrument(inst);

    auto result = store_->get_instrument("600000.SH");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->name, "Pudong Development Bank");
}

TEST_F(MetadataStoreTest, GetAllInstruments) {
    store_->upsert_instrument(
        make_instrument("600000.SH", "A", Market::kSH, Board::kMain, SWIndustry::kBanking));
    store_->upsert_instrument(
        make_instrument("000001.SZ", "B", Market::kSZ, Board::kMain, SWIndustry::kBanking));
    store_->upsert_instrument(
        make_instrument("300001.SZ", "C", Market::kSZ, Board::kChiNext, SWIndustry::kComputer));

    auto all = store_->get_all_instruments();
    EXPECT_EQ(all.size(), 3u);
}

// =============================================================================
// Query by market/industry tests
// =============================================================================

TEST_F(MetadataStoreTest, GetInstrumentsByMarket) {
    store_->upsert_instrument(
        make_instrument("600000.SH", "A", Market::kSH, Board::kMain, SWIndustry::kBanking));
    store_->upsert_instrument(
        make_instrument("600001.SH", "B", Market::kSH, Board::kMain, SWIndustry::kComputer));
    store_->upsert_instrument(
        make_instrument("000001.SZ", "C", Market::kSZ, Board::kMain, SWIndustry::kBanking));

    auto sh_instruments = store_->get_instruments_by_market(Market::kSH);
    EXPECT_EQ(sh_instruments.size(), 2u);

    auto sz_instruments = store_->get_instruments_by_market(Market::kSZ);
    EXPECT_EQ(sz_instruments.size(), 1u);

    auto hk_instruments = store_->get_instruments_by_market(Market::kHK);
    EXPECT_EQ(hk_instruments.size(), 0u);
}

TEST_F(MetadataStoreTest, GetInstrumentsByIndustry) {
    store_->upsert_instrument(
        make_instrument("600000.SH", "A", Market::kSH, Board::kMain, SWIndustry::kBanking));
    store_->upsert_instrument(
        make_instrument("000001.SZ", "B", Market::kSZ, Board::kMain, SWIndustry::kBanking));
    store_->upsert_instrument(
        make_instrument("300001.SZ", "C", Market::kSZ, Board::kChiNext, SWIndustry::kComputer));

    auto banking = store_->get_instruments_by_industry(SWIndustry::kBanking);
    EXPECT_EQ(banking.size(), 2u);

    auto computer = store_->get_instruments_by_industry(SWIndustry::kComputer);
    EXPECT_EQ(computer.size(), 1u);

    auto mining = store_->get_instruments_by_industry(SWIndustry::kMining);
    EXPECT_EQ(mining.size(), 0u);
}

// =============================================================================
// Download tracking tests
// =============================================================================

TEST_F(MetadataStoreTest, RecordDownloadAndQueryLastDate) {
    Date start = make_date(2024, 1, 1);
    Date end = make_date(2024, 1, 31);

    store_->record_download("600000.SH", start, end, 22);

    auto last_date = store_->last_download_date("600000.SH");
    ASSERT_TRUE(last_date.has_value());
    EXPECT_EQ(*last_date, end);
}

TEST_F(MetadataStoreTest, LastDownloadDateNonExistent) {
    auto last_date = store_->last_download_date("999999.XX");
    EXPECT_FALSE(last_date.has_value());
}

TEST_F(MetadataStoreTest, RecordDownloadUpdatesLastDate) {
    Date start1 = make_date(2024, 1, 1);
    Date end1 = make_date(2024, 1, 31);
    store_->record_download("600000.SH", start1, end1, 22);

    Date start2 = make_date(2024, 2, 1);
    Date end2 = make_date(2024, 2, 28);
    store_->record_download("600000.SH", start2, end2, 19);

    auto last_date = store_->last_download_date("600000.SH");
    ASSERT_TRUE(last_date.has_value());
    EXPECT_EQ(*last_date, end2);
}

TEST_F(MetadataStoreTest, SymbolsNeedingUpdate) {
    Date end1 = make_date(2024, 1, 15);
    Date end2 = make_date(2024, 2, 15);

    // Instrument A: downloaded up to Jan 15
    store_->upsert_instrument(
        make_instrument("600000.SH", "A", Market::kSH, Board::kMain, SWIndustry::kBanking));
    store_->record_download("600000.SH", make_date(2024, 1, 1), end1, 10);

    // Instrument B: downloaded up to Feb 15
    store_->upsert_instrument(
        make_instrument("000001.SZ", "B", Market::kSZ, Board::kMain, SWIndustry::kBanking));
    store_->record_download("000001.SZ", make_date(2024, 2, 1), end2, 10);

    // Cutoff: Feb 1 -- only A should need update (its last download is Jan 15 < Feb 1)
    Date cutoff = make_date(2024, 2, 1);
    auto needing_update = store_->symbols_needing_update(cutoff);

    EXPECT_EQ(needing_update.size(), 1u);
    if (!needing_update.empty()) {
        EXPECT_EQ(needing_update[0], "600000.SH");
    }
}

// =============================================================================
// File-based MetadataStore test (temp file)
// =============================================================================

class MetadataStoreFileTest : public ::testing::Test {
protected:
    std::string temp_db_path_;

    void SetUp() override {
        temp_db_path_ = "/tmp/trade_test_metadata.db";
        std::filesystem::remove(temp_db_path_);
    }

    void TearDown() override {
        std::filesystem::remove(temp_db_path_);
    }
};

TEST_F(MetadataStoreFileTest, PersistAcrossInstances) {
    {
        MetadataStore store(temp_db_path_);
        store.upsert_instrument(
            make_instrument("600000.SH", "Test", Market::kSH, Board::kMain, SWIndustry::kBanking));
    }

    // Re-open the store from the same file
    {
        MetadataStore store(temp_db_path_);
        auto result = store.get_instrument("600000.SH");
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result->symbol, "600000.SH");
        EXPECT_EQ(result->name, "Test");
    }
}

TEST_F(MetadataStoreFileTest, RequestFingerprintExpiresOutsideWindow) {
    const auto start = make_date(2024, 1, 15);
    const auto end = make_date(2024, 1, 15);
    {
        MetadataStore store(temp_db_path_);
        store.record_request_fingerprint("eastmoney",
                                         "cn_a_daily_bar",
                                         "600000.SH",
                                         start,
                                         end,
                                         "success",
                                         "run-1",
                                         1);
        EXPECT_TRUE(store.has_recent_successful_request("eastmoney",
                                                        "cn_a_daily_bar",
                                                        "600000.SH",
                                                        start,
                                                        end,
                                                        24));
    }

    sqlite3* db = nullptr;
    ASSERT_EQ(sqlite3_open(temp_db_path_.c_str(), &db), SQLITE_OK);
    char* err = nullptr;
    const char* sql = R"(
        UPDATE request_fingerprints
           SET updated_at = '2000-01-01 00:00:00'
         WHERE source = 'eastmoney'
           AND dataset = 'cn_a_daily_bar'
           AND symbol = '600000.SH'
           AND start_date = '2024-01-15'
           AND end_date = '2024-01-15'
    )";
    ASSERT_EQ(sqlite3_exec(db, sql, nullptr, nullptr, &err), SQLITE_OK) << (err ? err : "");
    if (err) sqlite3_free(err);
    sqlite3_close(db);

    MetadataStore store(temp_db_path_);
    EXPECT_FALSE(store.has_recent_successful_request("eastmoney",
                                                     "cn_a_daily_bar",
                                                     "600000.SH",
                                                     start,
                                                     end,
                                                     24));
}

// =============================================================================
// Incremental watermark + ingestion run tests
// =============================================================================

TEST_F(MetadataStoreTest, UpsertAndGetWatermarkDate) {
    auto date1 = make_date(2024, 1, 31);
    auto date2 = make_date(2024, 2, 29);

    store_->upsert_watermark("eastmoney", "cn_a_daily_bar", "600000.SH", date1);
    auto wm1 = store_->last_watermark_date("eastmoney", "cn_a_daily_bar", "600000.SH");
    ASSERT_TRUE(wm1.has_value());
    EXPECT_EQ(*wm1, date1);

    store_->upsert_watermark("eastmoney", "cn_a_daily_bar", "600000.SH", date2,
                             R"({"cursor":"test"})");
    auto wm2 = store_->last_watermark_date("eastmoney", "cn_a_daily_bar", "600000.SH");
    ASSERT_TRUE(wm2.has_value());
    EXPECT_EQ(*wm2, date2);
}

TEST_F(MetadataStoreTest, IngestionRunLifecycle) {
    const std::string run_id = "run-test-001";
    store_->begin_ingestion_run(run_id, "eastmoney", "cn_a_daily_bar", "600000.SH", "incremental");
    store_->finish_ingestion_run(run_id, true, 100, 90);

    // No read API yet; lifecycle should complete without throw/crash.
    SUCCEED();
}

TEST_F(MetadataStoreTest, RequestFingerprintDedupHitAndStatusFilter) {
    const auto start = make_date(2024, 1, 15);
    const auto end = make_date(2024, 1, 15);

    EXPECT_FALSE(store_->has_recent_successful_request("eastmoney",
                                                       "cn_a_daily_bar",
                                                       "600000.SH",
                                                       start,
                                                       end,
                                                       24));

    store_->record_request_fingerprint("eastmoney",
                                       "cn_a_daily_bar",
                                       "600000.SH",
                                       start,
                                       end,
                                       "failed",
                                       "run-failed",
                                       0);
    EXPECT_FALSE(store_->has_recent_successful_request("eastmoney",
                                                       "cn_a_daily_bar",
                                                       "600000.SH",
                                                       start,
                                                       end,
                                                       24));

    store_->record_request_fingerprint("eastmoney",
                                       "cn_a_daily_bar",
                                       "600000.SH",
                                       start,
                                       end,
                                       "success",
                                       "run-success",
                                       10);
    EXPECT_TRUE(store_->has_recent_successful_request("eastmoney",
                                                      "cn_a_daily_bar",
                                                      "600000.SH",
                                                      start,
                                                      end,
                                                      24));

    EXPECT_FALSE(store_->has_recent_successful_request("eastmoney",
                                                       "cn_a_daily_bar",
                                                       "600000.SH",
                                                       start,
                                                       end + std::chrono::days{1},
                                                       24));
    EXPECT_FALSE(store_->has_recent_successful_request("eastmoney",
                                                       "cn_a_daily_bar",
                                                       "600000.SH",
                                                       start,
                                                       end,
                                                       0));
}

TEST_F(MetadataStoreTest, DatasetCatalogSchemaAndVersionLifecycle) {
    auto d1 = make_date(2024, 1, 3);
    auto d2 = make_date(2024, 1, 5);

    store_->upsert_schema("silver.cn_a.daily", 1, "schema_v1", "hash_v1", true);
    auto s1 = store_->get_active_schema("silver.cn_a.daily");
    ASSERT_TRUE(s1.has_value());
    EXPECT_EQ(s1->schema_version, 1);
    EXPECT_EQ(s1->schema_hash, "hash_v1");

    int v1 = store_->upsert_dataset_file("silver.cn_a.daily",
                                         "silver",
                                         "cn_a",
                                         "daily",
                                         "silver/cn_a/daily",
                                         "silver/cn_a/daily/2024/600000.SH.parquet",
                                         100,
                                         d1,
                                         1,
                                         "run-1");
    int v2 = store_->upsert_dataset_file("silver.cn_a.daily",
                                         "silver",
                                         "cn_a",
                                         "daily",
                                         "silver/cn_a/daily",
                                         "silver/cn_a/daily/2024/600000.SH.parquet",
                                         120,
                                         d2,
                                         1,
                                         "run-2");
    EXPECT_EQ(v1, 1);
    EXPECT_EQ(v2, 2);

    auto files = store_->list_dataset_files("silver.cn_a.daily");
    ASSERT_EQ(files.size(), 1u);
    EXPECT_EQ(files[0].row_count, 120);
    EXPECT_EQ(files[0].current_version, 2);
    ASSERT_TRUE(files[0].max_event_date.has_value());
    EXPECT_EQ(*files[0].max_event_date, d2);

    auto vers = store_->list_dataset_file_versions(
        "silver.cn_a.daily",
        "silver/cn_a/daily/2024/600000.SH.parquet");
    ASSERT_EQ(vers.size(), 2u);
    EXPECT_EQ(vers[0].version, 2);
    EXPECT_EQ(vers[0].run_id, "run-2");
    EXPECT_EQ(vers[1].version, 1);
}

TEST_F(MetadataStoreTest, DatasetTombstoneLifecyclePurge) {
    auto d1 = make_date(2024, 1, 3);
    store_->upsert_dataset_file("raw.cn_a.daily",
                                "raw",
                                "cn_a",
                                "daily",
                                "raw/cn_a/daily",
                                "raw/cn_a/daily/2024/600000.SH.parquet",
                                100,
                                d1,
                                1,
                                "run-1");

    auto files_before = store_->list_dataset_files("raw.cn_a.daily");
    ASSERT_EQ(files_before.size(), 1u);

    store_->delete_dataset_file("raw.cn_a.daily",
                                "raw/cn_a/daily/2024/600000.SH.parquet",
                                "unit_test_delete");
    auto files_after = store_->list_dataset_files("raw.cn_a.daily");
    EXPECT_TRUE(files_after.empty());

    auto tombstones = store_->list_dataset_tombstones("raw.cn_a.daily", 10);
    ASSERT_EQ(tombstones.size(), 1u);
    EXPECT_EQ(tombstones[0].reason, "unit_test_delete");

    int purged = store_->purge_dataset_tombstones("raw.cn_a.daily", 0);
    EXPECT_EQ(purged, 1);
    auto after_purge = store_->list_dataset_tombstones("raw.cn_a.daily", 10);
    EXPECT_TRUE(after_purge.empty());
}

TEST_F(MetadataStoreTest, QualityCheckpointAndTrainingSnapshot) {
    MetadataStore::QualityCheckRecord qc;
    qc.run_id = "run-qc-1";
    qc.dataset_id = "silver.cn_a.daily";
    qc.check_name = "quality_score";
    qc.status = "pass";
    qc.severity = "info";
    qc.metric_value = 0.99;
    qc.threshold_value = 0.95;
    qc.message = "ok";
    qc.event_date = make_date(2024, 2, 1);
    store_->record_quality_check(qc);

    auto checks = store_->list_quality_checks("silver.cn_a.daily", 10);
    ASSERT_EQ(checks.size(), 1u);
    EXPECT_EQ(checks[0].run_id, "run-qc-1");
    EXPECT_EQ(checks[0].check_name, "quality_score");

    store_->upsert_stream_checkpoint("eastmoney",
                                     "cn_a_daily_bar",
                                     "600000.SH",
                                     R"({"cursor":"abc"})",
                                     make_date(2024, 2, 2));
    auto cp = store_->get_stream_checkpoint("eastmoney", "cn_a_daily_bar", "600000.SH");
    ASSERT_TRUE(cp.has_value());
    EXPECT_EQ(cp->cursor_payload, R"({"cursor":"abc"})");
    ASSERT_TRUE(cp->last_event_date.has_value());
    EXPECT_EQ(*cp->last_event_date, make_date(2024, 2, 2));

    MetadataStore::TrainingSnapshotRecord snap;
    snap.snapshot_id = "snap-1";
    snap.dataset_id = "silver.cn_a.daily";
    snap.query_spec = "symbol=600000.SH";
    snap.snapshot_path = "data/models/lgbm_factor_v1.model";
    snap.start_date = make_date(2024, 1, 1);
    snap.end_date = make_date(2024, 2, 1);
    snap.row_count = 1234;
    snap.schema_version = 1;
    snap.model_name = "lgbm";
    store_->record_training_snapshot(snap);

    auto snaps = store_->list_training_snapshots("silver.cn_a.daily", 10);
    ASSERT_EQ(snaps.size(), 1u);
    EXPECT_EQ(snaps[0].snapshot_id, "snap-1");
    EXPECT_EQ(snaps[0].row_count, 1234);
}

TEST_F(MetadataStoreTest, BrokerAccountSnapshotRoundTrip) {
    BrokerAccount account;
    account.account_id = "acc_demo_001";
    account.broker = "ths";
    account.account_name = "Main";
    account.auth_payload = R"({"token":"demo"})";
    account.is_active = true;
    store_->upsert_broker_account(account);

    auto fetched_account = store_->get_broker_account("acc_demo_001");
    ASSERT_TRUE(fetched_account.has_value());
    EXPECT_EQ(fetched_account->broker, "ths");
    EXPECT_EQ(fetched_account->account_name, "Main");

    auto all_accounts = store_->list_broker_accounts(true);
    ASSERT_EQ(all_accounts.size(), 1u);
    EXPECT_EQ(all_accounts[0].account_id, "acc_demo_001");

    AccountCashSnapshot cash;
    cash.account_id = "acc_demo_001";
    cash.as_of_date = make_date(2024, 2, 5);
    cash.total_asset = 1000000.0;
    cash.cash = 250000.0;
    cash.available_cash = 240000.0;
    cash.frozen_cash = 10000.0;
    cash.market_value = 750000.0;
    store_->upsert_account_cash(cash, "unit_test");

    auto latest_cash = store_->latest_account_cash("acc_demo_001");
    ASSERT_TRUE(latest_cash.has_value());
    EXPECT_EQ(latest_cash->as_of_date, make_date(2024, 2, 5));
    EXPECT_NEAR(latest_cash->total_asset, 1000000.0, 1e-6);

    AccountPositionSnapshot pos1;
    pos1.account_id = "acc_demo_001";
    pos1.as_of_date = make_date(2024, 2, 5);
    pos1.symbol = "600000.SH";
    pos1.quantity = 10000;
    pos1.available_quantity = 9000;
    pos1.cost_price = 10.1;
    pos1.last_price = 10.5;
    pos1.market_value = 105000.0;
    pos1.unrealized_pnl = 4000.0;
    pos1.unrealized_pnl_ratio = 0.0396;
    store_->upsert_account_position(pos1, "unit_test");

    AccountPositionSnapshot pos2 = pos1;
    pos2.symbol = "000001.SZ";
    pos2.quantity = 5000;
    pos2.available_quantity = 5000;
    pos2.cost_price = 12.0;
    pos2.last_price = 11.8;
    pos2.market_value = 59000.0;
    pos2.unrealized_pnl = -1000.0;
    pos2.unrealized_pnl_ratio = -0.0167;
    store_->upsert_account_position(pos2, "unit_test");

    auto latest_pos = store_->latest_account_positions("acc_demo_001");
    ASSERT_EQ(latest_pos.size(), 2u);

    AccountTradeRecord trade;
    trade.account_id = "acc_demo_001";
    trade.trade_id = "T202402050001";
    trade.trade_date = make_date(2024, 2, 5);
    trade.symbol = "600000.SH";
    trade.side = Side::kBuy;
    trade.price = 10.2;
    trade.quantity = 1000;
    trade.amount = 10200.0;
    trade.fee = 5.0;
    store_->upsert_account_trade(trade, "unit_test");

    auto trades = store_->list_account_trades("acc_demo_001", 10);
    ASSERT_EQ(trades.size(), 1u);
    EXPECT_EQ(trades[0].trade_id, "T202402050001");
    EXPECT_EQ(trades[0].side, Side::kBuy);
}
