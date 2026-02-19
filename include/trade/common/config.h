#pragma once

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace trade {

struct DataConfig {
    std::string data_root = "data";
    std::string raw_dir = "raw";
    std::string curated_dir = "curated";
    std::string models_dir = "models";
    std::string market_daily_subpath = "cn_a/daily";
};

struct IngestionConfig {
    int default_history_days = 30;             // default backfill window
    int incremental_lookback_days = 5;         // overlap window for correction
    std::string min_start_date = "2020-01-01"; // lower bound for historical pulls
    std::string daily_bar_dataset = "cn_a_daily_bar";
};

struct EastMoneyConfig {
    int timeout_ms = 30000;
    int retry_count = 3;
    int retry_delay_ms = 1000;
    int rate_limit_ms = 200;
    bool forward_adjust = true;
};


struct StorageConfig {
    bool enabled = false;
    std::string backend = "local";  // local, baidu_netdisk

    // Write policy:
    // - local_only: write to local disk only
    // - hybrid: keep hot partitions local, write cold partitions to cloud
    // - cloud_only: write directly to cloud, no local persistence
    std::string write_mode = "local_only";

    // Hot/cold partitioning
    int hot_days = 30;
    bool keep_local_cold_copy = false;
    bool mirror_hot_to_cloud = false;

    // Baidu Netdisk OpenAPI credentials
    std::string baidu_app_id = "";
    std::string baidu_root = "/apps/trade";
    std::string baidu_access_token = "";
    std::string baidu_refresh_token = "";
    std::string baidu_app_key = "";
    std::string baidu_app_secret = "";
    std::string baidu_sign_key = "";
    int baidu_timeout_ms = 30000;
    int baidu_retry_count = 2;
};

struct TradingCostConfig {
    double stamp_tax_rate = 0.0005;        // 印花税 0.05% (卖出)
    double commission_rate = 0.00025;       // 佣金 0.025% (双向)
    double commission_min_yuan = 5.0;       // 最低佣金 5元
    double transfer_fee_rate = 0.00001;     // 过户费 0.001% (沪市)
};

struct RiskConfig {
    double target_annual_vol = 0.11;       // 目标年化波动率 11%
    double max_single_weight = 0.10;       // 单股硬限 10%
    double soft_single_weight = 0.08;      // 单股软限 8%
    double max_industry_weight = 0.35;     // 行业硬限 35%
    double soft_industry_weight = 0.30;    // 行业软限 30%
    double max_top3_weight = 0.22;         // 前3大持仓上限 22%
    double max_style_z = 1.0;             // 风格因子z暴露上限
    double beta_low = 0.6;                // Beta下限
    double beta_high = 1.2;              // Beta上限
    double max_liquidation_days = 2.5;    // 最大清算天数
    double base_cash_pct = 0.10;          // 基础现金比例
    double drawdown_cash_level2 = 0.20;   // 二级回撤现金
    double drawdown_cash_shock = 0.35;    // 冲击市现金
};

struct BacktestConfig {
    double initial_capital = 1000000.0;
    int max_positions = 25;
    int min_positions = 15;
    double min_adv_participation = 0.08;
    double max_adv_participation = 0.12;
    double rebalance_threshold = 0.01;     // 权重偏差<1%不调
    double alpha_cost_multiple = 1.5;      // alpha > 1.5*cost才交易
};

struct SentimentFeedConfig {
    std::string name;
    std::string url;
};

struct SentimentConfig {
    std::string dict_path = "config/sentiment_dict.txt";
    std::string onnx_model_path = "";
    std::string tokenizer_path = "";
    bool use_onnx = false;                 // false = rule engine only
    int rss_fetch_interval_min = 15;
    std::string default_source = "rss";
    int default_history_days = 30;
    int incremental_lookback_days = 1;
    std::vector<SentimentFeedConfig> rss_feeds;
};

struct Config {
    DataConfig data;
    IngestionConfig ingestion;
    EastMoneyConfig eastmoney;
    StorageConfig storage;
    TradingCostConfig cost;
    RiskConfig risk;
    BacktestConfig backtest;
    SentimentConfig sentiment;

    static Config load(const std::string& path);
    static Config defaults();
};

} // namespace trade
