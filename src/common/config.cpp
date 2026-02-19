#include "trade/common/config.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace trade {
namespace {

YAML::Node merge_yaml(const YAML::Node& base, const YAML::Node& overlay) {
    if (!overlay || overlay.IsNull()) return base;
    if (!base || base.IsNull()) return overlay;

    if (!base.IsMap() || !overlay.IsMap()) {
        return overlay;
    }

    YAML::Node out(YAML::NodeType::Map);
    for (const auto& kv : base) {
        out[kv.first.Scalar()] = kv.second;
    }
    for (const auto& kv : overlay) {
        const std::string key = kv.first.Scalar();
        if (out[key] && out[key].IsMap() && kv.second.IsMap()) {
            out[key] = merge_yaml(out[key], kv.second);
        } else {
            out[key] = kv.second;
        }
    }
    return out;
}

YAML::Node load_yaml_with_includes(const std::filesystem::path& file_path,
                                   std::set<std::string>& visiting) {
    const auto canonical = std::filesystem::weakly_canonical(file_path).string();
    if (visiting.count(canonical)) {
        throw std::runtime_error("Recursive config include detected: " + canonical);
    }
    visiting.insert(canonical);

    YAML::Node raw = YAML::LoadFile(canonical);
    YAML::Node merged(YAML::NodeType::Map);

    if (raw["includes"] && raw["includes"].IsSequence()) {
        for (const auto& inc_node : raw["includes"]) {
            if (!inc_node.IsScalar()) continue;
            std::filesystem::path inc_path = inc_node.as<std::string>();
            if (inc_path.is_relative()) {
                inc_path = std::filesystem::path(canonical).parent_path() / inc_path;
            }
            YAML::Node inc_cfg = load_yaml_with_includes(inc_path, visiting);
            merged = merge_yaml(merged, inc_cfg);
        }
    }

    YAML::Node self_cfg(YAML::NodeType::Map);
    for (const auto& kv : raw) {
        const std::string key = kv.first.Scalar();
        if (key == "includes") continue;
        self_cfg[key] = kv.second;
    }

    visiting.erase(canonical);
    return merge_yaml(merged, self_cfg);
}

YAML::Node load_root_config(const std::string& path) {
    namespace fs = std::filesystem;
    fs::path p = path;
    if (fs::is_directory(p)) {
        p = p / "config.yaml";
    }
    if (!fs::exists(p)) {
        throw std::runtime_error("Config path not found: " + p.string());
    }
    std::set<std::string> visiting;
    return load_yaml_with_includes(p, visiting);
}

} // namespace

Config Config::load(const std::string& path) {
    Config cfg = defaults();
    try {
        YAML::Node root = load_root_config(path);

        if (auto n = root["data"]) {
            if (n["data_root"]) cfg.data.data_root = n["data_root"].as<std::string>();
            if (n["raw_dir"]) cfg.data.raw_dir = n["raw_dir"].as<std::string>();
            if (n["silver_dir"]) cfg.data.silver_dir = n["silver_dir"].as<std::string>();
            // Backward compatibility: old key curated_dir now maps to silver_dir.
            if (n["curated_dir"]) cfg.data.silver_dir = n["curated_dir"].as<std::string>();
            if (n["models_dir"]) cfg.data.models_dir = n["models_dir"].as<std::string>();
            if (n["market_daily_subpath"]) {
                cfg.data.market_daily_subpath = n["market_daily_subpath"].as<std::string>();
            }
        }

        if (auto n = root["ingestion"]) {
            if (n["default_history_days"]) {
                cfg.ingestion.default_history_days = n["default_history_days"].as<int>();
            }
            if (n["incremental_lookback_days"]) {
                cfg.ingestion.incremental_lookback_days = n["incremental_lookback_days"].as<int>();
            }
            if (n["min_start_date"]) cfg.ingestion.min_start_date = n["min_start_date"].as<std::string>();
            if (n["daily_bar_dataset"]) {
                cfg.ingestion.daily_bar_dataset = n["daily_bar_dataset"].as<std::string>();
            }
            if (n["write_raw_layer"]) {
                cfg.ingestion.write_raw_layer = n["write_raw_layer"].as<bool>();
            }
            if (n["write_silver_layer"]) {
                cfg.ingestion.write_silver_layer = n["write_silver_layer"].as<bool>();
            }
            // Backward compatibility: old curated naming maps to silver layer.
            if (n["write_curated_layer"]) {
                cfg.ingestion.write_silver_layer = n["write_curated_layer"].as<bool>();
            }
        }

        if (auto n = root["eastmoney"]) {
            if (n["timeout_ms"]) cfg.eastmoney.timeout_ms = n["timeout_ms"].as<int>();
            if (n["retry_count"]) cfg.eastmoney.retry_count = n["retry_count"].as<int>();
            if (n["retry_delay_ms"]) cfg.eastmoney.retry_delay_ms = n["retry_delay_ms"].as<int>();
            if (n["rate_limit_ms"]) cfg.eastmoney.rate_limit_ms = n["rate_limit_ms"].as<int>();
            if (n["forward_adjust"]) cfg.eastmoney.forward_adjust = n["forward_adjust"].as<bool>();
        }


        if (auto n = root["storage"]) {
            if (n["enabled"]) cfg.storage.enabled = n["enabled"].as<bool>();
            if (n["backend"]) cfg.storage.backend = n["backend"].as<std::string>();
            if (n["write_mode"]) cfg.storage.write_mode = n["write_mode"].as<std::string>();
            if (n["hot_days"]) cfg.storage.hot_days = n["hot_days"].as<int>();
            if (n["keep_local_cold_copy"]) {
                cfg.storage.keep_local_cold_copy = n["keep_local_cold_copy"].as<bool>();
            }
            if (n["mirror_hot_to_cloud"]) {
                cfg.storage.mirror_hot_to_cloud = n["mirror_hot_to_cloud"].as<bool>();
            }
            if (n["baidu_app_id"]) cfg.storage.baidu_app_id = n["baidu_app_id"].as<std::string>();
            if (n["baidu_root"]) cfg.storage.baidu_root = n["baidu_root"].as<std::string>();
            if (n["baidu_access_token"]) {
                cfg.storage.baidu_access_token = n["baidu_access_token"].as<std::string>();
            }
            if (n["baidu_refresh_token"]) {
                cfg.storage.baidu_refresh_token = n["baidu_refresh_token"].as<std::string>();
            }
            if (n["baidu_app_key"]) cfg.storage.baidu_app_key = n["baidu_app_key"].as<std::string>();
            if (n["baidu_app_secret"]) {
                cfg.storage.baidu_app_secret = n["baidu_app_secret"].as<std::string>();
            }
            if (n["baidu_sign_key"]) cfg.storage.baidu_sign_key = n["baidu_sign_key"].as<std::string>();
            if (n["baidu_timeout_ms"]) cfg.storage.baidu_timeout_ms = n["baidu_timeout_ms"].as<int>();
            if (n["baidu_retry_count"]) {
                cfg.storage.baidu_retry_count = n["baidu_retry_count"].as<int>();
            }
        }

        // Environment variable fallback for secrets
        if (cfg.storage.baidu_access_token.empty()) {
            if (const char* v = std::getenv("BAIDU_ACCESS_TOKEN")) {
                cfg.storage.baidu_access_token = v;
            }
        }
        if (cfg.storage.baidu_refresh_token.empty()) {
            if (const char* v = std::getenv("BAIDU_REFRESH_TOKEN")) {
                cfg.storage.baidu_refresh_token = v;
            }
        }
        if (cfg.storage.baidu_app_key.empty()) {
            if (const char* v = std::getenv("BAIDU_APP_KEY")) {
                cfg.storage.baidu_app_key = v;
            }
        }
        if (cfg.storage.baidu_app_secret.empty()) {
            if (const char* v = std::getenv("BAIDU_APP_SECRET")) {
                cfg.storage.baidu_app_secret = v;
            }
        }
        if (cfg.storage.baidu_app_id.empty()) {
            if (const char* v = std::getenv("BAIDU_APP_ID")) {
                cfg.storage.baidu_app_id = v;
            }
        }
        if (cfg.storage.baidu_sign_key.empty()) {
            if (const char* v = std::getenv("BAIDU_SIGN_KEY")) {
                cfg.storage.baidu_sign_key = v;
            }
        }

        if (auto n = root["cost"]) {
            if (n["stamp_tax_rate"]) cfg.cost.stamp_tax_rate = n["stamp_tax_rate"].as<double>();
            if (n["commission_rate"]) cfg.cost.commission_rate = n["commission_rate"].as<double>();
            if (n["commission_min_yuan"]) cfg.cost.commission_min_yuan = n["commission_min_yuan"].as<double>();
            if (n["transfer_fee_rate"]) cfg.cost.transfer_fee_rate = n["transfer_fee_rate"].as<double>();
        }

        if (auto n = root["risk"]) {
            if (n["target_annual_vol"]) cfg.risk.target_annual_vol = n["target_annual_vol"].as<double>();
            if (n["max_single_weight"]) cfg.risk.max_single_weight = n["max_single_weight"].as<double>();
            if (n["max_industry_weight"]) cfg.risk.max_industry_weight = n["max_industry_weight"].as<double>();
            if (n["base_cash_pct"]) cfg.risk.base_cash_pct = n["base_cash_pct"].as<double>();
        }

        if (auto n = root["backtest"]) {
            if (n["initial_capital"]) cfg.backtest.initial_capital = n["initial_capital"].as<double>();
            if (n["max_positions"]) cfg.backtest.max_positions = n["max_positions"].as<int>();
            if (n["min_positions"]) cfg.backtest.min_positions = n["min_positions"].as<int>();
        }

        if (auto n = root["sentiment"]) {
            if (n["dict_path"]) cfg.sentiment.dict_path = n["dict_path"].as<std::string>();
            if (n["onnx_model_path"]) cfg.sentiment.onnx_model_path = n["onnx_model_path"].as<std::string>();
            if (n["tokenizer_path"]) cfg.sentiment.tokenizer_path = n["tokenizer_path"].as<std::string>();
            if (n["use_onnx"]) cfg.sentiment.use_onnx = n["use_onnx"].as<bool>();
            if (n["rss_fetch_interval_min"]) {
                cfg.sentiment.rss_fetch_interval_min = n["rss_fetch_interval_min"].as<int>();
            }
            if (n["default_source"]) cfg.sentiment.default_source = n["default_source"].as<std::string>();
            if (n["default_history_days"]) {
                cfg.sentiment.default_history_days = n["default_history_days"].as<int>();
            }
            if (n["incremental_lookback_days"]) {
                cfg.sentiment.incremental_lookback_days = n["incremental_lookback_days"].as<int>();
            }
            if (n["rss_feeds"] && n["rss_feeds"].IsSequence()) {
                cfg.sentiment.rss_feeds.clear();
                for (const auto& feed : n["rss_feeds"]) {
                    SentimentFeedConfig f;
                    if (feed["name"]) f.name = feed["name"].as<std::string>();
                    if (feed["url"]) f.url = feed["url"].as<std::string>();
                    if (!f.name.empty() && !f.url.empty()) {
                        cfg.sentiment.rss_feeds.push_back(std::move(f));
                    }
                }
            }
        }

        spdlog::info("Config loaded from {}", path);
    } catch (const std::exception& e) {
        spdlog::warn("Failed to load config from {}: {}, using defaults", path, e.what());
    }
    return cfg;
}

Config Config::defaults() {
    return Config{};
}

} // namespace trade
