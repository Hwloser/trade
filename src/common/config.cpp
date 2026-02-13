#include "trade/common/config.h"
#include <fstream>
#include <spdlog/spdlog.h>

namespace trade {

Config Config::load(const std::string& path) {
    Config cfg = defaults();
    try {
        YAML::Node root = YAML::LoadFile(path);

        if (auto n = root["data"]) {
            if (n["data_root"]) cfg.data.data_root = n["data_root"].as<std::string>();
            if (n["raw_dir"]) cfg.data.raw_dir = n["raw_dir"].as<std::string>();
            if (n["curated_dir"]) cfg.data.curated_dir = n["curated_dir"].as<std::string>();
            if (n["models_dir"]) cfg.data.models_dir = n["models_dir"].as<std::string>();
        }

        if (auto n = root["akshare"]) {
            if (n["base_url"]) cfg.akshare.base_url = n["base_url"].as<std::string>();
            if (n["timeout_ms"]) cfg.akshare.timeout_ms = n["timeout_ms"].as<int>();
            if (n["retry_count"]) cfg.akshare.retry_count = n["retry_count"].as<int>();
            if (n["retry_delay_ms"]) cfg.akshare.retry_delay_ms = n["retry_delay_ms"].as<int>();
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
