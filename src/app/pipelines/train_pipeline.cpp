#include "trade/app/pipelines/train_pipeline.h"

#include "trade/common/time_utils.h"
#include "trade/features/feature_engine.h"
#include "trade/features/liquidity.h"
#include "trade/features/momentum.h"
#include "trade/features/volatility.h"
#include "trade/storage/metadata_store.h"
#include "trade/storage/parquet_reader.h"
#include "trade/storage/storage_path.h"

#ifdef HAVE_LIGHTGBM
#include "trade/ml/lgbm_model.h"
#include "trade/ml/model_evaluator.h"
#include "trade/ml/model_trainer.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <spdlog/spdlog.h>
#include <unordered_map>

namespace trade::app {
namespace {

std::vector<Bar> load_bars_for_train(const std::string& symbol,
                                     const Config& config) {
    StoragePath paths(config.data.data_root);
    std::vector<Bar> all_bars;
    const bool cloud_mode = config.storage.enabled &&
        (config.storage.backend == "baidu_netdisk" || config.storage.backend == "baidu");
    auto now = std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now());
    auto min_date = parse_date(config.ingestion.min_start_date);
    int start_year = date_year(min_date);
    int end_year = date_year(now);
    for (int year = start_year; year <= end_year; ++year) {
        auto path = paths.curated_daily(symbol, year);
        if (!std::filesystem::exists(path) && !cloud_mode) {
            continue;
        }
        try {
            auto bars = ParquetReader::read_bars(path);
            all_bars.insert(all_bars.end(), bars.begin(), bars.end());
        } catch (...) {}
    }
    return all_bars;
}

} // namespace

int run_train(const TrainRequest& request, const Config& config) {
#ifdef HAVE_LIGHTGBM
    if (request.symbol.empty()) {
        spdlog::error("--symbol required");
        return 1;
    }
    auto bars = load_bars_for_train(request.symbol, config);
    if (bars.size() < 252) {
        spdlog::error("Need >=252 bars, got {}", bars.size());
        return 1;
    }

    BarSeries series{request.symbol, bars};
    std::unordered_map<Symbol, Instrument> instruments;
    instruments[request.symbol] = Instrument{request.symbol};

    FeatureEngine engine;
    engine.register_calculator(std::make_unique<MomentumCalculator>());
    engine.register_calculator(std::make_unique<VolatilityCalculator>());
    engine.register_calculator(std::make_unique<LiquidityCalculator>());
    auto features = engine.build({series}, instruments);
    spdlog::info("Features: {} x {}", features.num_observations(), features.num_features());

    int n = features.num_observations();
    Eigen::VectorXd labels(n);
    labels.setZero();
    for (int i = 0; i + 5 < static_cast<int>(bars.size()); ++i) {
        if (bars[i].close > 0) {
            labels(i) = (bars[i + 5].close - bars[i].close) / bars[i].close;
        }
    }

    int split = static_cast<int>(n * 0.8);
    LGBMModel model;
    LGBMParams params;
    params.n_estimators = 300;
    params.learning_rate = 0.05;
    params.num_leaves = 31;

    auto result = model.train(
        features.matrix.topRows(split), labels.head(split), params,
        features.matrix.bottomRows(n - split), labels.tail(n - split));

    spdlog::info("Done: best_iter={} score={:.6f}", result.best_iteration, result.best_score);

    StoragePath paths(config.data.data_root);
    std::string mpath = paths.models_dir() + "/lgbm_factor_v1.model";
    model.save(mpath);
    spdlog::info("Saved to {}", mpath);

    try {
        MetadataStore metadata(paths.metadata_db());
        int schema_version = 1;
        auto datasets = metadata.list_datasets();
        auto it = std::find_if(datasets.begin(), datasets.end(), [](const auto& ds) {
            return ds.dataset_id == "curated.cn_a.daily";
        });
        if (it != datasets.end()) {
            schema_version = it->schema_version;
        }

        MetadataStore::TrainingSnapshotRecord snap;
        snap.snapshot_id = "lgbm_" +
            std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        snap.dataset_id = "curated.cn_a.daily";
        snap.query_spec = "symbol=" + request.symbol +
                          ",start=" + format_date(bars.front().date) +
                          ",end=" + format_date(bars.back().date);
        snap.snapshot_path = mpath;
        snap.start_date = bars.front().date;
        snap.end_date = bars.back().date;
        snap.row_count = features.num_observations();
        snap.schema_version = schema_version;
        snap.model_name = "lgbm_factor_v1";
        metadata.record_training_snapshot(snap);
    } catch (const std::exception& e) {
        spdlog::warn("Failed to record training snapshot metadata: {}", e.what());
    }

    auto imp = model.feature_importance_named(features.names, "gain");
    std::cout << "\nTop features:" << std::endl;
    for (int i = 0; i < std::min(20, static_cast<int>(imp.size())); ++i) {
        std::cout << "  " << std::left << std::setw(30) << imp[i].first
                  << std::fixed << std::setprecision(1) << imp[i].second << std::endl;
    }
    return 0;
#else
    (void)request;
    (void)config;
    spdlog::error("LightGBM not available");
    return 1;
#endif
}

} // namespace trade::app
