#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/eigen/dense.h>
#include "trade/features/feature_engine.h"
#include "trade/features/preprocessor.h"

namespace nb = nanobind;

void bind_features(nb::module_& m) {
    auto feat = m.def_submodule("features", "Feature engineering");

    // ========================================================================
    // PreprocessorConfig::StandardizeMode enum
    // ========================================================================
    nb::enum_<trade::PreprocessorConfig::StandardizeMode>(feat, "StandardizeMode")
        .value("ZScore", trade::PreprocessorConfig::StandardizeMode::kZScore)
        .value("QuantileRank", trade::PreprocessorConfig::StandardizeMode::kQuantileRank);

    // ========================================================================
    // PreprocessorConfig
    // ========================================================================
    nb::class_<trade::PreprocessorConfig>(feat, "PreprocessorConfig")
        .def(nb::init<>())
        // Missing value handling
        .def_rw("forward_fill", &trade::PreprocessorConfig::forward_fill)
        .def_rw("add_is_missing_flag", &trade::PreprocessorConfig::add_is_missing_flag)
        // Winsorization bounds (price/return features)
        .def_rw("price_lower_pct", &trade::PreprocessorConfig::price_lower_pct)
        .def_rw("price_upper_pct", &trade::PreprocessorConfig::price_upper_pct)
        // Winsorization bounds (fundamental features)
        .def_rw("fund_lower_pct", &trade::PreprocessorConfig::fund_lower_pct)
        .def_rw("fund_upper_pct", &trade::PreprocessorConfig::fund_upper_pct)
        // Feature names that use fundamental winsorization bounds
        .def_rw("fundamental_features", &trade::PreprocessorConfig::fundamental_features)
        // Neutralization toggles
        .def_rw("neutralize_industry", &trade::PreprocessorConfig::neutralize_industry)
        .def_rw("neutralize_market_cap", &trade::PreprocessorConfig::neutralize_market_cap)
        // Standardization mode
        .def_rw("mode", &trade::PreprocessorConfig::mode);

    // ========================================================================
    // FeatureSet
    // ========================================================================
    nb::class_<trade::FeatureSet>(feat, "FeatureSet")
        .def(nb::init<>())
        .def_rw("names", &trade::FeatureSet::names)
        .def_rw("symbols", &trade::FeatureSet::symbols)
        // Expose dates as vector<int> (days since epoch) via property
        .def_prop_rw("dates",
            [](const trade::FeatureSet& fs) {
                std::vector<int> v;
                v.reserve(fs.dates.size());
                for (auto& d : fs.dates) v.push_back(d.time_since_epoch().count());
                return v;
            },
            [](trade::FeatureSet& fs, const std::vector<int>& v) {
                fs.dates.clear();
                fs.dates.reserve(v.size());
                for (int d : v) fs.dates.push_back(trade::Date(std::chrono::days(d)));
            })
        .def_rw("matrix", &trade::FeatureSet::matrix)
        .def("num_features", &trade::FeatureSet::num_features)
        .def("num_observations", &trade::FeatureSet::num_observations)
        .def("col_index", &trade::FeatureSet::col_index)
        .def("column", &trade::FeatureSet::column)
        .def("merge", &trade::FeatureSet::merge);

    // ========================================================================
    // FeatureEngine::Config
    // ========================================================================
    nb::class_<trade::FeatureEngine::Config>(feat, "FeatureEngineConfig")
        .def(nb::init<>())
        .def_rw("fill_missing", &trade::FeatureEngine::Config::fill_missing)
        .def_rw("winsorize", &trade::FeatureEngine::Config::winsorize)
        .def_rw("neutralize", &trade::FeatureEngine::Config::neutralize)
        .def_rw("standardize", &trade::FeatureEngine::Config::standardize)
        .def_rw("standardize_mode", &trade::FeatureEngine::Config::standardize_mode)
        .def_rw("min_bar_count", &trade::FeatureEngine::Config::min_bar_count);

    // ========================================================================
    // FeatureEngine
    // ========================================================================
    nb::class_<trade::FeatureEngine>(feat, "FeatureEngine")
        .def(nb::init<>())
        .def(nb::init<trade::FeatureEngine::Config>())
        .def("compute_raw", &trade::FeatureEngine::compute_raw)
        .def("preprocess", &trade::FeatureEngine::preprocess)
        .def("build", &trade::FeatureEngine::build)
        .def("config", &trade::FeatureEngine::config, nb::rv_policy::reference_internal);

    // ========================================================================
    // Cross-sectional & time-series transform free functions
    // ========================================================================
    feat.def("cs_rank", &trade::cs_rank,
             nb::arg("v"),
             "Cross-sectional rank (fractional, 0-1)");
    feat.def("ts_zscore", &trade::ts_zscore,
             nb::arg("v"), nb::arg("lookback"),
             "Time-series z-score");
    feat.def("ewma", &trade::ewma,
             nb::arg("v"), nb::arg("halflife"),
             "Exponentially weighted moving average");
    feat.def("rolling_mean", &trade::rolling_mean,
             nb::arg("v"), nb::arg("window"),
             "Rolling window mean");
    feat.def("rolling_std", &trade::rolling_std,
             nb::arg("v"), nb::arg("window"),
             "Rolling window standard deviation");
    feat.def("rolling_sum", &trade::rolling_sum,
             nb::arg("v"), nb::arg("window"),
             "Rolling sum");
}
