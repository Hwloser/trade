#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>

#include "trade/regime/regime_detector.h"

namespace nb = nanobind;
using namespace nb::literals;

void bind_regime(nb::module_& parent) {
    auto m = parent.def_submodule("regime", "Market regime detection module");

    using RD = trade::RegimeDetector;

    // -----------------------------------------------------------------------
    // VolRegime enum
    // -----------------------------------------------------------------------
    nb::enum_<RD::VolRegime>(m, "VolRegime")
        .value("kLow",    RD::VolRegime::kLow)
        .value("kNormal", RD::VolRegime::kNormal)
        .value("kHigh",   RD::VolRegime::kHigh);

    // -----------------------------------------------------------------------
    // MarketBreadth
    // -----------------------------------------------------------------------
    nb::class_<RD::MarketBreadth>(m, "MarketBreadth")
        .def(nb::init<>())
        .def_rw("total_stocks", &RD::MarketBreadth::total_stocks)
        .def_rw("up_stocks",    &RD::MarketBreadth::up_stocks)
        .def_rw("limit_up",     &RD::MarketBreadth::limit_up)
        .def_rw("limit_down",   &RD::MarketBreadth::limit_down)
        .def("up_ratio",        &RD::MarketBreadth::up_ratio);

    // -----------------------------------------------------------------------
    // RegimeResult
    // -----------------------------------------------------------------------
    nb::class_<RD::RegimeResult>(m, "RegimeResult")
        .def(nb::init<>())
        .def_rw("market_regime",        &RD::RegimeResult::market_regime)
        .def_rw("vol_regime",           &RD::RegimeResult::vol_regime)
        .def_rw("index_price",          &RD::RegimeResult::index_price)
        .def_rw("dma_120",              &RD::RegimeResult::dma_120)
        .def_rw("index_above_dma_pct",  &RD::RegimeResult::index_above_dma_pct)
        .def_rw("up_ratio",             &RD::RegimeResult::up_ratio)
        .def_rw("annualised_vol",       &RD::RegimeResult::annualised_vol)
        .def_rw("single_day_return",    &RD::RegimeResult::single_day_return)
        .def_rw("vol_quantile",         &RD::RegimeResult::vol_quantile)
        .def_rw("vol_tercile_low",      &RD::RegimeResult::vol_tercile_low)
        .def_rw("vol_tercile_high",     &RD::RegimeResult::vol_tercile_high)
        .def_rw("trend_slope",          &RD::RegimeResult::trend_slope)
        .def_rw("trend_down",           &RD::RegimeResult::trend_down)
        .def_rw("shock_vol_trigger",    &RD::RegimeResult::shock_vol_trigger)
        .def_rw("shock_day_trigger",    &RD::RegimeResult::shock_day_trigger)
        .def("regime_name",             &RD::RegimeResult::regime_name)
        .def("vol_regime_name",         &RD::RegimeResult::vol_regime_name);

    // -----------------------------------------------------------------------
    // Config
    // -----------------------------------------------------------------------
    nb::class_<RD::Config>(m, "RegimeDetectorConfig")
        .def(nb::init<>())
        .def_rw("dma_period",                  &RD::Config::dma_period)
        .def_rw("bull_up_ratio_min",            &RD::Config::bull_up_ratio_min)
        .def_rw("bull_vol_max",                 &RD::Config::bull_vol_max)
        .def_rw("bear_up_ratio_max",            &RD::Config::bear_up_ratio_max)
        .def_rw("shock_vol_threshold",          &RD::Config::shock_vol_threshold)
        .def_rw("shock_day_return_threshold",   &RD::Config::shock_day_return_threshold)
        .def_rw("vol_history_days",             &RD::Config::vol_history_days)
        .def_rw("realized_vol_window",          &RD::Config::realized_vol_window)
        .def_rw("trend_window",                 &RD::Config::trend_window)
        .def_rw("min_persistence_days",         &RD::Config::min_persistence_days);

    // -----------------------------------------------------------------------
    // RegimeDetector
    // -----------------------------------------------------------------------
    nb::class_<RD>(m, "RegimeDetector")
        .def(nb::init<>())
        .def(nb::init<RD::Config>(), "config"_a)
        .def("detect", &RD::detect,
             "index_prices"_a, "market_breadth"_a)
        .def("detect_vol_regime", &RD::detect_vol_regime,
             "vol_history"_a)
        .def("update", &RD::update,
             "index_prices"_a, "market_breadth"_a)
        .def("current_regime",     &RD::current_regime)
        .def("current_vol_regime", &RD::current_vol_regime)
        .def("regime_duration",    &RD::regime_duration)
        .def_static("sma", &RD::sma,
                     "prices"_a, "period"_a)
        .def_static("realized_vol", &RD::realized_vol,
                     "prices"_a, "window"_a)
        .def_static("trend_slope", &RD::trend_slope,
                     "prices"_a, "window"_a)
        .def_static("quantile_rank", &RD::quantile_rank,
                     "value"_a, "distribution"_a)
        .def("config", &RD::config, nb::rv_policy::reference_internal);
}
