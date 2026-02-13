#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/eigen/dense.h>
#include "trade/common/types.h"
#include "trade/decision/signal.h"
#include "trade/decision/signal_combiner.h"
#include "trade/decision/universe_filter.h"
#include "trade/decision/portfolio_opt.h"
#include "trade/decision/order_manager.h"
#include "trade/decision/pre_trade_check.h"
#include "trade/decision/decision_report.h"
#include "trade/backtest/backtest_engine.h"

namespace nb = nanobind;
using namespace nb::literals;

void bind_decision(nb::module_& m) {
    auto decision = m.def_submodule("decision", "Decision and order management module");

    // -----------------------------------------------------------------------
    // Signal::ModelScore
    // -----------------------------------------------------------------------

    nb::class_<trade::Signal::ModelScore>(decision, "ModelScore")
        .def(nb::init<>())
        .def_rw("model_name", &trade::Signal::ModelScore::model_name)
        .def_rw("raw_score", &trade::Signal::ModelScore::raw_score)
        .def_rw("calibrated_score", &trade::Signal::ModelScore::calibrated_score)
        .def_rw("weight", &trade::Signal::ModelScore::weight);

    // -----------------------------------------------------------------------
    // Signal::SentimentOverlay
    // -----------------------------------------------------------------------

    nb::class_<trade::Signal::SentimentOverlay>(decision, "SentimentOverlay")
        .def(nb::init<>())
        .def_rw("stock_mood", &trade::Signal::SentimentOverlay::stock_mood)
        .def_rw("neg_shock", &trade::Signal::SentimentOverlay::neg_shock)
        .def_rw("key_news", &trade::Signal::SentimentOverlay::key_news);

    // -----------------------------------------------------------------------
    // Signal
    // -----------------------------------------------------------------------

    nb::class_<trade::Signal>(decision, "Signal")
        .def(nb::init<>())
        .def_rw("symbol", &trade::Signal::symbol)
        .def_rw("alpha_score", &trade::Signal::alpha_score)
        .def_rw("confidence", &trade::Signal::confidence)
        .def_rw("regime", &trade::Signal::regime)
        .def_rw("is_conflict", &trade::Signal::is_conflict)
        .def_rw("model_scores", &trade::Signal::model_scores)
        .def_rw("sentiment", &trade::Signal::sentiment)
        .def("is_tradable", &trade::Signal::is_tradable)
        .def("has_neg_shock", &trade::Signal::has_neg_shock)
        .def("num_models", &trade::Signal::num_models);

    // -----------------------------------------------------------------------
    // SignalCombiner::Config
    // -----------------------------------------------------------------------

    nb::class_<trade::SignalCombiner::Config>(decision, "SignalCombinerConfig")
        .def(nb::init<>())
        .def_rw("zscore_lookback", &trade::SignalCombiner::Config::zscore_lookback)
        .def_rw("icir_lookback", &trade::SignalCombiner::Config::icir_lookback)
        .def_rw("regime_fit_floor", &trade::SignalCombiner::Config::regime_fit_floor)
        .def_rw("regime_fit_cap", &trade::SignalCombiner::Config::regime_fit_cap)
        .def_rw("instability_penalty_coeff",
            &trade::SignalCombiner::Config::instability_penalty_coeff)
        .def_rw("turnover_threshold", &trade::SignalCombiner::Config::turnover_threshold)
        .def_rw("drawdown_threshold", &trade::SignalCombiner::Config::drawdown_threshold)
        .def_rw("conflict_alpha_threshold",
            &trade::SignalCombiner::Config::conflict_alpha_threshold)
        .def_rw("conflict_dispersion_threshold",
            &trade::SignalCombiner::Config::conflict_dispersion_threshold)
        .def_rw("min_confidence", &trade::SignalCombiner::Config::min_confidence);

    // -----------------------------------------------------------------------
    // SignalCombiner::ModelMeta
    // -----------------------------------------------------------------------

    nb::class_<trade::SignalCombiner::ModelMeta>(decision, "ModelMeta")
        .def(nb::init<>())
        .def_rw("model_name", &trade::SignalCombiner::ModelMeta::model_name)
        .def_rw("rolling_mean", &trade::SignalCombiner::ModelMeta::rolling_mean)
        .def_rw("rolling_std", &trade::SignalCombiner::ModelMeta::rolling_std)
        .def_rw("icir", &trade::SignalCombiner::ModelMeta::icir)
        .def_rw("regime_fit", &trade::SignalCombiner::ModelMeta::regime_fit)
        .def_rw("data_quality", &trade::SignalCombiner::ModelMeta::data_quality)
        .def_rw("stability", &trade::SignalCombiner::ModelMeta::stability)
        .def_rw("composite_weight", &trade::SignalCombiner::ModelMeta::composite_weight);

    // -----------------------------------------------------------------------
    // SignalCombiner
    // -----------------------------------------------------------------------

    nb::class_<trade::SignalCombiner>(decision, "SignalCombiner")
        .def(nb::init<>())
        .def(nb::init<trade::SignalCombiner::Config>(), "config"_a)
        .def("calibrate", &trade::SignalCombiner::calibrate,
             "raw_scores"_a, "model_metas"_a)
        .def("combine", &trade::SignalCombiner::combine,
             "symbol"_a, "calibrated_scores"_a, "regime"_a, "model_metas"_a,
             "sentiment"_a = trade::Signal::SentimentOverlay{})
        .def("combine_batch", &trade::SignalCombiner::combine_batch,
             "symbols"_a, "raw_matrix"_a, "regime"_a, "model_metas"_a,
             "sentiments"_a = std::unordered_map<trade::Symbol, trade::Signal::SentimentOverlay>{})
        .def("update_weights", &trade::SignalCombiner::update_weights,
             "model_metas"_a, "recent_ic_matrix"_a, "regime"_a)
        .def("config", &trade::SignalCombiner::config,
             nb::rv_policy::reference_internal);

    // -----------------------------------------------------------------------
    // UniverseFilter::Config
    // -----------------------------------------------------------------------

    nb::class_<trade::UniverseFilter::Config>(decision, "UniverseFilterConfig")
        .def(nb::init<>())
        .def_rw("min_listing_days", &trade::UniverseFilter::Config::min_listing_days)
        .def_rw("min_adv_20d", &trade::UniverseFilter::Config::min_adv_20d)
        .def_rw("exclude_st", &trade::UniverseFilter::Config::exclude_st)
        .def_rw("exclude_suspended", &trade::UniverseFilter::Config::exclude_suspended)
        .def_rw("exclude_limit_locked", &trade::UniverseFilter::Config::exclude_limit_locked)
        .def_rw("exclude_delisting", &trade::UniverseFilter::Config::exclude_delisting)
        .def_rw("limit_proximity_pct", &trade::UniverseFilter::Config::limit_proximity_pct);

    // -----------------------------------------------------------------------
    // UniverseFilter::FilterStats
    // -----------------------------------------------------------------------

    nb::class_<trade::UniverseFilter::FilterStats>(decision, "FilterStats")
        .def(nb::init<>())
        .def_rw("total_input", &trade::UniverseFilter::FilterStats::total_input)
        .def_rw("total_output", &trade::UniverseFilter::FilterStats::total_output)
        .def_rw("rejected_suspended", &trade::UniverseFilter::FilterStats::rejected_suspended)
        .def_rw("rejected_st", &trade::UniverseFilter::FilterStats::rejected_st)
        .def_rw("rejected_limit_locked",
            &trade::UniverseFilter::FilterStats::rejected_limit_locked)
        .def_rw("rejected_new_stock", &trade::UniverseFilter::FilterStats::rejected_new_stock)
        .def_rw("rejected_illiquid", &trade::UniverseFilter::FilterStats::rejected_illiquid)
        .def_rw("rejected_delisting", &trade::UniverseFilter::FilterStats::rejected_delisting);

    // -----------------------------------------------------------------------
    // UniverseFilter
    // -----------------------------------------------------------------------

    nb::class_<trade::UniverseFilter>(decision, "UniverseFilter")
        .def(nb::init<>())
        .def(nb::init<trade::UniverseFilter::Config>(), "config"_a)
        .def("filter", [](const trade::UniverseFilter& f,
                          const std::unordered_map<trade::Symbol, trade::Instrument>& instruments,
                          const trade::MarketSnapshot& snapshot,
                          int date,
                          const std::unordered_map<trade::Symbol, double>& adv_20d) {
            return f.filter(instruments, snapshot,
                            trade::Date(std::chrono::days(date)), adv_20d);
        }, "all_instruments"_a, "snapshot"_a, "date"_a,
           "adv_20d"_a = std::unordered_map<trade::Symbol, double>{})
        .def_static("is_suspended", &trade::UniverseFilter::is_suspended, "inst"_a)
        .def_static("is_st", &trade::UniverseFilter::is_st, "inst"_a)
        .def("is_limit_locked", &trade::UniverseFilter::is_limit_locked,
             "bar"_a, "inst"_a, "side"_a)
        .def("is_new_stock", [](const trade::UniverseFilter& f,
                                const trade::Instrument& inst, int date) {
            return f.is_new_stock(inst, trade::Date(std::chrono::days(date)));
        }, "inst"_a, "date"_a)
        .def("is_illiquid", &trade::UniverseFilter::is_illiquid,
             "symbol"_a, "adv_20d"_a)
        .def_static("is_delisting", &trade::UniverseFilter::is_delisting, "inst"_a)
        .def("last_stats", &trade::UniverseFilter::last_stats,
             nb::rv_policy::reference_internal)
        .def("config", &trade::UniverseFilter::config,
             nb::rv_policy::reference_internal)
        .def("set_config", &trade::UniverseFilter::set_config, "config"_a);

    // -----------------------------------------------------------------------
    // PortfolioOptimizer::Constraints
    // -----------------------------------------------------------------------

    nb::class_<trade::PortfolioOptimizer::Constraints>(decision, "PortfolioConstraints")
        .def(nb::init<>())
        .def_rw("max_var_99_1d", &trade::PortfolioOptimizer::Constraints::max_var_99_1d)
        .def_rw("beta_min", &trade::PortfolioOptimizer::Constraints::beta_min)
        .def_rw("beta_max", &trade::PortfolioOptimizer::Constraints::beta_max)
        .def_rw("max_single_weight",
            &trade::PortfolioOptimizer::Constraints::max_single_weight)
        .def_rw("max_industry_weight",
            &trade::PortfolioOptimizer::Constraints::max_industry_weight)
        .def_rw("max_top3_weight", &trade::PortfolioOptimizer::Constraints::max_top3_weight)
        .def_rw("cash_floor", &trade::PortfolioOptimizer::Constraints::cash_floor)
        .def_rw("max_turnover", &trade::PortfolioOptimizer::Constraints::max_turnover)
        .def_rw("max_factor_z", &trade::PortfolioOptimizer::Constraints::max_factor_z)
        .def_rw("max_positions", &trade::PortfolioOptimizer::Constraints::max_positions)
        .def_rw("min_positions", &trade::PortfolioOptimizer::Constraints::min_positions)
        .def_rw("risk_aversion", &trade::PortfolioOptimizer::Constraints::risk_aversion);

    // -----------------------------------------------------------------------
    // PortfolioOptimizer::Candidate
    // -----------------------------------------------------------------------

    nb::class_<trade::PortfolioOptimizer::Candidate>(decision, "Candidate")
        .def(nb::init<>())
        .def_rw("symbol", &trade::PortfolioOptimizer::Candidate::symbol)
        .def_rw("alpha", &trade::PortfolioOptimizer::Candidate::alpha)
        .def_rw("confidence", &trade::PortfolioOptimizer::Candidate::confidence)
        .def_rw("estimated_cost", &trade::PortfolioOptimizer::Candidate::estimated_cost)
        .def_rw("beta", &trade::PortfolioOptimizer::Candidate::beta)
        .def_rw("adv_20d", &trade::PortfolioOptimizer::Candidate::adv_20d)
        .def_rw("industry", &trade::PortfolioOptimizer::Candidate::industry);

    // -----------------------------------------------------------------------
    // PortfolioOptimizer::TradeInstruction
    // -----------------------------------------------------------------------

    nb::class_<trade::PortfolioOptimizer::TradeInstruction>(decision, "TradeInstruction")
        .def(nb::init<>())
        .def_rw("symbol", &trade::PortfolioOptimizer::TradeInstruction::symbol)
        .def_rw("side", &trade::PortfolioOptimizer::TradeInstruction::side)
        .def_rw("target_weight", &trade::PortfolioOptimizer::TradeInstruction::target_weight)
        .def_rw("current_weight", &trade::PortfolioOptimizer::TradeInstruction::current_weight)
        .def_rw("delta_weight", &trade::PortfolioOptimizer::TradeInstruction::delta_weight)
        .def_rw("estimated_cost_bps",
            &trade::PortfolioOptimizer::TradeInstruction::estimated_cost_bps)
        .def_rw("reason", &trade::PortfolioOptimizer::TradeInstruction::reason);

    // -----------------------------------------------------------------------
    // PortfolioOptimizer::OptimizationResult::RiskMetrics
    // -----------------------------------------------------------------------

    nb::class_<trade::PortfolioOptimizer::OptimizationResult::RiskMetrics>(
            decision, "OptRiskMetrics")
        .def(nb::init<>())
        .def_rw("portfolio_var_99_1d",
            &trade::PortfolioOptimizer::OptimizationResult::RiskMetrics::portfolio_var_99_1d)
        .def_rw("portfolio_cvar_99_1d",
            &trade::PortfolioOptimizer::OptimizationResult::RiskMetrics::portfolio_cvar_99_1d)
        .def_rw("portfolio_beta",
            &trade::PortfolioOptimizer::OptimizationResult::RiskMetrics::portfolio_beta)
        .def_rw("gross_exposure",
            &trade::PortfolioOptimizer::OptimizationResult::RiskMetrics::gross_exposure)
        .def_rw("net_exposure",
            &trade::PortfolioOptimizer::OptimizationResult::RiskMetrics::net_exposure)
        .def_rw("cash_weight",
            &trade::PortfolioOptimizer::OptimizationResult::RiskMetrics::cash_weight)
        .def_rw("turnover",
            &trade::PortfolioOptimizer::OptimizationResult::RiskMetrics::turnover)
        .def_rw("max_single_weight",
            &trade::PortfolioOptimizer::OptimizationResult::RiskMetrics::max_single_weight)
        .def_rw("max_industry_weight",
            &trade::PortfolioOptimizer::OptimizationResult::RiskMetrics::max_industry_weight)
        .def_rw("top3_weight",
            &trade::PortfolioOptimizer::OptimizationResult::RiskMetrics::top3_weight)
        .def_rw("num_positions",
            &trade::PortfolioOptimizer::OptimizationResult::RiskMetrics::num_positions);

    // -----------------------------------------------------------------------
    // PortfolioOptimizer::OptimizationResult
    // -----------------------------------------------------------------------

    nb::class_<trade::PortfolioOptimizer::OptimizationResult>(decision, "OptimizationResult")
        .def(nb::init<>())
        .def_rw("symbols", &trade::PortfolioOptimizer::OptimizationResult::symbols)
        .def_rw("target_weights",
            &trade::PortfolioOptimizer::OptimizationResult::target_weights)
        .def_rw("trades", &trade::PortfolioOptimizer::OptimizationResult::trades)
        .def_rw("expected_alpha",
            &trade::PortfolioOptimizer::OptimizationResult::expected_alpha)
        .def_rw("expected_cost",
            &trade::PortfolioOptimizer::OptimizationResult::expected_cost)
        .def_rw("expected_risk",
            &trade::PortfolioOptimizer::OptimizationResult::expected_risk)
        .def_rw("risk_metrics",
            &trade::PortfolioOptimizer::OptimizationResult::risk_metrics)
        .def_rw("converged", &trade::PortfolioOptimizer::OptimizationResult::converged)
        .def_rw("iterations", &trade::PortfolioOptimizer::OptimizationResult::iterations)
        .def_rw("objective_value",
            &trade::PortfolioOptimizer::OptimizationResult::objective_value)
        .def_rw("constraint_violations",
            &trade::PortfolioOptimizer::OptimizationResult::constraint_violations);

    // -----------------------------------------------------------------------
    // PortfolioOptimizer
    // -----------------------------------------------------------------------

    nb::class_<trade::PortfolioOptimizer>(decision, "PortfolioOptimizer")
        .def(nb::init<>())
        .def(nb::init<trade::PortfolioOptimizer::Constraints>(), "constraints"_a)
        .def("optimize", &trade::PortfolioOptimizer::optimize,
             "candidates"_a, "current_weights"_a, "covariance"_a,
             "factor_loadings"_a = Eigen::MatrixXd{},
             "betas"_a = Eigen::VectorXd{})
        .def_static("select_candidates", &trade::PortfolioOptimizer::select_candidates,
             "signals"_a, "cost_estimates"_a, "betas"_a, "industries"_a,
             "adv_20d"_a, "alpha_cost_multiple"_a = 1.5, "max_k"_a = 25)
        .def_static("generate_trades", &trade::PortfolioOptimizer::generate_trades,
             "symbols"_a, "target_weights"_a, "current_weights"_a,
             "rebalance_threshold"_a = 0.01)
        .def("constraints", &trade::PortfolioOptimizer::constraints,
             nb::rv_policy::reference_internal)
        .def("set_constraints", &trade::PortfolioOptimizer::set_constraints, "c"_a);

    // -----------------------------------------------------------------------
    // OrderManager::Config
    // -----------------------------------------------------------------------

    nb::class_<trade::OrderManager::Config>(decision, "OrderManagerConfig")
        .def(nb::init<>())
        .def_rw("avoid_open_minutes", &trade::OrderManager::Config::avoid_open_minutes)
        .def_rw("main_window_start_min",
            &trade::OrderManager::Config::main_window_start_min)
        .def_rw("main_window_end_min", &trade::OrderManager::Config::main_window_end_min)
        .def_rw("mopup_end_min", &trade::OrderManager::Config::mopup_end_min)
        .def_rw("slice_interval_minutes",
            &trade::OrderManager::Config::slice_interval_minutes)
        .def_rw("target_participation",
            &trade::OrderManager::Config::target_participation)
        .def_rw("max_participation", &trade::OrderManager::Config::max_participation)
        .def_rw("large_order_adv_pct", &trade::OrderManager::Config::large_order_adv_pct)
        .def_rw("large_order_max_sessions",
            &trade::OrderManager::Config::large_order_max_sessions)
        .def_rw("slippage_spread_bps", &trade::OrderManager::Config::slippage_spread_bps)
        .def_rw("slippage_impact_a", &trade::OrderManager::Config::slippage_impact_a)
        .def_rw("slippage_impact_exp", &trade::OrderManager::Config::slippage_impact_exp)
        .def_rw("slippage_vol_b", &trade::OrderManager::Config::slippage_vol_b)
        .def_rw("urgency_multiplier", &trade::OrderManager::Config::urgency_multiplier);

    // -----------------------------------------------------------------------
    // OrderManager::ChildOrder
    // -----------------------------------------------------------------------

    nb::class_<trade::OrderManager::ChildOrder>(decision, "ChildOrder")
        .def(nb::init<>())
        .def_rw("symbol", &trade::OrderManager::ChildOrder::symbol)
        .def_rw("side", &trade::OrderManager::ChildOrder::side)
        .def_rw("quantity", &trade::OrderManager::ChildOrder::quantity)
        .def_rw("session_day", &trade::OrderManager::ChildOrder::session_day)
        .def_rw("start_minute", &trade::OrderManager::ChildOrder::start_minute)
        .def_rw("end_minute", &trade::OrderManager::ChildOrder::end_minute)
        .def_rw("participation_target",
            &trade::OrderManager::ChildOrder::participation_target)
        .def_rw("estimated_slippage_bps",
            &trade::OrderManager::ChildOrder::estimated_slippage_bps)
        .def_rw("parent_reason", &trade::OrderManager::ChildOrder::parent_reason);

    // -----------------------------------------------------------------------
    // OrderManager::ExecutionPlan::SessionSummary
    // -----------------------------------------------------------------------

    nb::class_<trade::OrderManager::ExecutionPlan::SessionSummary>(
            decision, "SessionSummary")
        .def(nb::init<>())
        .def_rw("session_day",
            &trade::OrderManager::ExecutionPlan::SessionSummary::session_day)
        .def_rw("num_child_orders",
            &trade::OrderManager::ExecutionPlan::SessionSummary::num_child_orders)
        .def_rw("notional_value",
            &trade::OrderManager::ExecutionPlan::SessionSummary::notional_value)
        .def_rw("estimated_slippage_bps",
            &trade::OrderManager::ExecutionPlan::SessionSummary::estimated_slippage_bps);

    // -----------------------------------------------------------------------
    // OrderManager::ExecutionPlan
    // -----------------------------------------------------------------------

    nb::class_<trade::OrderManager::ExecutionPlan>(decision, "ExecutionPlan")
        .def(nb::init<>())
        .def_rw("child_orders",
            &trade::OrderManager::ExecutionPlan::child_orders)
        .def_rw("total_parent_orders",
            &trade::OrderManager::ExecutionPlan::total_parent_orders)
        .def_rw("total_child_orders",
            &trade::OrderManager::ExecutionPlan::total_child_orders)
        .def_rw("sessions_needed",
            &trade::OrderManager::ExecutionPlan::sessions_needed)
        .def_rw("total_estimated_slippage_bps",
            &trade::OrderManager::ExecutionPlan::total_estimated_slippage_bps)
        .def_rw("total_estimated_cost_yuan",
            &trade::OrderManager::ExecutionPlan::total_estimated_cost_yuan)
        .def_rw("sessions", &trade::OrderManager::ExecutionPlan::sessions)
        .def("empty", &trade::OrderManager::ExecutionPlan::empty)
        .def("size", &trade::OrderManager::ExecutionPlan::size);

    // -----------------------------------------------------------------------
    // OrderManager::VolumeProfile
    // -----------------------------------------------------------------------

    nb::class_<trade::OrderManager::VolumeProfile>(decision, "VolumeProfile")
        .def(nb::init<>())
        .def_rw("bucket_fractions",
            &trade::OrderManager::VolumeProfile::bucket_fractions)
        .def_rw("estimated_daily_volume",
            &trade::OrderManager::VolumeProfile::estimated_daily_volume);

    // -----------------------------------------------------------------------
    // OrderManager
    // -----------------------------------------------------------------------

    nb::class_<trade::OrderManager>(decision, "OrderManager")
        .def(nb::init<>())
        .def(nb::init<trade::OrderManager::Config>(), "config"_a)
        .def("create_execution_plan", &trade::OrderManager::create_execution_plan,
             "orders"_a, "adv_20d"_a, "volatility"_a,
             "volume_profiles"_a = std::unordered_map<trade::Symbol,
                                       trade::OrderManager::VolumeProfile>{})
        .def("estimate_slippage_bps", &trade::OrderManager::estimate_slippage_bps,
             "participation"_a, "volatility"_a, "urgency"_a = 0.5)
        .def_static("estimate_execution_cost",
            &trade::OrderManager::estimate_execution_cost,
            "notional"_a, "slippage_bps"_a, "is_sell"_a)
        .def("is_large_order", &trade::OrderManager::is_large_order,
             "order_notional"_a, "adv_20d"_a)
        .def("split_order",
            static_cast<std::vector<trade::OrderManager::ChildOrder>(trade::OrderManager::*)(
                const trade::Order&, double, double, double,
                const trade::OrderManager::VolumeProfile&) const>(
                &trade::OrderManager::split_order),
            "order"_a, "price"_a, "adv_20d"_a, "volatility"_a, "profile"_a)
        .def("split_order_default",
            static_cast<std::vector<trade::OrderManager::ChildOrder>(trade::OrderManager::*)(
                const trade::Order&, double, double, double) const>(
                &trade::OrderManager::split_order),
            "order"_a, "price"_a, "adv_20d"_a, "volatility"_a)
        .def("sessions_for_order", &trade::OrderManager::sessions_for_order,
             "order_notional"_a, "adv_20d"_a)
        .def("config", &trade::OrderManager::config,
             nb::rv_policy::reference_internal)
        .def("set_config", &trade::OrderManager::set_config, "config"_a);

    // -----------------------------------------------------------------------
    // PreTradeChecker::Config
    // -----------------------------------------------------------------------

    nb::class_<trade::PreTradeChecker::Config>(decision, "PreTradeCheckerConfig")
        .def(nb::init<>())
        .def_rw("max_participation", &trade::PreTradeChecker::Config::max_participation)
        .def_rw("limit_proximity_warn_pct",
            &trade::PreTradeChecker::Config::limit_proximity_warn_pct)
        .def_rw("limit_proximity_reject_pct",
            &trade::PreTradeChecker::Config::limit_proximity_reject_pct)
        .def_rw("limit_proximity_size_factor",
            &trade::PreTradeChecker::Config::limit_proximity_size_factor)
        .def_rw("min_order_notional",
            &trade::PreTradeChecker::Config::min_order_notional)
        .def_rw("lot_size", &trade::PreTradeChecker::Config::lot_size);

    // -----------------------------------------------------------------------
    // PreTradeChecker::PortfolioState
    // -----------------------------------------------------------------------

    nb::class_<trade::PreTradeChecker::PortfolioState>(decision, "PortfolioState")
        .def(nb::init<>())
        .def_rw("holdings", &trade::PreTradeChecker::PortfolioState::holdings)
        .def_rw("sellable_qty", &trade::PreTradeChecker::PortfolioState::sellable_qty)
        .def_rw("cash", &trade::PreTradeChecker::PortfolioState::cash)
        .def_rw("nav", &trade::PreTradeChecker::PortfolioState::nav);

    // -----------------------------------------------------------------------
    // PreTradeChecker::MarketData
    // -----------------------------------------------------------------------

    nb::class_<trade::PreTradeChecker::MarketData>(decision, "PreTradeMarketData")
        .def(nb::init<>())
        .def_rw("bars", &trade::PreTradeChecker::MarketData::bars)
        .def_rw("instruments", &trade::PreTradeChecker::MarketData::instruments)
        .def_rw("adv_20d", &trade::PreTradeChecker::MarketData::adv_20d)
        .def_rw("limit_up", &trade::PreTradeChecker::MarketData::limit_up)
        .def_rw("limit_down", &trade::PreTradeChecker::MarketData::limit_down);

    // -----------------------------------------------------------------------
    // PreTradeChecker::PreTradeResult::CheckDetail
    // -----------------------------------------------------------------------

    nb::class_<trade::PreTradeChecker::PreTradeResult::CheckDetail>(decision, "CheckDetail")
        .def(nb::init<>())
        .def_rw("t1_sellable",
            &trade::PreTradeChecker::PreTradeResult::CheckDetail::t1_sellable)
        .def_rw("price_limit_ok",
            &trade::PreTradeChecker::PreTradeResult::CheckDetail::price_limit_ok)
        .def_rw("not_suspended",
            &trade::PreTradeChecker::PreTradeResult::CheckDetail::not_suspended)
        .def_rw("participation_ok",
            &trade::PreTradeChecker::PreTradeResult::CheckDetail::participation_ok)
        .def_rw("lot_size_ok",
            &trade::PreTradeChecker::PreTradeResult::CheckDetail::lot_size_ok)
        .def_rw("min_notional_ok",
            &trade::PreTradeChecker::PreTradeResult::CheckDetail::min_notional_ok);

    // -----------------------------------------------------------------------
    // PreTradeChecker::PreTradeResult
    // -----------------------------------------------------------------------

    nb::class_<trade::PreTradeChecker::PreTradeResult>(decision, "PreTradeResult")
        .def(nb::init<>())
        .def_rw("pass_", &trade::PreTradeChecker::PreTradeResult::pass)
        .def_rw("original_qty", &trade::PreTradeChecker::PreTradeResult::original_qty)
        .def_rw("adjusted_qty", &trade::PreTradeChecker::PreTradeResult::adjusted_qty)
        .def_rw("rejection_reason",
            &trade::PreTradeChecker::PreTradeResult::rejection_reason)
        .def_rw("detail", &trade::PreTradeChecker::PreTradeResult::detail)
        .def_rw("warnings", &trade::PreTradeChecker::PreTradeResult::warnings);

    // -----------------------------------------------------------------------
    // PreTradeChecker
    // -----------------------------------------------------------------------

    nb::class_<trade::PreTradeChecker>(decision, "PreTradeChecker")
        .def(nb::init<>())
        .def(nb::init<trade::PreTradeChecker::Config>(), "config"_a)
        .def("check", &trade::PreTradeChecker::check,
             "order"_a, "portfolio_state"_a, "market_data"_a)
        .def("check_batch", &trade::PreTradeChecker::check_batch,
             "orders"_a, "portfolio_state"_a, "market_data"_a)
        .def("check_t1_sellable", &trade::PreTradeChecker::check_t1_sellable,
             "symbol"_a, "sell_qty"_a, "state"_a)
        .def("check_price_limit", &trade::PreTradeChecker::check_price_limit,
             "symbol"_a, "side"_a, "market_data"_a)
        .def("max_participation_qty", &trade::PreTradeChecker::max_participation_qty,
             "symbol"_a, "price"_a, "market_data"_a)
        .def("round_to_lot", &trade::PreTradeChecker::round_to_lot, "qty"_a)
        .def("config", &trade::PreTradeChecker::config,
             nb::rv_policy::reference_internal)
        .def("set_config", &trade::PreTradeChecker::set_config, "config"_a);

    // -----------------------------------------------------------------------
    // DecisionReporter::PositionRisk
    // -----------------------------------------------------------------------

    nb::class_<trade::DecisionReporter::PositionRisk>(decision, "PositionRisk")
        .def(nb::init<>())
        .def_rw("symbol", &trade::DecisionReporter::PositionRisk::symbol)
        .def_rw("target_weight", &trade::DecisionReporter::PositionRisk::target_weight)
        .def_rw("current_weight", &trade::DecisionReporter::PositionRisk::current_weight)
        .def_rw("risk_contribution",
            &trade::DecisionReporter::PositionRisk::risk_contribution)
        .def_rw("marginal_var", &trade::DecisionReporter::PositionRisk::marginal_var)
        .def_rw("liquidity_days",
            &trade::DecisionReporter::PositionRisk::liquidity_days);

    // -----------------------------------------------------------------------
    // DecisionReporter::ExitPlan
    // -----------------------------------------------------------------------

    nb::class_<trade::DecisionReporter::ExitPlan>(decision, "ExitPlan")
        .def(nb::init<>())
        .def_rw("time_stop_days", &trade::DecisionReporter::ExitPlan::time_stop_days)
        .def_rw("signal_stop", &trade::DecisionReporter::ExitPlan::signal_stop)
        .def_rw("risk_stop_pct", &trade::DecisionReporter::ExitPlan::risk_stop_pct)
        .def_rw("take_profit_pct", &trade::DecisionReporter::ExitPlan::take_profit_pct);

    // -----------------------------------------------------------------------
    // DecisionReporter::RiskDashboard
    // -----------------------------------------------------------------------

    nb::class_<trade::DecisionReporter::RiskDashboard>(decision, "RiskDashboard")
        .def(nb::init<>())
        .def_rw("gross_exposure",
            &trade::DecisionReporter::RiskDashboard::gross_exposure)
        .def_rw("net_exposure",
            &trade::DecisionReporter::RiskDashboard::net_exposure)
        .def_rw("cash_weight",
            &trade::DecisionReporter::RiskDashboard::cash_weight)
        .def_rw("sector_breakdown",
            &trade::DecisionReporter::RiskDashboard::sector_breakdown)
        .def_rw("style_exposure",
            &trade::DecisionReporter::RiskDashboard::style_exposure)
        .def_rw("ex_ante_return",
            &trade::DecisionReporter::RiskDashboard::ex_ante_return)
        .def_rw("var_99_1d",
            &trade::DecisionReporter::RiskDashboard::var_99_1d)
        .def_rw("cvar_99_1d",
            &trade::DecisionReporter::RiskDashboard::cvar_99_1d)
        .def_rw("stress_loss_2015_crash",
            &trade::DecisionReporter::RiskDashboard::stress_loss_2015_crash)
        .def_rw("stress_loss_2018_trade_war",
            &trade::DecisionReporter::RiskDashboard::stress_loss_2018_trade_war)
        .def_rw("stress_loss_covid_2020",
            &trade::DecisionReporter::RiskDashboard::stress_loss_covid_2020)
        .def_rw("top_risk_contributors",
            &trade::DecisionReporter::RiskDashboard::top_risk_contributors)
        .def_rw("constraint_violations",
            &trade::DecisionReporter::RiskDashboard::constraint_violations)
        .def_rw("market_regime",
            &trade::DecisionReporter::RiskDashboard::market_regime)
        .def_rw("market_sentiment",
            &trade::DecisionReporter::RiskDashboard::market_sentiment);

    // -----------------------------------------------------------------------
    // DecisionReporter::PortfolioSnapshot
    // -----------------------------------------------------------------------

    nb::class_<trade::DecisionReporter::PortfolioSnapshot>(decision, "PortfolioSnapshot")
        .def(nb::init<>())
        .def_prop_rw("date",
            [](const trade::DecisionReporter::PortfolioSnapshot& s) {
                return s.date.time_since_epoch().count();
            },
            [](trade::DecisionReporter::PortfolioSnapshot& s, int d) {
                s.date = trade::Date(std::chrono::days(d));
            })
        .def_rw("nav", &trade::DecisionReporter::PortfolioSnapshot::nav)
        .def_rw("num_positions",
            &trade::DecisionReporter::PortfolioSnapshot::num_positions)
        .def_rw("symbols", &trade::DecisionReporter::PortfolioSnapshot::symbols)
        .def_rw("weights", &trade::DecisionReporter::PortfolioSnapshot::weights)
        .def_rw("signals", &trade::DecisionReporter::PortfolioSnapshot::signals)
        .def_rw("position_risks",
            &trade::DecisionReporter::PortfolioSnapshot::position_risks)
        .def_rw("exit_plans",
            &trade::DecisionReporter::PortfolioSnapshot::exit_plans);

    // -----------------------------------------------------------------------
    // DecisionReporter
    // -----------------------------------------------------------------------

    nb::class_<trade::DecisionReporter>(decision, "DecisionReporter")
        .def(nb::init<>())
        .def("generate_position_report",
            [](const trade::DecisionReporter& r,
               const trade::Signal& signal,
               const trade::DecisionReporter::PositionRisk& risk,
               const trade::DecisionReporter::ExitPlan& exit_plan,
               const std::string& action,
               const std::string& entry_reason,
               const std::vector<std::string>& invalidators) {
                return r.generate_position_report(
                    signal, risk, exit_plan, action, entry_reason, invalidators).dump(2);
            },
            "signal"_a, "risk"_a, "exit_plan"_a, "action"_a,
            "entry_reason"_a = "", "invalidators"_a = std::vector<std::string>{})
        .def("generate_portfolio_report",
            [](const trade::DecisionReporter& r,
               const trade::DecisionReporter::RiskDashboard& dashboard) {
                return r.generate_portfolio_report(dashboard).dump(2);
            },
            "dashboard"_a)
        .def("generate_full_report",
            [](const trade::DecisionReporter& r,
               const trade::DecisionReporter::PortfolioSnapshot& portfolio,
               const trade::DecisionReporter::RiskDashboard& dashboard,
               const std::unordered_map<trade::Symbol, std::string>& actions,
               const std::unordered_map<trade::Symbol, std::string>& entry_reasons,
               const std::unordered_map<trade::Symbol, std::vector<std::string>>& invalidators) {
                return r.generate_full_report(
                    portfolio, dashboard, actions, entry_reasons, invalidators).dump(2);
            },
            "portfolio"_a, "dashboard"_a,
            "actions"_a = std::unordered_map<trade::Symbol, std::string>{},
            "entry_reasons"_a = std::unordered_map<trade::Symbol, std::string>{},
            "invalidators"_a = std::unordered_map<trade::Symbol, std::vector<std::string>>{})
        .def_static("write_to_file",
            [](const std::string& json_str, const std::string& path) {
                auto j = nlohmann::json::parse(json_str);
                trade::DecisionReporter::write_to_file(j, path);
            },
            "json_str"_a, "path"_a)
        .def_static("regime_to_string", &trade::DecisionReporter::regime_to_string, "r"_a)
        .def_static("side_to_string", &trade::DecisionReporter::side_to_string, "s"_a);
}
