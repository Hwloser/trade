#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/eigen/dense.h>

#include "trade/backtest/backtest_engine.h"
#include "trade/backtest/performance.h"
#include "trade/backtest/broker_sim.h"
#include "trade/backtest/portfolio_state.h"
#include "trade/backtest/slippage.h"
#include "trade/backtest/validation.h"
#include "trade/backtest/reporting.h"
#include "trade/backtest/strategy.h"

namespace nb = nanobind;
using namespace nb::literals;

// ============================================================================
// bind_backtest: main binding function
// ============================================================================

void bind_backtest(nb::module_& m) {
    auto bt = m.def_submodule("backtest", "Backtest engine and validation");

    // ========================================================================
    // Enums from backtest_engine.h
    // ========================================================================

    nb::enum_<trade::OrderType>(bt, "OrderType")
        .value("kMarketOnOpen", trade::OrderType::kMarketOnOpen)
        .value("kLimitOnOpen",  trade::OrderType::kLimitOnOpen)
        .value("kVWAP",         trade::OrderType::kVWAP)
        .value("kTWAP",         trade::OrderType::kTWAP);

    nb::enum_<trade::FillStatus>(bt, "FillStatus")
        .value("kFilled",      trade::FillStatus::kFilled)
        .value("kPartialFill", trade::FillStatus::kPartialFill)
        .value("kRejected",    trade::FillStatus::kRejected)
        .value("kCancelled",   trade::FillStatus::kCancelled);

    // ========================================================================
    // Order
    // ========================================================================

    nb::class_<trade::Order>(bt, "Order")
        .def(nb::init<>())
        .def_rw("symbol",      &trade::Order::symbol)
        .def_rw("side",        &trade::Order::side)
        .def_rw("quantity",    &trade::Order::quantity)
        .def_rw("order_type",  &trade::Order::order_type)
        .def_rw("limit_price", &trade::Order::limit_price)
        .def_rw("urgency",     &trade::Order::urgency)
        .def_rw("reason",      &trade::Order::reason)
        .def("is_buy",         &trade::Order::is_buy)
        .def("is_sell",        &trade::Order::is_sell)
        .def("__repr__", [](const trade::Order& o) {
            return "<Order symbol='" + o.symbol + "' side=" +
                   (o.is_buy() ? "Buy" : "Sell") +
                   " qty=" + std::to_string(o.quantity) + ">";
        });

    // ========================================================================
    // OrderResult
    // ========================================================================

    nb::class_<trade::OrderResult>(bt, "OrderResult")
        .def(nb::init<>())
        .def_rw("order",         &trade::OrderResult::order)
        .def_rw("status",        &trade::OrderResult::status)
        .def_rw("fill_price",    &trade::OrderResult::fill_price)
        .def_rw("fill_qty",      &trade::OrderResult::fill_qty)
        .def_rw("commission",    &trade::OrderResult::commission)
        .def_rw("stamp_tax",     &trade::OrderResult::stamp_tax)
        .def_rw("transfer_fee",  &trade::OrderResult::transfer_fee)
        .def_rw("slippage_cost", &trade::OrderResult::slippage_cost)
        .def("total_cost",       &trade::OrderResult::total_cost)
        .def_rw("reject_reason", &trade::OrderResult::reject_reason)
        .def("is_filled",        &trade::OrderResult::is_filled)
        .def("__repr__", [](const trade::OrderResult& r) {
            std::string status_str;
            switch (r.status) {
                case trade::FillStatus::kFilled:      status_str = "Filled"; break;
                case trade::FillStatus::kPartialFill: status_str = "PartialFill"; break;
                case trade::FillStatus::kRejected:    status_str = "Rejected"; break;
                case trade::FillStatus::kCancelled:   status_str = "Cancelled"; break;
            }
            return "<OrderResult symbol='" + r.order.symbol +
                   "' status=" + status_str +
                   " fill_price=" + std::to_string(r.fill_price) +
                   " fill_qty=" + std::to_string(r.fill_qty) + ">";
        });

    // ========================================================================
    // PositionRecord
    // ========================================================================

    nb::class_<trade::PositionRecord>(bt, "PositionRecord")
        .def(nb::init<>())
        .def_rw("symbol",          &trade::PositionRecord::symbol)
        .def_rw("quantity",        &trade::PositionRecord::quantity)
        .def_rw("market_value",    &trade::PositionRecord::market_value)
        .def_rw("weight",          &trade::PositionRecord::weight)
        .def_rw("unrealised_pnl",  &trade::PositionRecord::unrealised_pnl)
        .def_rw("cost_basis",      &trade::PositionRecord::cost_basis)
        .def("__repr__", [](const trade::PositionRecord& p) {
            return "<PositionRecord symbol='" + p.symbol +
                   "' qty=" + std::to_string(p.quantity) +
                   " weight=" + std::to_string(p.weight) + ">";
        });

    // ========================================================================
    // DailyRecord
    // ========================================================================

    nb::class_<trade::DailyRecord>(bt, "DailyRecord")
        .def(nb::init<>())
        .def_prop_rw("date",
            [](const trade::DailyRecord& r) {
                return static_cast<int>(r.date.time_since_epoch().count());
            },
            [](trade::DailyRecord& r, int d) {
                r.date = trade::Date(std::chrono::days(d));
            })
        .def_rw("nav",               &trade::DailyRecord::nav)
        .def_rw("daily_return",      &trade::DailyRecord::daily_return)
        .def_rw("cumulative_return", &trade::DailyRecord::cumulative_return)
        .def_rw("drawdown",          &trade::DailyRecord::drawdown)
        .def_rw("cash",              &trade::DailyRecord::cash)
        .def_rw("gross_exposure",    &trade::DailyRecord::gross_exposure)
        .def_rw("num_positions",     &trade::DailyRecord::num_positions)
        .def_rw("turnover",          &trade::DailyRecord::turnover)
        .def_rw("total_cost",        &trade::DailyRecord::total_cost)
        .def_rw("positions",         &trade::DailyRecord::positions)
        .def_rw("fills",             &trade::DailyRecord::fills)
        .def("__repr__", [](const trade::DailyRecord& r) {
            return "<DailyRecord date=" +
                   std::to_string(r.date.time_since_epoch().count()) +
                   " nav=" + std::to_string(r.nav) +
                   " ret=" + std::to_string(r.daily_return) + ">";
        });

    // ========================================================================
    // BacktestResult
    // ========================================================================

    nb::class_<trade::BacktestResult>(bt, "BacktestResult")
        .def(nb::init<>())
        .def_rw("strategy_name",  &trade::BacktestResult::strategy_name)
        .def_prop_rw("start_date",
            [](const trade::BacktestResult& r) {
                return static_cast<int>(r.start_date.time_since_epoch().count());
            },
            [](trade::BacktestResult& r, int d) {
                r.start_date = trade::Date(std::chrono::days(d));
            })
        .def_prop_rw("end_date",
            [](const trade::BacktestResult& r) {
                return static_cast<int>(r.end_date.time_since_epoch().count());
            },
            [](trade::BacktestResult& r, int d) {
                r.end_date = trade::Date(std::chrono::days(d));
            })
        .def_rw("initial_capital", &trade::BacktestResult::initial_capital)
        .def_rw("final_nav",       &trade::BacktestResult::final_nav)
        .def_rw("trading_days",    &trade::BacktestResult::trading_days)
        .def_rw("daily_records",   &trade::BacktestResult::daily_records)
        .def("total_return",       &trade::BacktestResult::total_return)
        .def("nav_series",         &trade::BacktestResult::nav_series)
        .def("return_series",      &trade::BacktestResult::return_series)
        .def("drawdown_series",    &trade::BacktestResult::drawdown_series)
        .def("__repr__", [](const trade::BacktestResult& r) {
            return "<BacktestResult strategy='" + r.strategy_name +
                   "' days=" + std::to_string(r.trading_days) +
                   " final_nav=" + std::to_string(r.final_nav) +
                   " total_return=" + std::to_string(r.total_return()) + ">";
        });

    // ========================================================================
    // BacktestEngine::Config
    // ========================================================================

    nb::class_<trade::BacktestEngine::Config>(bt, "BacktestEngineConfig")
        .def(nb::init<>())
        .def_rw("initial_capital",       &trade::BacktestEngine::Config::initial_capital)
        .def_rw("max_positions",         &trade::BacktestEngine::Config::max_positions)
        .def_rw("min_positions",         &trade::BacktestEngine::Config::min_positions)
        .def_rw("max_adv_participation", &trade::BacktestEngine::Config::max_adv_participation)
        .def_rw("rebalance_threshold",   &trade::BacktestEngine::Config::rebalance_threshold)
        .def_rw("alpha_cost_multiple",   &trade::BacktestEngine::Config::alpha_cost_multiple)
        .def_rw("verbose",              &trade::BacktestEngine::Config::verbose);

    // ========================================================================
    // BacktestEngine
    // ========================================================================

    nb::class_<trade::BacktestEngine>(bt, "BacktestEngine")
        .def(nb::init<
                std::shared_ptr<trade::IMarketDataFeed>,
                std::shared_ptr<trade::IExecutionVenue>,
                std::shared_ptr<trade::IClock>>(),
             "market_data"_a, "execution"_a, "clock"_a)
        .def(nb::init<
                std::shared_ptr<trade::IMarketDataFeed>,
                std::shared_ptr<trade::IExecutionVenue>,
                std::shared_ptr<trade::IClock>,
                trade::BacktestEngine::Config>(),
             "market_data"_a, "execution"_a, "clock"_a, "config"_a)
        .def("run", [](trade::BacktestEngine& engine,
                       trade::IStrategy& strategy,
                       int start_date, int end_date) {
            return engine.run(
                strategy,
                trade::Date(std::chrono::days(start_date)),
                trade::Date(std::chrono::days(end_date)));
        }, "strategy"_a, "start_date"_a, "end_date"_a,
           "Run backtest over [start_date, end_date]. Dates are ints (days since epoch).")
        .def("config", &trade::BacktestEngine::config,
             nb::rv_policy::reference_internal)
        .def("portfolio", &trade::BacktestEngine::portfolio,
             nb::rv_policy::reference_internal)
        .def("set_progress_callback", &trade::BacktestEngine::set_progress_callback,
             "callback"_a);

    // ========================================================================
    // IMarketDataFeed (abstract interface)
    // ========================================================================

    nb::class_<trade::IMarketDataFeed>(bt, "IMarketDataFeed");

    // ========================================================================
    // IExecutionVenue (abstract interface)
    // ========================================================================

    nb::class_<trade::IExecutionVenue>(bt, "IExecutionVenue");

    // ========================================================================
    // IClock (abstract interface)
    // ========================================================================

    nb::class_<trade::IClock>(bt, "IClock");

    // ========================================================================
    // PerformanceReport
    // ========================================================================

    nb::class_<trade::PerformanceReport>(bt, "PerformanceReport")
        .def(nb::init<>())
        // Return metrics
        .def_rw("annualised_return",  &trade::PerformanceReport::annualised_return)
        .def_rw("cumulative_return",  &trade::PerformanceReport::cumulative_return)
        .def_rw("cagr",              &trade::PerformanceReport::cagr)
        .def_rw("monthly_returns",    &trade::PerformanceReport::monthly_returns)
        // Risk-adjusted metrics
        .def_rw("sharpe_ratio",       &trade::PerformanceReport::sharpe_ratio)
        .def_rw("sortino_ratio",      &trade::PerformanceReport::sortino_ratio)
        .def_rw("calmar_ratio",       &trade::PerformanceReport::calmar_ratio)
        .def_rw("information_ratio",  &trade::PerformanceReport::information_ratio)
        // Drawdown metrics
        .def_rw("max_drawdown",           &trade::PerformanceReport::max_drawdown)
        .def_rw("max_drawdown_duration",  &trade::PerformanceReport::max_drawdown_duration)
        .def_rw("avg_drawdown",           &trade::PerformanceReport::avg_drawdown)
        .def_prop_rw("max_drawdown_start",
            [](const trade::PerformanceReport& r) {
                return static_cast<int>(r.max_drawdown_start.time_since_epoch().count());
            },
            [](trade::PerformanceReport& r, int d) {
                r.max_drawdown_start = trade::Date(std::chrono::days(d));
            })
        .def_prop_rw("max_drawdown_end",
            [](const trade::PerformanceReport& r) {
                return static_cast<int>(r.max_drawdown_end.time_since_epoch().count());
            },
            [](trade::PerformanceReport& r, int d) {
                r.max_drawdown_end = trade::Date(std::chrono::days(d));
            })
        .def_prop_rw("max_drawdown_recovery",
            [](const trade::PerformanceReport& r) {
                return static_cast<int>(r.max_drawdown_recovery.time_since_epoch().count());
            },
            [](trade::PerformanceReport& r, int d) {
                r.max_drawdown_recovery = trade::Date(std::chrono::days(d));
            })
        // Trading metrics
        .def_rw("win_rate",            &trade::PerformanceReport::win_rate)
        .def_rw("profit_loss_ratio",   &trade::PerformanceReport::profit_loss_ratio)
        .def_rw("profit_factor",       &trade::PerformanceReport::profit_factor)
        .def_rw("avg_holding_days",    &trade::PerformanceReport::avg_holding_days)
        .def_rw("total_trades",        &trade::PerformanceReport::total_trades)
        .def_rw("winning_trades",      &trade::PerformanceReport::winning_trades)
        .def_rw("losing_trades",       &trade::PerformanceReport::losing_trades)
        .def_rw("avg_trade_pnl",       &trade::PerformanceReport::avg_trade_pnl)
        .def_rw("largest_win",         &trade::PerformanceReport::largest_win)
        .def_rw("largest_loss",        &trade::PerformanceReport::largest_loss)
        // Turnover and cost metrics
        .def_rw("avg_daily_turnover",  &trade::PerformanceReport::avg_daily_turnover)
        .def_rw("total_turnover",      &trade::PerformanceReport::total_turnover)
        .def_rw("total_trades_count",  &trade::PerformanceReport::total_trades_count)
        .def_rw("avg_cost_per_trade",  &trade::PerformanceReport::avg_cost_per_trade)
        .def_rw("total_costs",         &trade::PerformanceReport::total_costs)
        .def_rw("cost_drag",           &trade::PerformanceReport::cost_drag)
        // Benchmark-relative metrics
        .def_rw("alpha",              &trade::PerformanceReport::alpha)
        .def_rw("beta",               &trade::PerformanceReport::beta)
        .def_rw("tracking_error",     &trade::PerformanceReport::tracking_error)
        .def_rw("active_return",      &trade::PerformanceReport::active_return)
        .def_rw("benchmark_return",   &trade::PerformanceReport::benchmark_return)
        .def_rw("benchmark_sharpe",   &trade::PerformanceReport::benchmark_sharpe)
        .def_rw("correlation",        &trade::PerformanceReport::correlation)
        .def_rw("r_squared",          &trade::PerformanceReport::r_squared)
        // Statistical confidence metrics
        .def_rw("sharpe_bootstrap_ci_lower", &trade::PerformanceReport::sharpe_bootstrap_ci_lower)
        .def_rw("sharpe_bootstrap_ci_upper", &trade::PerformanceReport::sharpe_bootstrap_ci_upper)
        .def_rw("deflated_sharpe_ratio",     &trade::PerformanceReport::deflated_sharpe_ratio)
        .def_rw("sharpe_t_statistic",        &trade::PerformanceReport::sharpe_t_statistic)
        .def_rw("sharpe_p_value",            &trade::PerformanceReport::sharpe_p_value)
        // VaR / Tail risk
        .def_rw("var_95",             &trade::PerformanceReport::var_95)
        .def_rw("var_99",             &trade::PerformanceReport::var_99)
        .def_rw("cvar_95",            &trade::PerformanceReport::cvar_95)
        .def_rw("skewness",           &trade::PerformanceReport::skewness)
        .def_rw("kurtosis",           &trade::PerformanceReport::kurtosis)
        // Exposure metrics
        .def_rw("avg_num_positions",   &trade::PerformanceReport::avg_num_positions)
        .def_rw("avg_gross_exposure",  &trade::PerformanceReport::avg_gross_exposure)
        .def_rw("avg_cash_weight",     &trade::PerformanceReport::avg_cash_weight)
        .def("__repr__", [](const trade::PerformanceReport& r) {
            return "<PerformanceReport sharpe=" + std::to_string(r.sharpe_ratio) +
                   " ann_ret=" + std::to_string(r.annualised_return) +
                   " max_dd=" + std::to_string(r.max_drawdown) + ">";
        });

    // ========================================================================
    // PerformanceCalculator::Config
    // ========================================================================

    nb::class_<trade::PerformanceCalculator::Config>(bt, "PerformanceCalculatorConfig")
        .def(nb::init<>())
        .def_rw("risk_free_rate",       &trade::PerformanceCalculator::Config::risk_free_rate)
        .def_rw("annualisation_factor", &trade::PerformanceCalculator::Config::annualisation_factor)
        .def_rw("bootstrap_samples",    &trade::PerformanceCalculator::Config::bootstrap_samples)
        .def_rw("bootstrap_block_size", &trade::PerformanceCalculator::Config::bootstrap_block_size)
        .def_rw("benchmark_name",       &trade::PerformanceCalculator::Config::benchmark_name);

    // ========================================================================
    // PerformanceCalculator::DrawdownInfo
    // ========================================================================

    nb::class_<trade::PerformanceCalculator::DrawdownInfo>(bt, "DrawdownInfo")
        .def(nb::init<>())
        .def_rw("max_drawdown",          &trade::PerformanceCalculator::DrawdownInfo::max_drawdown)
        .def_rw("max_drawdown_duration", &trade::PerformanceCalculator::DrawdownInfo::max_drawdown_duration)
        .def_rw("avg_drawdown",          &trade::PerformanceCalculator::DrawdownInfo::avg_drawdown)
        .def_rw("peak_index",            &trade::PerformanceCalculator::DrawdownInfo::peak_index)
        .def_rw("trough_index",          &trade::PerformanceCalculator::DrawdownInfo::trough_index)
        .def_rw("recovery_index",        &trade::PerformanceCalculator::DrawdownInfo::recovery_index);

    // ========================================================================
    // PerformanceCalculator::TradeStats
    // ========================================================================

    nb::class_<trade::PerformanceCalculator::TradeStats>(bt, "TradeStats")
        .def(nb::init<>())
        .def_rw("win_rate",          &trade::PerformanceCalculator::TradeStats::win_rate)
        .def_rw("profit_loss_ratio", &trade::PerformanceCalculator::TradeStats::profit_loss_ratio)
        .def_rw("profit_factor",     &trade::PerformanceCalculator::TradeStats::profit_factor)
        .def_rw("avg_holding_days",  &trade::PerformanceCalculator::TradeStats::avg_holding_days)
        .def_rw("total_trades",      &trade::PerformanceCalculator::TradeStats::total_trades)
        .def_rw("winning_trades",    &trade::PerformanceCalculator::TradeStats::winning_trades)
        .def_rw("losing_trades",     &trade::PerformanceCalculator::TradeStats::losing_trades)
        .def_rw("avg_trade_pnl",     &trade::PerformanceCalculator::TradeStats::avg_trade_pnl)
        .def_rw("largest_win",       &trade::PerformanceCalculator::TradeStats::largest_win)
        .def_rw("largest_loss",      &trade::PerformanceCalculator::TradeStats::largest_loss);

    // ========================================================================
    // PerformanceCalculator::BenchmarkStats
    // ========================================================================

    nb::class_<trade::PerformanceCalculator::BenchmarkStats>(bt, "BenchmarkStats")
        .def(nb::init<>())
        .def_rw("alpha",             &trade::PerformanceCalculator::BenchmarkStats::alpha)
        .def_rw("beta",              &trade::PerformanceCalculator::BenchmarkStats::beta)
        .def_rw("tracking_error",    &trade::PerformanceCalculator::BenchmarkStats::tracking_error)
        .def_rw("correlation",       &trade::PerformanceCalculator::BenchmarkStats::correlation)
        .def_rw("r_squared",         &trade::PerformanceCalculator::BenchmarkStats::r_squared)
        .def_rw("information_ratio", &trade::PerformanceCalculator::BenchmarkStats::information_ratio);

    // ========================================================================
    // PerformanceCalculator::ConfidenceStats
    // ========================================================================

    nb::class_<trade::PerformanceCalculator::ConfidenceStats>(bt, "ConfidenceStats")
        .def(nb::init<>())
        .def_rw("ci_lower",         &trade::PerformanceCalculator::ConfidenceStats::ci_lower)
        .def_rw("ci_upper",         &trade::PerformanceCalculator::ConfidenceStats::ci_upper)
        .def_rw("t_statistic",      &trade::PerformanceCalculator::ConfidenceStats::t_statistic)
        .def_rw("p_value",          &trade::PerformanceCalculator::ConfidenceStats::p_value)
        .def_rw("deflated_sharpe",  &trade::PerformanceCalculator::ConfidenceStats::deflated_sharpe);

    // ========================================================================
    // PerformanceCalculator
    // ========================================================================

    nb::class_<trade::PerformanceCalculator>(bt, "PerformanceCalculator")
        .def(nb::init<>())
        .def(nb::init<trade::PerformanceCalculator::Config>(), "config"_a)
        // Core computation
        .def("compute",
            nb::overload_cast<const trade::BacktestResult&>(
                &trade::PerformanceCalculator::compute, nb::const_),
            "result"_a)
        .def("compute_with_benchmark",
            nb::overload_cast<const trade::BacktestResult&, const std::vector<double>&>(
                &trade::PerformanceCalculator::compute, nb::const_),
            "result"_a, "benchmark_returns"_a)
        // Individual metric groups
        .def("annualised_return", &trade::PerformanceCalculator::annualised_return,
             "daily_returns"_a)
        .def("cumulative_return", &trade::PerformanceCalculator::cumulative_return,
             "daily_returns"_a)
        .def("sharpe_ratio", &trade::PerformanceCalculator::sharpe_ratio,
             "daily_returns"_a)
        .def("sortino_ratio", &trade::PerformanceCalculator::sortino_ratio,
             "daily_returns"_a)
        .def("calmar_ratio", &trade::PerformanceCalculator::calmar_ratio,
             "daily_returns"_a, "max_dd"_a)
        .def("information_ratio", &trade::PerformanceCalculator::information_ratio,
             "active_returns"_a)
        // Drawdown analysis
        .def("analyse_drawdowns", &trade::PerformanceCalculator::analyse_drawdowns,
             "nav_series"_a)
        // Trading statistics
        .def("compute_trade_stats", &trade::PerformanceCalculator::compute_trade_stats,
             "records"_a)
        // Benchmark regression
        .def("compute_benchmark_stats", &trade::PerformanceCalculator::compute_benchmark_stats,
             "strategy_returns"_a, "benchmark_returns"_a)
        // Statistical confidence
        .def("compute_sharpe_confidence", &trade::PerformanceCalculator::compute_sharpe_confidence,
             "daily_returns"_a, "num_trials"_a = 1)
        .def("bootstrap_sharpe_ci", &trade::PerformanceCalculator::bootstrap_sharpe_ci,
             "daily_returns"_a, "confidence"_a = 0.95)
        .def("deflated_sharpe_ratio", &trade::PerformanceCalculator::deflated_sharpe_ratio,
             "observed_sharpe"_a, "num_trials"_a, "num_observations"_a,
             "skewness"_a = 0.0, "kurtosis"_a = 3.0)
        // VaR and tail risk
        .def("historical_var", &trade::PerformanceCalculator::historical_var,
             "returns"_a, "alpha"_a)
        .def("conditional_var", &trade::PerformanceCalculator::conditional_var,
             "returns"_a, "alpha"_a)
        // Distribution moments
        .def("compute_skewness", &trade::PerformanceCalculator::compute_skewness,
             "returns"_a)
        .def("compute_kurtosis", &trade::PerformanceCalculator::compute_kurtosis,
             "returns"_a)
        .def("config", &trade::PerformanceCalculator::config,
             nb::rv_policy::reference_internal);

    // ========================================================================
    // Slippage enums
    // ========================================================================

    nb::enum_<trade::MarketCapBucket>(bt, "MarketCapBucket")
        .value("kLarge", trade::MarketCapBucket::kLarge)
        .value("kMid",   trade::MarketCapBucket::kMid)
        .value("kSmall", trade::MarketCapBucket::kSmall);

    nb::enum_<trade::ResearchPhase>(bt, "ResearchPhase")
        .value("kResearch",      trade::ResearchPhase::kResearch)
        .value("kPreProduction", trade::ResearchPhase::kPreProduction)
        .value("kProduction",    trade::ResearchPhase::kProduction);

    // ========================================================================
    // SlippageModel (abstract base)
    // ========================================================================

    nb::class_<trade::SlippageModel>(bt, "SlippageModel")
        .def("compute", &trade::SlippageModel::compute, "order"_a, "bar"_a)
        .def("apply",   &trade::SlippageModel::apply,
             "base_price"_a, "side"_a, "order"_a, "bar"_a)
        .def("name",    &trade::SlippageModel::name);

    // ========================================================================
    // FixedSlippage
    // ========================================================================

    nb::class_<trade::FixedSlippage::Config>(bt, "FixedSlippageConfig")
        .def(nb::init<>())
        .def_rw("large_cap_bps",              &trade::FixedSlippage::Config::large_cap_bps)
        .def_rw("mid_cap_bps",                &trade::FixedSlippage::Config::mid_cap_bps)
        .def_rw("small_cap_bps",              &trade::FixedSlippage::Config::small_cap_bps)
        .def_rw("large_cap_amount_threshold", &trade::FixedSlippage::Config::large_cap_amount_threshold)
        .def_rw("mid_cap_amount_threshold",   &trade::FixedSlippage::Config::mid_cap_amount_threshold);

    nb::class_<trade::FixedSlippage, trade::SlippageModel>(bt, "FixedSlippage")
        .def(nb::init<>())
        .def(nb::init<trade::FixedSlippage::Config>(), "config"_a)
        .def("classify", &trade::FixedSlippage::classify, "bar"_a)
        .def("config", &trade::FixedSlippage::config,
             nb::rv_policy::reference_internal);

    // ========================================================================
    // ParticipationSlippage
    // ========================================================================

    nb::class_<trade::ParticipationSlippage::Config>(bt, "ParticipationSlippageConfig")
        .def(nb::init<>())
        .def_rw("impact_coefficient", &trade::ParticipationSlippage::Config::impact_coefficient)
        .def_rw("min_slippage_bps",   &trade::ParticipationSlippage::Config::min_slippage_bps)
        .def_rw("max_slippage_bps",   &trade::ParticipationSlippage::Config::max_slippage_bps);

    nb::class_<trade::ParticipationSlippage, trade::SlippageModel>(bt, "ParticipationSlippage")
        .def(nb::init<>())
        .def(nb::init<trade::ParticipationSlippage::Config>(), "config"_a)
        .def("config", &trade::ParticipationSlippage::config,
             nb::rv_policy::reference_internal);

    // ========================================================================
    // AlmgrenChrissSlippage
    // ========================================================================

    nb::class_<trade::AlmgrenChrissSlippage::Config>(bt, "AlmgrenChrissSlippageConfig")
        .def(nb::init<>())
        .def_rw("eta",                &trade::AlmgrenChrissSlippage::Config::eta)
        .def_rw("gamma",              &trade::AlmgrenChrissSlippage::Config::gamma)
        .def_rw("temp_exponent",      &trade::AlmgrenChrissSlippage::Config::temp_exponent)
        .def_rw("daily_vol",          &trade::AlmgrenChrissSlippage::Config::daily_vol)
        .def_rw("default_adv",        &trade::AlmgrenChrissSlippage::Config::default_adv)
        .def_rw("execution_horizon",  &trade::AlmgrenChrissSlippage::Config::execution_horizon)
        .def_rw("min_slippage_bps",   &trade::AlmgrenChrissSlippage::Config::min_slippage_bps)
        .def_rw("max_slippage_bps",   &trade::AlmgrenChrissSlippage::Config::max_slippage_bps);

    nb::class_<trade::AlmgrenChrissSlippage, trade::SlippageModel>(bt, "AlmgrenChrissSlippage")
        .def(nb::init<>())
        .def(nb::init<trade::AlmgrenChrissSlippage::Config>(), "config"_a)
        .def("set_adv",        &trade::AlmgrenChrissSlippage::set_adv,
             "symbol"_a, "adv_shares"_a)
        .def("set_volatility", &trade::AlmgrenChrissSlippage::set_volatility,
             "symbol"_a, "daily_vol"_a)
        .def("config", &trade::AlmgrenChrissSlippage::config,
             nb::rv_policy::reference_internal);

    // ========================================================================
    // SlippageFactory
    // ========================================================================

    nb::class_<trade::SlippageFactory>(bt, "SlippageFactory")
        .def_static("create_by_phase",
            nb::overload_cast<trade::ResearchPhase>(&trade::SlippageFactory::create),
            "phase"_a)
        .def_static("create_by_name",
            nb::overload_cast<const std::string&>(&trade::SlippageFactory::create),
            "name"_a);

    // ========================================================================
    // PortfolioState types
    // ========================================================================

    // TaxLot
    nb::class_<trade::TaxLot>(bt, "TaxLot")
        .def(nb::init<>())
        .def_prop_rw("buy_date",
            [](const trade::TaxLot& t) {
                return static_cast<int>(t.buy_date.time_since_epoch().count());
            },
            [](trade::TaxLot& t, int d) {
                t.buy_date = trade::Date(std::chrono::days(d));
            })
        .def_rw("quantity",   &trade::TaxLot::quantity)
        .def_rw("cost_price", &trade::TaxLot::cost_price)
        .def_prop_rw("sellable_date",
            [](const trade::TaxLot& t) {
                return static_cast<int>(t.sellable_date.time_since_epoch().count());
            },
            [](trade::TaxLot& t, int d) {
                t.sellable_date = trade::Date(std::chrono::days(d));
            })
        .def("is_sellable", [](const trade::TaxLot& t, int today) {
            return t.is_sellable(trade::Date(std::chrono::days(today)));
        }, "today"_a)
        .def("total_cost", &trade::TaxLot::total_cost);

    // Position
    nb::class_<trade::Position>(bt, "Position")
        .def(nb::init<>())
        .def_rw("symbol",           &trade::Position::symbol)
        .def_rw("total_qty",        &trade::Position::total_qty)
        .def_rw("avg_cost",         &trade::Position::avg_cost)
        .def_rw("market_price",     &trade::Position::market_price)
        .def_rw("market_value",     &trade::Position::market_value)
        .def_rw("unrealised_pnl",   &trade::Position::unrealised_pnl)
        .def_rw("realised_pnl",     &trade::Position::realised_pnl)
        .def_rw("total_cost_basis", &trade::Position::total_cost_basis)
        .def_rw("weight",           &trade::Position::weight)
        .def("is_empty",            &trade::Position::is_empty)
        .def("__repr__", [](const trade::Position& p) {
            return "<Position symbol='" + p.symbol +
                   "' qty=" + std::to_string(p.total_qty) +
                   " mv=" + std::to_string(p.market_value) +
                   " weight=" + std::to_string(p.weight) + ">";
        });

    // PortfolioState
    nb::class_<trade::PortfolioState>(bt, "PortfolioState")
        .def(nb::init<double>(), "initial_cash"_a)
        // T+1 sellability
        .def("sellable_qty", [](const trade::PortfolioState& ps,
                                const std::string& symbol, int today) {
            return ps.sellable_qty(symbol, trade::Date(std::chrono::days(today)));
        }, "symbol"_a, "today"_a)
        .def("unsellable_qty", [](const trade::PortfolioState& ps,
                                  const std::string& symbol, int today) {
            return ps.unsellable_qty(symbol, trade::Date(std::chrono::days(today)));
        }, "symbol"_a, "today"_a)
        .def("total_qty", &trade::PortfolioState::total_qty, "symbol"_a)
        // NAV and metrics
        .def("total_nav",       &trade::PortfolioState::total_nav)
        .def("cash",            &trade::PortfolioState::cash)
        .def("gross_exposure",  &trade::PortfolioState::gross_exposure)
        .def("net_exposure",    &trade::PortfolioState::net_exposure)
        .def("cash_weight",     &trade::PortfolioState::cash_weight)
        .def("num_positions",   &trade::PortfolioState::num_positions)
        .def("position_weights", &trade::PortfolioState::position_weights)
        // Position access
        .def("position", &trade::PortfolioState::position, "symbol"_a,
             nb::rv_policy::reference_internal)
        .def("has_position",   &trade::PortfolioState::has_position, "symbol"_a)
        .def("positions",      &trade::PortfolioState::positions,
             nb::rv_policy::reference_internal)
        // Snapshot and P&L
        .def("snapshot",             &trade::PortfolioState::snapshot)
        .def("total_realised_pnl",   &trade::PortfolioState::total_realised_pnl)
        .def("total_unrealised_pnl", &trade::PortfolioState::total_unrealised_pnl)
        // Maintenance
        .def("cleanup_empty_positions", &trade::PortfolioState::cleanup_empty_positions)
        .def("adjust_cash",  &trade::PortfolioState::adjust_cash, "amount"_a)
        .def("set_cash",     &trade::PortfolioState::set_cash, "cash"_a)
        .def("reset",        &trade::PortfolioState::reset, "initial_cash"_a)
        .def("compute_weights", &trade::PortfolioState::compute_weights)
        .def("__repr__", [](const trade::PortfolioState& ps) {
            return "<PortfolioState nav=" + std::to_string(ps.total_nav()) +
                   " cash=" + std::to_string(ps.cash()) +
                   " positions=" + std::to_string(ps.num_positions()) + ">";
        });

    // ========================================================================
    // BrokerSim
    // ========================================================================

    nb::class_<trade::BrokerSim::Config>(bt, "BrokerSimConfig")
        .def(nb::init<>())
        .def_rw("stamp_tax_rate",           &trade::BrokerSim::Config::stamp_tax_rate)
        .def_rw("commission_rate",          &trade::BrokerSim::Config::commission_rate)
        .def_rw("commission_min_yuan",      &trade::BrokerSim::Config::commission_min_yuan)
        .def_rw("transfer_fee_rate",        &trade::BrokerSim::Config::transfer_fee_rate)
        .def_rw("reject_limit_up_buy",      &trade::BrokerSim::Config::reject_limit_up_buy)
        .def_rw("reject_limit_down_sell",   &trade::BrokerSim::Config::reject_limit_down_sell)
        .def_rw("reject_st_buy",           &trade::BrokerSim::Config::reject_st_buy)
        .def_rw("reject_delisting_buy",    &trade::BrokerSim::Config::reject_delisting_buy)
        .def_rw("enforce_lot_size",        &trade::BrokerSim::Config::enforce_lot_size)
        .def_rw("lot_size",               &trade::BrokerSim::Config::lot_size)
        .def_rw("max_participation_rate",  &trade::BrokerSim::Config::max_participation_rate);

    nb::class_<trade::BrokerSim, trade::IExecutionVenue>(bt, "BrokerSim")
        .def(nb::init<
                std::shared_ptr<trade::PortfolioState>,
                std::shared_ptr<trade::SlippageModel>>(),
             "portfolio"_a, "slippage"_a)
        .def(nb::init<
                std::shared_ptr<trade::PortfolioState>,
                std::shared_ptr<trade::SlippageModel>,
                trade::BrokerSim::Config>(),
             "portfolio"_a, "slippage"_a, "config"_a)
        .def(nb::init<
                std::shared_ptr<trade::PortfolioState>,
                std::shared_ptr<trade::SlippageModel>,
                const trade::TradingCostConfig&>(),
             "portfolio"_a, "slippage"_a, "cost_config"_a)
        // IExecutionVenue interface
        .def("execute", &trade::BrokerSim::execute, "order"_a, "bar"_a)
        .def("execute_batch", &trade::BrokerSim::execute_batch,
             "orders"_a, "snapshot"_a)
        // Instrument metadata
        .def("set_instrument", &trade::BrokerSim::set_instrument,
             "symbol"_a, "instrument"_a)
        .def("set_instruments", &trade::BrokerSim::set_instruments,
             "instruments"_a)
        .def("set_suspended", [](trade::BrokerSim& b,
                                 const std::string& symbol, int date) {
            b.set_suspended(symbol, trade::Date(std::chrono::days(date)));
        }, "symbol"_a, "date"_a)
        .def("set_delist_date", [](trade::BrokerSim& b,
                                   const std::string& symbol, int date) {
            b.set_delist_date(symbol, trade::Date(std::chrono::days(date)));
        }, "symbol"_a, "date"_a)
        // Force-close
        .def("force_close_delisting", [](trade::BrokerSim& b,
                                         int date,
                                         const trade::MarketSnapshot& snapshot) {
            return b.force_close_delisting(trade::Date(std::chrono::days(date)), snapshot);
        }, "date"_a, "snapshot"_a)
        // Accessors
        .def("config", &trade::BrokerSim::config,
             nb::rv_policy::reference_internal)
        .def("portfolio", &trade::BrokerSim::portfolio,
             nb::rv_policy::reference_internal)
        .def("total_commission",       &trade::BrokerSim::total_commission)
        .def("total_stamp_tax",        &trade::BrokerSim::total_stamp_tax)
        .def("total_transfer_fee",     &trade::BrokerSim::total_transfer_fee)
        .def("total_slippage_cost",    &trade::BrokerSim::total_slippage_cost)
        .def("total_orders_submitted", &trade::BrokerSim::total_orders_submitted)
        .def("total_orders_filled",    &trade::BrokerSim::total_orders_filled)
        .def("total_orders_rejected",  &trade::BrokerSim::total_orders_rejected);

    // ========================================================================
    // Validation types
    // ========================================================================

    // FoldResult
    nb::class_<trade::FoldResult>(bt, "FoldResult")
        .def(nb::init<>())
        .def_rw("fold_index",    &trade::FoldResult::fold_index)
        .def_prop_rw("train_start",
            [](const trade::FoldResult& f) {
                return static_cast<int>(f.train_start.time_since_epoch().count());
            },
            [](trade::FoldResult& f, int d) {
                f.train_start = trade::Date(std::chrono::days(d));
            })
        .def_prop_rw("train_end",
            [](const trade::FoldResult& f) {
                return static_cast<int>(f.train_end.time_since_epoch().count());
            },
            [](trade::FoldResult& f, int d) {
                f.train_end = trade::Date(std::chrono::days(d));
            })
        .def_prop_rw("test_start",
            [](const trade::FoldResult& f) {
                return static_cast<int>(f.test_start.time_since_epoch().count());
            },
            [](trade::FoldResult& f, int d) {
                f.test_start = trade::Date(std::chrono::days(d));
            })
        .def_prop_rw("test_end",
            [](const trade::FoldResult& f) {
                return static_cast<int>(f.test_end.time_since_epoch().count());
            },
            [](trade::FoldResult& f, int d) {
                f.test_end = trade::Date(std::chrono::days(d));
            })
        .def_rw("train_days",    &trade::FoldResult::train_days)
        .def_rw("test_days",     &trade::FoldResult::test_days)
        .def_rw("train_perf",    &trade::FoldResult::train_perf)
        .def_rw("test_perf",     &trade::FoldResult::test_perf)
        .def_rw("train_sharpe",  &trade::FoldResult::train_sharpe)
        .def_rw("test_sharpe",   &trade::FoldResult::test_sharpe)
        .def_rw("overfit_ratio", &trade::FoldResult::overfit_ratio)
        .def("__repr__", [](const trade::FoldResult& f) {
            return "<FoldResult fold=" + std::to_string(f.fold_index) +
                   " train_sharpe=" + std::to_string(f.train_sharpe) +
                   " test_sharpe=" + std::to_string(f.test_sharpe) + ">";
        });

    // OverfitTestResults
    nb::class_<trade::OverfitTestResults>(bt, "OverfitTestResults")
        .def(nb::init<>())
        // DSR
        .def_rw("dsr",       &trade::OverfitTestResults::dsr)
        .def_rw("dsr_pass",  &trade::OverfitTestResults::dsr_pass)
        // PBO
        .def_rw("pbo",       &trade::OverfitTestResults::pbo)
        .def_rw("pbo_pass",  &trade::OverfitTestResults::pbo_pass)
        // MBL
        .def_rw("mbl_years",    &trade::OverfitTestResults::mbl_years)
        .def_rw("actual_years", &trade::OverfitTestResults::actual_years)
        .def_rw("mbl_pass",     &trade::OverfitTestResults::mbl_pass)
        // FDR
        .def_rw("fdr",           &trade::OverfitTestResults::fdr)
        .def_rw("fdr_pass",      &trade::OverfitTestResults::fdr_pass)
        .def_rw("num_rejected",  &trade::OverfitTestResults::num_rejected)
        .def_rw("num_trials",    &trade::OverfitTestResults::num_trials)
        // Bootstrap
        .def_rw("bootstrap_ci_lower", &trade::OverfitTestResults::bootstrap_ci_lower)
        .def_rw("bootstrap_ci_upper", &trade::OverfitTestResults::bootstrap_ci_upper)
        .def_rw("bootstrap_pass",     &trade::OverfitTestResults::bootstrap_pass)
        // Overall
        .def("all_pass",           &trade::OverfitTestResults::all_pass)
        .def("num_tests_passed",   &trade::OverfitTestResults::num_tests_passed)
        .def("__repr__", [](const trade::OverfitTestResults& r) {
            return "<OverfitTestResults passed=" +
                   std::to_string(r.num_tests_passed()) + "/5" +
                   " dsr=" + std::to_string(r.dsr) +
                   " pbo=" + std::to_string(r.pbo) + ">";
        });

    // ValidationResult
    nb::class_<trade::ValidationResult>(bt, "ValidationResult")
        .def(nb::init<>())
        .def_rw("method",              &trade::ValidationResult::method)
        .def_rw("folds",               &trade::ValidationResult::folds)
        .def_rw("overfit_tests",       &trade::ValidationResult::overfit_tests)
        .def_rw("mean_train_sharpe",   &trade::ValidationResult::mean_train_sharpe)
        .def_rw("mean_test_sharpe",    &trade::ValidationResult::mean_test_sharpe)
        .def_rw("std_test_sharpe",     &trade::ValidationResult::std_test_sharpe)
        .def_rw("sharpe_decay",        &trade::ValidationResult::sharpe_decay)
        .def_rw("mean_overfit_ratio",  &trade::ValidationResult::mean_overfit_ratio)
        .def("is_valid",               &trade::ValidationResult::is_valid)
        .def("__repr__", [](const trade::ValidationResult& v) {
            return "<ValidationResult method='" + v.method +
                   "' folds=" + std::to_string(v.folds.size()) +
                   " valid=" + (v.is_valid() ? "true" : "false") + ">";
        });

    // ========================================================================
    // BacktestValidator::Config
    // ========================================================================

    nb::class_<trade::BacktestValidator::Config>(bt, "BacktestValidatorConfig")
        .def(nb::init<>())
        // Walk-forward params
        .def_rw("wf_train_years", &trade::BacktestValidator::Config::wf_train_years)
        .def_rw("wf_test_years",  &trade::BacktestValidator::Config::wf_test_years)
        .def_rw("wf_step_years",  &trade::BacktestValidator::Config::wf_step_years)
        // Purged K-fold params
        .def_rw("num_folds",                &trade::BacktestValidator::Config::num_folds)
        .def_rw("prediction_horizon_days",  &trade::BacktestValidator::Config::prediction_horizon_days)
        .def_rw("min_embargo_days",         &trade::BacktestValidator::Config::min_embargo_days)
        .def_rw("embargo_pct",              &trade::BacktestValidator::Config::embargo_pct)
        // DSR params
        .def_rw("dsr_threshold",    &trade::BacktestValidator::Config::dsr_threshold)
        .def_rw("dsr_num_trials",   &trade::BacktestValidator::Config::dsr_num_trials)
        // PBO params
        .def_rw("pbo_threshold",    &trade::BacktestValidator::Config::pbo_threshold)
        .def_rw("pbo_num_subsets",  &trade::BacktestValidator::Config::pbo_num_subsets)
        // FDR params
        .def_rw("fdr_threshold",    &trade::BacktestValidator::Config::fdr_threshold)
        // Bootstrap params
        .def_rw("bootstrap_samples",    &trade::BacktestValidator::Config::bootstrap_samples)
        .def_rw("bootstrap_block_size", &trade::BacktestValidator::Config::bootstrap_block_size)
        .def_rw("bootstrap_confidence", &trade::BacktestValidator::Config::bootstrap_confidence)
        // General
        .def_rw("verbose",  &trade::BacktestValidator::Config::verbose);

    // ========================================================================
    // BacktestValidator::FDRResult
    // ========================================================================

    nb::class_<trade::BacktestValidator::FDRResult>(bt, "FDRResult")
        .def(nb::init<>())
        .def_rw("num_rejected",   &trade::BacktestValidator::FDRResult::num_rejected)
        .def_rw("estimated_fdr",  &trade::BacktestValidator::FDRResult::estimated_fdr)
        .def_rw("rejected",       &trade::BacktestValidator::FDRResult::rejected);

    // ========================================================================
    // BacktestValidator
    // ========================================================================

    nb::class_<trade::BacktestValidator>(bt, "BacktestValidator")
        .def(nb::init<
                std::shared_ptr<trade::IMarketDataFeed>,
                std::shared_ptr<trade::IExecutionVenue>,
                std::shared_ptr<trade::IClock>>(),
             "market_data"_a, "execution"_a, "clock"_a)
        .def(nb::init<
                std::shared_ptr<trade::IMarketDataFeed>,
                std::shared_ptr<trade::IExecutionVenue>,
                std::shared_ptr<trade::IClock>,
                trade::BacktestValidator::Config>(),
             "market_data"_a, "execution"_a, "clock"_a, "config"_a)
        // Validation methods (wrap Date parameters)
        .def("walk_forward", [](trade::BacktestValidator& v,
                                trade::IStrategy& strategy,
                                int full_start, int full_end,
                                const trade::BacktestEngine::Config& engine_config) {
            return v.walk_forward(
                strategy,
                trade::Date(std::chrono::days(full_start)),
                trade::Date(std::chrono::days(full_end)),
                engine_config);
        }, "strategy"_a, "full_start"_a, "full_end"_a,
           "engine_config"_a = trade::BacktestEngine::Config{})
        .def("purged_kfold", [](trade::BacktestValidator& v,
                                trade::IStrategy& strategy,
                                int full_start, int full_end,
                                const trade::BacktestEngine::Config& engine_config) {
            return v.purged_kfold(
                strategy,
                trade::Date(std::chrono::days(full_start)),
                trade::Date(std::chrono::days(full_end)),
                engine_config);
        }, "strategy"_a, "full_start"_a, "full_end"_a,
           "engine_config"_a = trade::BacktestEngine::Config{})
        .def("full_validation", [](trade::BacktestValidator& v,
                                   trade::IStrategy& strategy,
                                   int full_start, int full_end,
                                   const trade::BacktestEngine::Config& engine_config) {
            return v.full_validation(
                strategy,
                trade::Date(std::chrono::days(full_start)),
                trade::Date(std::chrono::days(full_end)),
                engine_config);
        }, "strategy"_a, "full_start"_a, "full_end"_a,
           "engine_config"_a = trade::BacktestEngine::Config{})
        // Anti-overfitting tests
        .def("compute_dsr", &trade::BacktestValidator::compute_dsr,
             "observed_sharpe"_a, "sharpe_estimates"_a, "num_observations"_a)
        .def("compute_pbo", &trade::BacktestValidator::compute_pbo,
             "returns_matrix"_a, "num_subsets"_a)
        .def("compute_mbl", &trade::BacktestValidator::compute_mbl,
             "target_sharpe"_a, "num_trials"_a,
             "skewness"_a = 0.0, "kurtosis"_a = 3.0)
        .def("benjamini_hochberg", &trade::BacktestValidator::benjamini_hochberg,
             "p_values"_a, "fdr_level"_a = 0.05)
        .def("run_overfit_tests", &trade::BacktestValidator::run_overfit_tests,
             "folds"_a, "daily_returns"_a)
        // Accessors
        .def("config", &trade::BacktestValidator::config,
             nb::rv_policy::reference_internal)
        .def("set_progress_callback", &trade::BacktestValidator::set_progress_callback,
             "callback"_a);

    // ========================================================================
    // Reporting types
    // ========================================================================

    // TradeDetail
    nb::class_<trade::TradeDetail>(bt, "TradeDetail")
        .def(nb::init<>())
        .def_rw("symbol",        &trade::TradeDetail::symbol)
        .def_rw("side",          &trade::TradeDetail::side)
        .def_prop_rw("entry_date",
            [](const trade::TradeDetail& t) {
                return static_cast<int>(t.entry_date.time_since_epoch().count());
            },
            [](trade::TradeDetail& t, int d) {
                t.entry_date = trade::Date(std::chrono::days(d));
            })
        .def_prop_rw("exit_date",
            [](const trade::TradeDetail& t) {
                return static_cast<int>(t.exit_date.time_since_epoch().count());
            },
            [](trade::TradeDetail& t, int d) {
                t.exit_date = trade::Date(std::chrono::days(d));
            })
        .def_rw("entry_price",   &trade::TradeDetail::entry_price)
        .def_rw("exit_price",    &trade::TradeDetail::exit_price)
        .def_rw("quantity",      &trade::TradeDetail::quantity)
        .def_rw("pnl",           &trade::TradeDetail::pnl)
        .def_rw("return_pct",    &trade::TradeDetail::return_pct)
        .def_rw("holding_days",  &trade::TradeDetail::holding_days)
        .def_rw("commission",    &trade::TradeDetail::commission)
        .def_rw("slippage_cost", &trade::TradeDetail::slippage_cost)
        .def_rw("entry_reason",  &trade::TradeDetail::entry_reason)
        .def_rw("exit_reason",   &trade::TradeDetail::exit_reason);

    // DailyAttribution
    nb::class_<trade::DailyAttribution>(bt, "DailyAttribution")
        .def(nb::init<>())
        .def_prop_rw("date",
            [](const trade::DailyAttribution& a) {
                return static_cast<int>(a.date.time_since_epoch().count());
            },
            [](trade::DailyAttribution& a, int d) {
                a.date = trade::Date(std::chrono::days(d));
            })
        .def_rw("total_return",   &trade::DailyAttribution::total_return)
        .def_rw("market_return",  &trade::DailyAttribution::market_return)
        .def_rw("alpha_return",   &trade::DailyAttribution::alpha_return)
        .def_rw("cost_return",    &trade::DailyAttribution::cost_return)
        .def_rw("timing_return",  &trade::DailyAttribution::timing_return);

    // AttributionSummary
    nb::class_<trade::AttributionSummary>(bt, "AttributionSummary")
        .def(nb::init<>())
        .def_rw("total_return",          &trade::AttributionSummary::total_return)
        .def_rw("market_contribution",   &trade::AttributionSummary::market_contribution)
        .def_rw("alpha_contribution",    &trade::AttributionSummary::alpha_contribution)
        .def_rw("cost_drag",             &trade::AttributionSummary::cost_drag)
        .def_rw("timing_contribution",   &trade::AttributionSummary::timing_contribution)
        .def_rw("residual",              &trade::AttributionSummary::residual)
        .def("market_pct", &trade::AttributionSummary::market_pct)
        .def("alpha_pct",  &trade::AttributionSummary::alpha_pct)
        .def("cost_pct",   &trade::AttributionSummary::cost_pct);

    // FactorExposure
    nb::class_<trade::FactorExposure>(bt, "FactorExposure")
        .def(nb::init<>())
        .def_rw("factor_name",   &trade::FactorExposure::factor_name)
        .def_prop_rw("dates",
            [](const trade::FactorExposure& f) {
                std::vector<int> out;
                out.reserve(f.dates.size());
                for (const auto& d : f.dates)
                    out.push_back(static_cast<int>(d.time_since_epoch().count()));
                return out;
            },
            [](trade::FactorExposure& f, const std::vector<int>& v) {
                f.dates.clear();
                f.dates.reserve(v.size());
                for (int d : v)
                    f.dates.push_back(trade::Date(std::chrono::days(d)));
            })
        .def_rw("exposures",     &trade::FactorExposure::exposures)
        .def_rw("avg_exposure",  &trade::FactorExposure::avg_exposure)
        .def_rw("std_exposure",  &trade::FactorExposure::std_exposure)
        .def_rw("max_exposure",  &trade::FactorExposure::max_exposure)
        .def_rw("min_exposure",  &trade::FactorExposure::min_exposure);

    // EquityCurvePoint
    nb::class_<trade::EquityCurvePoint>(bt, "EquityCurvePoint")
        .def(nb::init<>())
        .def_prop_rw("date",
            [](const trade::EquityCurvePoint& p) {
                return static_cast<int>(p.date.time_since_epoch().count());
            },
            [](trade::EquityCurvePoint& p, int d) {
                p.date = trade::Date(std::chrono::days(d));
            })
        .def_rw("strategy_nav",  &trade::EquityCurvePoint::strategy_nav)
        .def_rw("benchmark_nav", &trade::EquityCurvePoint::benchmark_nav)
        .def_rw("active_nav",    &trade::EquityCurvePoint::active_nav)
        .def_rw("drawdown",      &trade::EquityCurvePoint::drawdown);

    // BacktestReport::DailyPositionSnapshot
    nb::class_<trade::BacktestReport::DailyPositionSnapshot>(bt, "DailyPositionSnapshot")
        .def(nb::init<>())
        .def_prop_rw("date",
            [](const trade::BacktestReport::DailyPositionSnapshot& s) {
                return static_cast<int>(s.date.time_since_epoch().count());
            },
            [](trade::BacktestReport::DailyPositionSnapshot& s, int d) {
                s.date = trade::Date(std::chrono::days(d));
            })
        .def_rw("top_positions",         &trade::BacktestReport::DailyPositionSnapshot::top_positions)
        .def_rw("total_positions",       &trade::BacktestReport::DailyPositionSnapshot::total_positions)
        .def_rw("concentration_top5",    &trade::BacktestReport::DailyPositionSnapshot::concentration_top5)
        .def_rw("concentration_top10",   &trade::BacktestReport::DailyPositionSnapshot::concentration_top10);

    // BacktestReport::CostSummary
    nb::class_<trade::BacktestReport::CostSummary>(bt, "CostSummary")
        .def(nb::init<>())
        .def_rw("total_commission",            &trade::BacktestReport::CostSummary::total_commission)
        .def_rw("total_stamp_tax",             &trade::BacktestReport::CostSummary::total_stamp_tax)
        .def_rw("total_transfer_fee",          &trade::BacktestReport::CostSummary::total_transfer_fee)
        .def_rw("total_slippage",              &trade::BacktestReport::CostSummary::total_slippage)
        .def_rw("total_costs",                 &trade::BacktestReport::CostSummary::total_costs)
        .def_rw("cost_as_annual_return_drag",  &trade::BacktestReport::CostSummary::cost_as_annual_return_drag);

    // BacktestReport
    nb::class_<trade::BacktestReport>(bt, "BacktestReport")
        .def(nb::init<>())
        // Metadata
        .def_rw("strategy_name",    &trade::BacktestReport::strategy_name)
        .def_rw("strategy_version", &trade::BacktestReport::strategy_version)
        .def_rw("strategy_params",  &trade::BacktestReport::strategy_params)
        .def_prop_rw("start_date",
            [](const trade::BacktestReport& r) {
                return static_cast<int>(r.start_date.time_since_epoch().count());
            },
            [](trade::BacktestReport& r, int d) {
                r.start_date = trade::Date(std::chrono::days(d));
            })
        .def_prop_rw("end_date",
            [](const trade::BacktestReport& r) {
                return static_cast<int>(r.end_date.time_since_epoch().count());
            },
            [](trade::BacktestReport& r, int d) {
                r.end_date = trade::Date(std::chrono::days(d));
            })
        .def_rw("initial_capital",   &trade::BacktestReport::initial_capital)
        .def_rw("trading_days",      &trade::BacktestReport::trading_days)
        .def_rw("benchmark_name",    &trade::BacktestReport::benchmark_name)
        .def_rw("generated_at",      &trade::BacktestReport::generated_at)
        // Performance
        .def_rw("performance",       &trade::BacktestReport::performance)
        // Equity curve
        .def_rw("equity_curve",      &trade::BacktestReport::equity_curve)
        // Drawdown curve (vector<pair<Date, double>> -> Python-friendly)
        .def_prop_rw("drawdown_curve",
            [](const trade::BacktestReport& r) {
                std::vector<std::pair<int, double>> out;
                out.reserve(r.drawdown_curve.size());
                for (const auto& [d, v] : r.drawdown_curve)
                    out.emplace_back(static_cast<int>(d.time_since_epoch().count()), v);
                return out;
            },
            [](trade::BacktestReport& r,
               const std::vector<std::pair<int, double>>& v) {
                r.drawdown_curve.clear();
                r.drawdown_curve.reserve(v.size());
                for (const auto& [d, val] : v)
                    r.drawdown_curve.emplace_back(trade::Date(std::chrono::days(d)), val);
            })
        // Factor exposures
        .def_rw("factor_exposures",   &trade::BacktestReport::factor_exposures)
        // Trade details
        .def_rw("trades",            &trade::BacktestReport::trades)
        // Attribution
        .def_rw("daily_attribution",    &trade::BacktestReport::daily_attribution)
        .def_rw("attribution_summary",  &trade::BacktestReport::attribution_summary)
        // Overfit tests
        .def_rw("overfit_tests",     &trade::BacktestReport::overfit_tests)
        // Validation
        .def_rw("validation",        &trade::BacktestReport::validation)
        // Position history
        .def_rw("position_history",  &trade::BacktestReport::position_history)
        // Cost summary
        .def_rw("cost_summary",      &trade::BacktestReport::cost_summary);

    // ========================================================================
    // BacktestReporter::Config
    // ========================================================================

    nb::class_<trade::BacktestReporter::Config>(bt, "BacktestReporterConfig")
        .def(nb::init<>())
        .def_rw("top_n_positions",          &trade::BacktestReporter::Config::top_n_positions)
        .def_rw("include_trade_details",    &trade::BacktestReporter::Config::include_trade_details)
        .def_rw("include_daily_attribution",&trade::BacktestReporter::Config::include_daily_attribution)
        .def_rw("include_factor_exposures", &trade::BacktestReporter::Config::include_factor_exposures)
        .def_rw("include_position_history", &trade::BacktestReporter::Config::include_position_history)
        .def_rw("benchmark_name",           &trade::BacktestReporter::Config::benchmark_name);

    // ========================================================================
    // BacktestReporter
    // ========================================================================

    nb::class_<trade::BacktestReporter>(bt, "BacktestReporter")
        .def(nb::init<>())
        .def(nb::init<trade::BacktestReporter::Config>(), "config"_a)
        // Report generation
        .def("generate",
            nb::overload_cast<const trade::BacktestResult&, const trade::IStrategy&>(
                &trade::BacktestReporter::generate, nb::const_),
            "result"_a, "strategy"_a)
        .def("generate_with_benchmark",
            nb::overload_cast<const trade::BacktestResult&, const trade::IStrategy&,
                              const std::vector<double>&>(
                &trade::BacktestReporter::generate, nb::const_),
            "result"_a, "strategy"_a, "benchmark_returns"_a)
        .def("generate_with_validation",
            nb::overload_cast<const trade::BacktestResult&, const trade::IStrategy&,
                              const std::vector<double>&, const trade::ValidationResult&>(
                &trade::BacktestReporter::generate, nb::const_),
            "result"_a, "strategy"_a, "benchmark_returns"_a, "validation"_a)
        // Output formats (static)
        .def_static("to_json",          &trade::BacktestReporter::to_json, "report"_a)
        .def_static("to_json_file",     &trade::BacktestReporter::to_json_file,
                    "report"_a, "path"_a)
        .def_static("equity_curve_to_csv",   &trade::BacktestReporter::equity_curve_to_csv,
                    "report"_a)
        .def_static("trades_to_csv",         &trade::BacktestReporter::trades_to_csv,
                    "report"_a)
        .def_static("monthly_returns_to_csv", &trade::BacktestReporter::monthly_returns_to_csv,
                    "report"_a)
        .def_static("to_text_summary", &trade::BacktestReporter::to_text_summary, "report"_a)
        // Component builders
        .def("build_equity_curve", &trade::BacktestReporter::build_equity_curve,
             "records"_a, "benchmark_returns"_a)
        .def("build_trade_details", &trade::BacktestReporter::build_trade_details,
             "records"_a)
        .def("build_attribution", &trade::BacktestReporter::build_attribution,
             "records"_a, "benchmark_returns"_a, "beta"_a)
        .def("summarise_attribution", &trade::BacktestReporter::summarise_attribution,
             "daily"_a)
        .def("build_position_history", &trade::BacktestReporter::build_position_history,
             "records"_a)
        .def("build_cost_summary", &trade::BacktestReporter::build_cost_summary,
             "records"_a, "initial_capital"_a, "trading_days"_a)
        .def("config", &trade::BacktestReporter::config,
             nb::rv_policy::reference_internal);

    // ========================================================================
    // Strategy interfaces (read-only from Python; strategies written in C++)
    // ========================================================================

    // IStrategy (abstract, not constructible from Python)
    nb::class_<trade::IStrategy>(bt, "IStrategy")
        .def("name",           &trade::IStrategy::name)
        .def("description",    &trade::IStrategy::description)
        .def("version",        &trade::IStrategy::version)
        .def("params_summary", &trade::IStrategy::params_summary);

    // StrategyBase::Config
    nb::class_<trade::StrategyBase::Config>(bt, "StrategyBaseConfig")
        .def(nb::init<>())
        .def_rw("max_positions",         &trade::StrategyBase::Config::max_positions)
        .def_rw("min_positions",         &trade::StrategyBase::Config::min_positions)
        .def_rw("max_single_weight",     &trade::StrategyBase::Config::max_single_weight)
        .def_rw("max_turnover_per_day",  &trade::StrategyBase::Config::max_turnover_per_day)
        .def_rw("min_adv_participation", &trade::StrategyBase::Config::min_adv_participation)
        .def_rw("rebalance_threshold",   &trade::StrategyBase::Config::rebalance_threshold)
        .def_rw("min_listing_days",      &trade::StrategyBase::Config::min_listing_days)
        .def_rw("exclude_st",           &trade::StrategyBase::Config::exclude_st);

    // StrategyBase (abstract, not constructible from Python)
    nb::class_<trade::StrategyBase, trade::IStrategy>(bt, "StrategyBase")
        .def("strategy_config", &trade::StrategyBase::strategy_config,
             nb::rv_policy::reference_internal);

    // ========================================================================
    // TradingCostConfig (from common/config.h, used by BrokerSim)
    // ========================================================================

    nb::class_<trade::TradingCostConfig>(bt, "TradingCostConfig")
        .def(nb::init<>())
        .def_rw("stamp_tax_rate",      &trade::TradingCostConfig::stamp_tax_rate)
        .def_rw("commission_rate",     &trade::TradingCostConfig::commission_rate)
        .def_rw("commission_min_yuan", &trade::TradingCostConfig::commission_min_yuan)
        .def_rw("transfer_fee_rate",   &trade::TradingCostConfig::transfer_fee_rate);
}
