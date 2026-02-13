#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/eigen/dense.h>

#include "trade/risk/var.h"
#include "trade/risk/covariance.h"
#include "trade/risk/kelly.h"
#include "trade/risk/position_sizer.h"
#include "trade/risk/drawdown.h"
#include "trade/risk/stress_test.h"
#include "trade/risk/risk_monitor.h"
#include "trade/risk/risk_attribution.h"

namespace nb = nanobind;
using namespace nb::literals;

void bind_risk(nb::module_& parent) {
    auto m = parent.def_submodule("risk", "Risk management module");

    // ===================================================================
    // VaRCalculator
    // ===================================================================

    using VaRC = trade::VaRCalculator;

    nb::class_<VaRC::Config>(m, "VaRConfig")
        .def(nb::init<>())
        .def_rw("confidence_level",  &VaRC::Config::confidence_level)
        .def_rw("historical_window", &VaRC::Config::historical_window)
        .def_rw("mc_simulations",    &VaRC::Config::mc_simulations)
        .def_rw("mc_t_df",           &VaRC::Config::mc_t_df)
        .def_rw("horizon_days",      &VaRC::Config::horizon_days)
        .def_rw("random_seed",       &VaRC::Config::random_seed);

    nb::class_<VaRC::VaRResult>(m, "VaRResult")
        .def(nb::init<>())
        .def_rw("var",        &VaRC::VaRResult::var)
        .def_rw("cvar",       &VaRC::VaRResult::cvar)
        .def_rw("confidence", &VaRC::VaRResult::confidence)
        .def_rw("method",     &VaRC::VaRResult::method);

    nb::class_<VaRC::CombinedVaR>(m, "CombinedVaR")
        .def(nb::init<>())
        .def_rw("parametric",   &VaRC::CombinedVaR::parametric)
        .def_rw("historical",   &VaRC::CombinedVaR::historical)
        .def_rw("monte_carlo",  &VaRC::CombinedVaR::monte_carlo)
        .def_rw("var_1d_99",    &VaRC::CombinedVaR::var_1d_99)
        .def_rw("cvar_1d_99",   &VaRC::CombinedVaR::cvar_1d_99);

    nb::class_<VaRC>(m, "VaRCalculator")
        .def(nb::init<>())
        .def(nb::init<VaRC::Config>(), "config"_a)
        .def("parametric_var", &VaRC::parametric_var,
             "weights"_a, "cov"_a, "mu"_a = Eigen::VectorXd{})
        .def("historical_var", &VaRC::historical_var,
             "weights"_a, "returns_matrix"_a)
        .def("monte_carlo_var", &VaRC::monte_carlo_var,
             "weights"_a, "cov"_a, "mu"_a,
             "adv"_a = Eigen::VectorXd{}, "positions"_a = Eigen::VectorXd{})
        .def("compute", &VaRC::compute,
             "weights"_a, "cov"_a, "returns_matrix"_a,
             "mu"_a = Eigen::VectorXd{}, "adv"_a = Eigen::VectorXd{},
             "positions"_a = Eigen::VectorXd{})
        .def("marginal_var", &VaRC::marginal_var,
             "weights"_a, "cov"_a)
        .def("component_var", &VaRC::component_var,
             "weights"_a, "cov"_a)
        .def("config", &VaRC::config, nb::rv_policy::reference_internal);

    // ===================================================================
    // CovarianceEstimator
    // ===================================================================

    using CovE = trade::CovarianceEstimator;

    nb::class_<CovE::Config>(m, "CovarianceConfig")
        .def(nb::init<>())
        .def_rw("lookback_days",        &CovE::Config::lookback_days)
        .def_rw("min_shrinkage",        &CovE::Config::min_shrinkage)
        .def_rw("max_shrinkage",        &CovE::Config::max_shrinkage)
        .def_rw("use_exponential_decay", &CovE::Config::use_exponential_decay)
        .def_rw("ewma_halflife",        &CovE::Config::ewma_halflife);

    nb::class_<CovE::EigenDecomp>(m, "EigenDecomp")
        .def(nb::init<>())
        .def_rw("eigenvalues",      &CovE::EigenDecomp::eigenvalues)
        .def_rw("eigenvectors",     &CovE::EigenDecomp::eigenvectors)
        .def_rw("condition_number", &CovE::EigenDecomp::condition_number);

    nb::class_<CovE>(m, "CovarianceEstimator")
        .def(nb::init<>())
        .def(nb::init<CovE::Config>(), "config"_a)
        .def("estimate", &CovE::estimate, "returns_matrix"_a)
        .def("shrinkage_intensity", &CovE::shrinkage_intensity)
        .def_static("to_correlation", &CovE::to_correlation, "cov"_a)
        .def_static("annualised_vol", &CovE::annualised_vol, "cov"_a)
        .def_static("decompose", &CovE::decompose, "cov"_a)
        .def_static("build_returns_matrix", &CovE::build_returns_matrix,
                     "series"_a, "lookback_days"_a = 250)
        .def("config", &CovE::config, nb::rv_policy::reference_internal);

    // ===================================================================
    // KellyCalculator
    // ===================================================================

    using KellyC = trade::KellyCalculator;

    nb::class_<KellyC::Config>(m, "KellyConfig")
        .def(nb::init<>())
        .def_rw("kelly_fraction",          &KellyC::Config::kelly_fraction)
        .def_rw("f_max",                   &KellyC::Config::f_max)
        .def_rw("target_gross_exposure",   &KellyC::Config::target_gross_exposure)
        .def_rw("min_confidence",          &KellyC::Config::min_confidence);

    nb::class_<KellyC::KellyDiagnostics>(m, "KellyDiagnostics")
        .def(nb::init<>())
        .def_rw("raw_kelly",        &KellyC::KellyDiagnostics::raw_kelly)
        .def_rw("clamped_kelly",    &KellyC::KellyDiagnostics::clamped_kelly)
        .def_rw("risk_budgets",     &KellyC::KellyDiagnostics::risk_budgets)
        .def_rw("final_weights",    &KellyC::KellyDiagnostics::final_weights)
        .def_rw("implied_leverage", &KellyC::KellyDiagnostics::implied_leverage)
        .def_rw("effective_n",      &KellyC::KellyDiagnostics::effective_n);

    nb::class_<KellyC>(m, "KellyCalculator")
        .def(nb::init<>())
        .def(nb::init<KellyC::Config>(), "config"_a)
        .def("kelly_fraction", &KellyC::kelly_fraction,
             "mu"_a, "sigma"_a)
        .def("risk_budget", &KellyC::risk_budget,
             "kelly"_a, "confidence"_a)
        .def("risk_parity_weights", &KellyC::risk_parity_weights,
             "risk_budget"_a, "sigma"_a)
        .def("compute_weights", &KellyC::compute_weights,
             "mu"_a, "sigma"_a, "confidence"_a)
        .def("compute_diagnostics", &KellyC::compute_diagnostics,
             "mu"_a, "sigma"_a, "confidence"_a)
        .def("config", &KellyC::config, nb::rv_policy::reference_internal);

    // ===================================================================
    // PositionSizer
    // ===================================================================

    using PS = trade::PositionSizer;

    nb::class_<PS::Constraints>(m, "PositionSizerConstraints")
        .def(nb::init<>())
        .def_rw("single_stock_soft_pct",   &PS::Constraints::single_stock_soft_pct)
        .def_rw("single_stock_hard_pct",   &PS::Constraints::single_stock_hard_pct)
        .def_rw("liquidity_adv_fraction",  &PS::Constraints::liquidity_adv_fraction)
        .def_rw("industry_soft_pct",       &PS::Constraints::industry_soft_pct)
        .def_rw("industry_hard_pct",       &PS::Constraints::industry_hard_pct)
        .def_rw("top_n",                   &PS::Constraints::top_n)
        .def_rw("top_n_combined_pct",      &PS::Constraints::top_n_combined_pct)
        .def_rw("factor_exposure_abs_max", &PS::Constraints::factor_exposure_abs_max)
        .def_rw("beta_min",               &PS::Constraints::beta_min)
        .def_rw("beta_max",               &PS::Constraints::beta_max)
        .def_rw("max_liquidation_days",   &PS::Constraints::max_liquidation_days)
        .def_rw("low_vol_cap_min",        &PS::Constraints::low_vol_cap_min)
        .def_rw("low_vol_cap_max",        &PS::Constraints::low_vol_cap_max)
        .def_rw("mid_vol_cap_min",        &PS::Constraints::mid_vol_cap_min)
        .def_rw("mid_vol_cap_max",        &PS::Constraints::mid_vol_cap_max)
        .def_rw("high_vol_cap_min",       &PS::Constraints::high_vol_cap_min)
        .def_rw("high_vol_cap_max",       &PS::Constraints::high_vol_cap_max)
        .def_rw("low_vol_threshold",      &PS::Constraints::low_vol_threshold)
        .def_rw("high_vol_threshold",     &PS::Constraints::high_vol_threshold)
        .def_rw("max_iterations",         &PS::Constraints::max_iterations)
        .def_rw("convergence_tol",        &PS::Constraints::convergence_tol);

    nb::class_<PS::StockRisk>(m, "StockRisk")
        .def(nb::init<>())
        .def_rw("symbol",            &PS::StockRisk::symbol)
        .def_rw("industry",          &PS::StockRisk::industry)
        .def_rw("annualised_vol",    &PS::StockRisk::annualised_vol)
        .def_rw("beta",              &PS::StockRisk::beta)
        .def_rw("adv_20d",           &PS::StockRisk::adv_20d)
        .def_rw("position_notional", &PS::StockRisk::position_notional);

    nb::class_<PS::SizingResult::Violations>(m, "SizingViolations")
        .def(nb::init<>())
        .def_rw("single_stock",     &PS::SizingResult::Violations::single_stock)
        .def_rw("liquidity",        &PS::SizingResult::Violations::liquidity)
        .def_rw("industry",         &PS::SizingResult::Violations::industry)
        .def_rw("top_n",            &PS::SizingResult::Violations::top_n)
        .def_rw("factor_exposure",  &PS::SizingResult::Violations::factor_exposure)
        .def_rw("beta",             &PS::SizingResult::Violations::beta)
        .def_rw("liquidation",      &PS::SizingResult::Violations::liquidation)
        .def_rw("vol_bucket",       &PS::SizingResult::Violations::vol_bucket);

    nb::class_<PS::SizingResult>(m, "SizingResult")
        .def(nb::init<>())
        .def_rw("weights",             &PS::SizingResult::weights)
        .def_rw("symbols",             &PS::SizingResult::symbols)
        .def_rw("portfolio_beta",      &PS::SizingResult::portfolio_beta)
        .def_rw("liquidation_days",    &PS::SizingResult::liquidation_days)
        .def_rw("gross_exposure",      &PS::SizingResult::gross_exposure)
        .def_rw("max_single_stock",    &PS::SizingResult::max_single_stock)
        .def_rw("max_industry_weight", &PS::SizingResult::max_industry_weight)
        .def_rw("top_n_combined",      &PS::SizingResult::top_n_combined)
        .def_rw("iterations_used",     &PS::SizingResult::iterations_used)
        .def_rw("converged",           &PS::SizingResult::converged)
        .def_rw("violations",          &PS::SizingResult::violations);

    nb::class_<PS>(m, "PositionSizer")
        .def(nb::init<>())
        .def(nb::init<PS::Constraints>(), "constraints"_a)
        .def("size_positions", &PS::size_positions,
             "alphas"_a, "risks"_a,
             "factor_loadings"_a = Eigen::MatrixXd{},
             "cov"_a = Eigen::MatrixXd{})
        .def_static("clamp_single_stock", &PS::clamp_single_stock,
                     "weights"_a, "hard_pct"_a)
        .def_static("clamp_industry", &PS::clamp_industry,
                     "weights"_a, "industries"_a, "hard_pct"_a)
        .def_static("clamp_top_n", &PS::clamp_top_n,
                     "weights"_a, "n"_a, "max_combined"_a)
        .def_static("adjust_beta", &PS::adjust_beta,
                     "weights"_a, "betas"_a, "beta_min"_a, "beta_max"_a)
        .def_static("compute_liquidation_days", &PS::compute_liquidation_days,
                     "weights"_a, "adv"_a, "nav"_a)
        .def("vol_bucket_cap", &PS::vol_bucket_cap, "annualised_vol"_a)
        .def("constraints", &PS::constraints, nb::rv_policy::reference_internal)
        .def("set_constraints", &PS::set_constraints, "c"_a);

    // ===================================================================
    // DrawdownController
    // ===================================================================

    using DDC = trade::DrawdownController;

    nb::enum_<DDC::DrawdownLevel>(m, "DrawdownLevel")
        .value("kNormal",          DDC::DrawdownLevel::kNormal)
        .value("kLevel1",          DDC::DrawdownLevel::kLevel1)
        .value("kLevel2",          DDC::DrawdownLevel::kLevel2)
        .value("kLevel3",          DDC::DrawdownLevel::kLevel3)
        .value("kCapitalPreserve", DDC::DrawdownLevel::kCapitalPreserve);

    nb::class_<DDC::Config>(m, "DrawdownConfig")
        .def(nb::init<>())
        .def_rw("level1_threshold",           &DDC::Config::level1_threshold)
        .def_rw("level2_threshold",           &DDC::Config::level2_threshold)
        .def_rw("level3_threshold",           &DDC::Config::level3_threshold)
        .def_rw("capital_preserve_threshold",  &DDC::Config::capital_preserve_threshold)
        .def_rw("level1_reduction",           &DDC::Config::level1_reduction)
        .def_rw("level2_reduction",           &DDC::Config::level2_reduction)
        .def_rw("level3_reduction",           &DDC::Config::level3_reduction)
        .def_rw("level3_single_stock_cap",    &DDC::Config::level3_single_stock_cap)
        .def_rw("target_vol",                 &DDC::Config::target_vol)
        .def_rw("vol_scale_floor",            &DDC::Config::vol_scale_floor)
        .def_rw("realized_vol_window",        &DDC::Config::realized_vol_window)
        .def_rw("lock_prob_threshold",        &DDC::Config::lock_prob_threshold)
        .def_rw("min_cash_buffer",            &DDC::Config::min_cash_buffer)
        .def_rw("lock_cap_reduction",         &DDC::Config::lock_cap_reduction)
        .def_rw("high_beta_threshold",        &DDC::Config::high_beta_threshold);

    nb::class_<DDC::LockInfo>(m, "LockInfo")
        .def(nb::init<>())
        .def_rw("symbol",           &DDC::LockInfo::symbol)
        .def_rw("lock_probability", &DDC::LockInfo::lock_probability)
        .def_rw("is_locked",        &DDC::LockInfo::is_locked)
        .def_rw("is_suspended",     &DDC::LockInfo::is_suspended);

    nb::class_<DDC::DrawdownAction>(m, "DrawdownAction")
        .def(nb::init<>())
        .def_rw("level",                           &DDC::DrawdownAction::level)
        .def_rw("current_drawdown",                &DDC::DrawdownAction::current_drawdown)
        .def_rw("nav_peak",                        &DDC::DrawdownAction::nav_peak)
        .def_rw("nav_current",                     &DDC::DrawdownAction::nav_current)
        .def_rw("target_exposure_multiplier",      &DDC::DrawdownAction::target_exposure_multiplier)
        .def_rw("vol_scale",                       &DDC::DrawdownAction::vol_scale)
        .def_rw("effective_multiplier",            &DDC::DrawdownAction::effective_multiplier)
        .def_rw("freeze_high_beta_new",            &DDC::DrawdownAction::freeze_high_beta_new)
        .def_rw("cut_lowest_confidence_quartile",  &DDC::DrawdownAction::cut_lowest_confidence_quartile)
        .def_rw("single_stock_cap",                &DDC::DrawdownAction::single_stock_cap)
        .def_rw("capital_preservation_mode",       &DDC::DrawdownAction::capital_preservation_mode)
        .def_rw("required_cash_buffer",            &DDC::DrawdownAction::required_cash_buffer)
        .def_rw("high_lock_prob_count",            &DDC::DrawdownAction::high_lock_prob_count)
        .def_rw("locked_symbols",                  &DDC::DrawdownAction::locked_symbols);

    nb::class_<DDC::VolScaling>(m, "VolScaling")
        .def(nb::init<>())
        .def_rw("realized_vol_20d", &DDC::VolScaling::realized_vol_20d)
        .def_rw("target_vol",       &DDC::VolScaling::target_vol)
        .def_rw("raw_scale",        &DDC::VolScaling::raw_scale)
        .def_rw("clamped_scale",    &DDC::VolScaling::clamped_scale);

    nb::class_<DDC>(m, "DrawdownController")
        .def(nb::init<>())
        .def(nb::init<DDC::Config>(), "config"_a)
        .def("evaluate", &DDC::evaluate,
             "nav_series"_a, "returns"_a,
             "locks"_a = std::vector<DDC::LockInfo>{},
             "betas"_a = std::unordered_map<trade::Symbol, double>{})
        .def_static("compute_drawdown", &DDC::compute_drawdown, "nav_series"_a)
        .def_static("compute_peak", &DDC::compute_peak, "nav_series"_a)
        .def("classify_drawdown", &DDC::classify_drawdown, "drawdown"_a)
        .def("compute_vol_scaling", &DDC::compute_vol_scaling, "returns"_a)
        .def_static("realized_vol", &DDC::realized_vol, "returns"_a, "window"_a)
        .def("adjust_weights", &DDC::adjust_weights,
             "weights"_a, "action"_a,
             "confidence"_a = Eigen::VectorXd{},
             "betas"_a = Eigen::VectorXd{})
        .def("apply_lock_adjustments", &DDC::apply_lock_adjustments,
             "weights"_a, "locks"_a, "single_stock_cap"_a)
        .def("config", &DDC::config, nb::rv_policy::reference_internal)
        .def("set_config", &DDC::set_config, "c"_a);

    // ===================================================================
    // StressTester
    // ===================================================================

    using ST = trade::StressTester;

    nb::enum_<ST::ScenarioType>(m, "ScenarioType")
        .value("kHistorical",      ST::ScenarioType::kHistorical)
        .value("kFactorShock",     ST::ScenarioType::kFactorShock)
        .value("kLiquidityStress", ST::ScenarioType::kLiquidityStress)
        .value("kCustom",          ST::ScenarioType::kCustom);

    nb::class_<ST::Scenario>(m, "Scenario")
        .def(nb::init<>())
        .def_rw("name",                     &ST::Scenario::name)
        .def_rw("type",                     &ST::Scenario::type)
        .def_rw("description",              &ST::Scenario::description)
        .def_rw("scenario_returns",         &ST::Scenario::scenario_returns)
        .def_rw("factor_shocks",            &ST::Scenario::factor_shocks)
        .def_rw("freeze_illiquid_quartile", &ST::Scenario::freeze_illiquid_quartile)
        .def_rw("freeze_days",              &ST::Scenario::freeze_days)
        .def_rw("forced_exit_adv_pct",      &ST::Scenario::forced_exit_adv_pct)
        .def_rw("slippage_bps_low",         &ST::Scenario::slippage_bps_low)
        .def_rw("slippage_bps_high",        &ST::Scenario::slippage_bps_high)
        .def_rw("duration_days",            &ST::Scenario::duration_days);

    nb::class_<ST::ScenarioResult::Contributor>(m, "ScenarioContributor")
        .def(nb::init<>())
        .def_rw("symbol",   &ST::ScenarioResult::Contributor::symbol)
        .def_rw("loss_pct", &ST::ScenarioResult::Contributor::loss_pct)
        .def_rw("weight",   &ST::ScenarioResult::Contributor::weight);

    nb::class_<ST::ScenarioResult>(m, "ScenarioResult")
        .def(nb::init<>())
        .def_rw("scenario_name",          &ST::ScenarioResult::scenario_name)
        .def_rw("type",                   &ST::ScenarioResult::type)
        .def_rw("total_loss_pct",         &ST::ScenarioResult::total_loss_pct)
        .def_rw("worst_day_loss_pct",     &ST::ScenarioResult::worst_day_loss_pct)
        .def_rw("stress_var_99",          &ST::ScenarioResult::stress_var_99)
        .def_rw("symbols",               &ST::ScenarioResult::symbols)
        .def_rw("loss_contribution",      &ST::ScenarioResult::loss_contribution)
        .def_rw("top_contributors",       &ST::ScenarioResult::top_contributors)
        .def_rw("liquidity_adjusted_loss", &ST::ScenarioResult::liquidity_adjusted_loss)
        .def_rw("slippage_cost",          &ST::ScenarioResult::slippage_cost)
        .def_rw("frozen_positions",       &ST::ScenarioResult::frozen_positions);

    nb::class_<ST::StressReport>(m, "StressReport")
        .def(nb::init<>())
        .def_rw("results",                &ST::StressReport::results)
        .def_rw("worst_scenario_loss",     &ST::StressReport::worst_scenario_loss)
        .def_rw("worst_scenario_name",     &ST::StressReport::worst_scenario_name)
        .def_rw("worst_stress_var_99",     &ST::StressReport::worst_stress_var_99)
        .def_rw("worst_liquidity_3d_loss", &ST::StressReport::worst_liquidity_3d_loss)
        .def_rw("pass_scenario_loss",      &ST::StressReport::pass_scenario_loss)
        .def_rw("pass_stress_var",         &ST::StressReport::pass_stress_var)
        .def_rw("pass_liquidity_loss",     &ST::StressReport::pass_liquidity_loss)
        .def_rw("overall_pass",            &ST::StressReport::overall_pass);

    nb::class_<ST::Config>(m, "StressTestConfig")
        .def(nb::init<>())
        .def_rw("max_scenario_loss",    &ST::Config::max_scenario_loss)
        .def_rw("max_stress_var_99",    &ST::Config::max_stress_var_99)
        .def_rw("max_liquidity_3d_loss", &ST::Config::max_liquidity_3d_loss)
        .def_rw("top_n_contributors",   &ST::Config::top_n_contributors)
        .def_rw("illiquid_quantile",    &ST::Config::illiquid_quantile);

    nb::class_<ST>(m, "StressTester")
        .def(nb::init<>())
        .def(nb::init<ST::Config>(), "config"_a)
        .def_static("make_2015_crash",          &ST::make_2015_crash)
        .def_static("make_2020_covid",          &ST::make_2020_covid)
        .def_static("make_2024_regulatory",     &ST::make_2024_regulatory)
        .def_static("make_northbound_reversal", &ST::make_northbound_reversal)
        .def_static("make_csi300_crash",        &ST::make_csi300_crash)
        .def_static("make_momentum_crash",      &ST::make_momentum_crash)
        .def_static("make_liquidity_stress",    &ST::make_liquidity_stress)
        .def("run_scenario", &ST::run_scenario,
             "weights"_a, "scenario"_a,
             "factor_loadings"_a = Eigen::MatrixXd{},
             "adv"_a = Eigen::VectorXd{},
             "symbols"_a = std::vector<trade::Symbol>{})
        .def("run_all", &ST::run_all,
             "weights"_a,
             "factor_loadings"_a = Eigen::MatrixXd{},
             "adv"_a = Eigen::VectorXd{},
             "symbols"_a = std::vector<trade::Symbol>{},
             "additional_scenarios"_a = std::vector<ST::Scenario>{})
        .def("add_scenario", &ST::add_scenario, "scenario"_a)
        .def_static("liquidity_adjusted_loss", &ST::liquidity_adjusted_loss,
                     "weights"_a, "adv"_a, "nav"_a, "adv_pct"_a, "slippage_bps"_a)
        .def("config", &ST::config, nb::rv_policy::reference_internal)
        .def("custom_scenarios", &ST::custom_scenarios,
             nb::rv_policy::reference_internal);

    // ===================================================================
    // RiskMonitor
    // ===================================================================

    using RM = trade::RiskMonitor;

    nb::class_<RM::ExAnteMetrics>(m, "ExAnteMetrics")
        .def(nb::init<>())
        .def_rw("var_1d_99",            &RM::ExAnteMetrics::var_1d_99)
        .def_rw("cvar_1d_99",           &RM::ExAnteMetrics::cvar_1d_99)
        .def_rw("target_vol",           &RM::ExAnteMetrics::target_vol)
        .def_rw("ex_ante_vol",          &RM::ExAnteMetrics::ex_ante_vol)
        .def_rw("vol_gap",              &RM::ExAnteMetrics::vol_gap)
        .def_rw("portfolio_beta",       &RM::ExAnteMetrics::portfolio_beta)
        .def_rw("hhi_concentration",    &RM::ExAnteMetrics::hhi_concentration)
        .def_rw("effective_n",          &RM::ExAnteMetrics::effective_n)
        .def_rw("factor_exposures",     &RM::ExAnteMetrics::factor_exposures)
        .def_rw("max_factor_exposure",  &RM::ExAnteMetrics::max_factor_exposure);

    nb::class_<RM::ExPostMetrics>(m, "ExPostMetrics")
        .def(nb::init<>())
        .def_rw("realized_vol_20d",  &RM::ExPostMetrics::realized_vol_20d)
        .def_rw("realized_vol_60d",  &RM::ExPostMetrics::realized_vol_60d)
        .def_rw("current_drawdown",  &RM::ExPostMetrics::current_drawdown)
        .def_rw("max_drawdown",      &RM::ExPostMetrics::max_drawdown)
        .def_rw("win_rate_20d",      &RM::ExPostMetrics::win_rate_20d)
        .def_rw("daily_turnover",    &RM::ExPostMetrics::daily_turnover)
        .def_rw("avg_turnover_20d",  &RM::ExPostMetrics::avg_turnover_20d)
        .def_rw("avg_slippage_bps",  &RM::ExPostMetrics::avg_slippage_bps)
        .def_rw("tracking_error",    &RM::ExPostMetrics::tracking_error);

    nb::class_<RM::LiquidityMetrics>(m, "LiquidityMetrics")
        .def(nb::init<>())
        .def_rw("liquidation_days",        &RM::LiquidityMetrics::liquidation_days)
        .def_rw("avg_adv_participation",   &RM::LiquidityMetrics::avg_adv_participation)
        .def_rw("max_adv_participation",   &RM::LiquidityMetrics::max_adv_participation)
        .def_rw("locked_weight",           &RM::LiquidityMetrics::locked_weight)
        .def_rw("suspended_weight",        &RM::LiquidityMetrics::suspended_weight)
        .def_rw("combined_illiquid_weight", &RM::LiquidityMetrics::combined_illiquid_weight)
        .def_rw("locked_count",            &RM::LiquidityMetrics::locked_count)
        .def_rw("suspended_count",         &RM::LiquidityMetrics::suspended_count);

    nb::class_<RM::TailMetrics::RiskContributor>(m, "RiskContributor")
        .def(nb::init<>())
        .def_rw("symbol",        &RM::TailMetrics::RiskContributor::symbol)
        .def_rw("weight",        &RM::TailMetrics::RiskContributor::weight)
        .def_rw("marginal_var",  &RM::TailMetrics::RiskContributor::marginal_var)
        .def_rw("component_var", &RM::TailMetrics::RiskContributor::component_var)
        .def_rw("stress_loss",   &RM::TailMetrics::RiskContributor::stress_loss);

    nb::class_<RM::TailMetrics>(m, "TailMetrics")
        .def(nb::init<>())
        .def_rw("stress_test_worst_loss",     &RM::TailMetrics::stress_test_worst_loss)
        .def_rw("stress_test_worst_scenario", &RM::TailMetrics::stress_test_worst_scenario)
        .def_rw("stress_var_99",              &RM::TailMetrics::stress_var_99)
        .def_rw("top_contributors",           &RM::TailMetrics::top_contributors);

    nb::class_<RM::AlertThresholds>(m, "AlertThresholds")
        .def(nb::init<>())
        .def_rw("yellow_var",           &RM::AlertThresholds::yellow_var)
        .def_rw("yellow_drawdown",      &RM::AlertThresholds::yellow_drawdown)
        .def_rw("yellow_industry",      &RM::AlertThresholds::yellow_industry)
        .def_rw("orange_var",           &RM::AlertThresholds::orange_var)
        .def_rw("orange_drawdown",      &RM::AlertThresholds::orange_drawdown)
        .def_rw("orange_liq_days",      &RM::AlertThresholds::orange_liq_days)
        .def_rw("red_var",              &RM::AlertThresholds::red_var)
        .def_rw("red_drawdown",         &RM::AlertThresholds::red_drawdown)
        .def_rw("red_illiquid_weight",  &RM::AlertThresholds::red_illiquid_weight);

    nb::class_<RM::Alert>(m, "Alert")
        .def(nb::init<>())
        .def_rw("metric_name",   &RM::Alert::metric_name)
        .def_rw("level",         &RM::Alert::level)
        .def_rw("current_value", &RM::Alert::current_value)
        .def_rw("threshold",     &RM::Alert::threshold)
        .def_rw("message",       &RM::Alert::message);

    nb::class_<RM::RiskDashboard>(m, "RiskDashboard")
        .def(nb::init<>())
        .def_prop_rw("date",
            [](const RM::RiskDashboard& d) {
                return d.date.time_since_epoch().count();
            },
            [](RM::RiskDashboard& d, int v) {
                d.date = trade::Date(std::chrono::days(v));
            })
        .def_rw("ex_ante",              &RM::RiskDashboard::ex_ante)
        .def_rw("ex_post",              &RM::RiskDashboard::ex_post)
        .def_rw("liquidity",            &RM::RiskDashboard::liquidity)
        .def_rw("tail",                 &RM::RiskDashboard::tail)
        .def_rw("overall_level",        &RM::RiskDashboard::overall_level)
        .def_rw("alerts",               &RM::RiskDashboard::alerts)
        .def_rw("nav",                  &RM::RiskDashboard::nav)
        .def_rw("cash_weight",          &RM::RiskDashboard::cash_weight)
        .def_rw("num_positions",        &RM::RiskDashboard::num_positions)
        .def_rw("gross_exposure",       &RM::RiskDashboard::gross_exposure)
        .def_rw("net_exposure",         &RM::RiskDashboard::net_exposure)
        .def_rw("industry_weights",     &RM::RiskDashboard::industry_weights)
        .def_rw("max_industry_weight",  &RM::RiskDashboard::max_industry_weight)
        .def_rw("max_industry",         &RM::RiskDashboard::max_industry);

    nb::class_<RM::Config>(m, "RiskMonitorConfig")
        .def(nb::init<>())
        .def_rw("thresholds",          &RM::Config::thresholds)
        .def_rw("top_n_contributors",  &RM::Config::top_n_contributors)
        .def_rw("target_vol",          &RM::Config::target_vol);

    nb::class_<RM>(m, "RiskMonitor")
        .def(nb::init<>())
        .def(nb::init<RM::Config>(), "config"_a)
        .def("build_dashboard", [](const RM& rm,
                const Eigen::VectorXd& weights,
                const Eigen::MatrixXd& cov,
                const Eigen::MatrixXd& returns_matrix,
                const std::vector<double>& nav_series,
                const std::unordered_map<trade::Symbol, trade::Instrument>& instruments,
                const Eigen::VectorXd& adv,
                const Eigen::MatrixXd& factor_loadings,
                const std::vector<std::string>& factor_names,
                const std::vector<trade::Symbol>& symbols,
                int date) {
            return rm.build_dashboard(
                weights, cov, returns_matrix, nav_series, instruments,
                adv, factor_loadings, factor_names, symbols,
                trade::Date(std::chrono::days(date)));
        },
             "weights"_a, "cov"_a, "returns_matrix"_a,
             "nav_series"_a, "instruments"_a, "adv"_a,
             "factor_loadings"_a, "factor_names"_a, "symbols"_a, "date"_a)
        .def("compute_ex_ante", &RM::compute_ex_ante,
             "weights"_a, "cov"_a, "factor_loadings"_a,
             "factor_names"_a, "betas"_a)
        .def("compute_ex_post", &RM::compute_ex_post,
             "nav_series"_a, "daily_returns"_a,
             "daily_turnovers"_a, "daily_slippages"_a)
        .def("compute_liquidity", &RM::compute_liquidity,
             "weights"_a, "adv"_a, "symbols"_a, "instruments"_a)
        .def("evaluate_alerts", &RM::evaluate_alerts,
             "ex_ante"_a, "ex_post"_a, "liquidity"_a, "industry_weights"_a)
        .def_static("worst_alert", &RM::worst_alert, "alerts"_a)
        .def("config", &RM::config, nb::rv_policy::reference_internal)
        .def("set_config", &RM::set_config, "c"_a);

    // ===================================================================
    // RiskAttribution
    // ===================================================================

    using RA = trade::RiskAttribution;

    nb::class_<RA::StockContribution>(m, "StockContribution")
        .def(nb::init<>())
        .def_rw("symbol",              &RA::StockContribution::symbol)
        .def_rw("weight",              &RA::StockContribution::weight)
        .def_rw("marginal_var",        &RA::StockContribution::marginal_var)
        .def_rw("component_var",       &RA::StockContribution::component_var)
        .def_rw("component_var_pct",   &RA::StockContribution::component_var_pct)
        .def_rw("factor_risk_contrib", &RA::StockContribution::factor_risk_contrib)
        .def_rw("idio_risk_contrib",   &RA::StockContribution::idio_risk_contrib);

    nb::class_<RA::GroupContribution>(m, "GroupContribution")
        .def(nb::init<>())
        .def_rw("group_name",        &RA::GroupContribution::group_name)
        .def_rw("total_weight",      &RA::GroupContribution::total_weight)
        .def_rw("component_var",     &RA::GroupContribution::component_var)
        .def_rw("component_var_pct", &RA::GroupContribution::component_var_pct)
        .def_rw("num_stocks",        &RA::GroupContribution::num_stocks)
        .def_rw("members",           &RA::GroupContribution::members);

    nb::class_<RA::FactorContribution>(m, "FactorContribution")
        .def(nb::init<>())
        .def_rw("factor_name",       &RA::FactorContribution::factor_name)
        .def_rw("exposure",          &RA::FactorContribution::exposure)
        .def_rw("factor_var_contrib", &RA::FactorContribution::factor_var_contrib)
        .def_rw("factor_var_pct",    &RA::FactorContribution::factor_var_pct);

    nb::class_<RA::RiskDecomposition>(m, "RiskDecomposition")
        .def(nb::init<>())
        .def_rw("total_variance",       &RA::RiskDecomposition::total_variance)
        .def_rw("total_vol",            &RA::RiskDecomposition::total_vol)
        .def_rw("total_var_99",         &RA::RiskDecomposition::total_var_99)
        .def_rw("factor_variance",      &RA::RiskDecomposition::factor_variance)
        .def_rw("idio_variance",        &RA::RiskDecomposition::idio_variance)
        .def_rw("factor_pct",           &RA::RiskDecomposition::factor_pct)
        .def_rw("idio_pct",             &RA::RiskDecomposition::idio_pct)
        .def_rw("by_stock",             &RA::RiskDecomposition::by_stock)
        .def_rw("top5_contributors",    &RA::RiskDecomposition::top5_contributors)
        .def_rw("by_industry",          &RA::RiskDecomposition::by_industry)
        .def_rw("by_factor",            &RA::RiskDecomposition::by_factor)
        .def_rw("by_liquidity_bucket",  &RA::RiskDecomposition::by_liquidity_bucket)
        .def_rw("diversification_ratio", &RA::RiskDecomposition::diversification_ratio)
        .def_rw("effective_bets",       &RA::RiskDecomposition::effective_bets);

    nb::class_<RA::LiquidityBucketConfig>(m, "LiquidityBucketConfig")
        .def(nb::init<>())
        .def_rw("high_liquidity_min", &RA::LiquidityBucketConfig::high_liquidity_min)
        .def_rw("mid_liquidity_min",  &RA::LiquidityBucketConfig::mid_liquidity_min);

    nb::class_<RA::Config>(m, "RiskAttributionConfig")
        .def(nb::init<>())
        .def_rw("top_n",             &RA::Config::top_n)
        .def_rw("liquidity_buckets", &RA::Config::liquidity_buckets)
        .def_rw("var_confidence",    &RA::Config::var_confidence);

    nb::class_<RA>(m, "RiskAttribution")
        .def(nb::init<>())
        .def(nb::init<RA::Config>(), "config"_a)
        .def("decompose", &RA::decompose,
             "weights"_a, "covariance"_a, "factor_loadings"_a,
             "factor_cov"_a, "idio_var"_a, "symbols"_a,
             "industries"_a, "factor_names"_a,
             "adv"_a = Eigen::VectorXd{})
        .def("decompose_simple", &RA::decompose_simple,
             "weights"_a, "covariance"_a, "symbols"_a, "industries"_a)
        .def_static("marginal_var", &RA::marginal_var,
                     "weights"_a, "cov"_a, "confidence"_a = 0.99)
        .def_static("component_var", &RA::component_var,
                     "weights"_a, "cov"_a, "confidence"_a = 0.99)
        .def_static("diversification_ratio", &RA::diversification_ratio,
                     "weights"_a, "cov"_a)
        .def("config", &RA::config, nb::rv_policy::reference_internal);
}
