#include "trade/cli/commands.h"

#include "trade/app/pipelines/train_pipeline.h"
#include "trade/cli/shared.h"
#include "trade/common/time_utils.h"
#include "trade/backtest/backtest_engine.h"
#include "trade/backtest/performance.h"
#include "trade/decision/decision_report.h"
#include "trade/features/feature_engine.h"
#include "trade/features/liquidity.h"
#include "trade/features/momentum.h"
#include "trade/features/preprocessor.h"
#include "trade/features/volatility.h"
#include "trade/regime/regime_detector.h"
#include "trade/risk/covariance.h"
#include "trade/risk/kelly.h"
#include "trade/risk/position_sizer.h"
#include "trade/risk/risk_monitor.h"
#include "trade/risk/var.h"
#include "trade/storage/metadata_store.h"
#include "trade/storage/storage_path.h"

#ifdef HAVE_LIGHTGBM
#include "trade/ml/lgbm_model.h"
#include "trade/ml/model_evaluator.h"
#include "trade/ml/model_trainer.h"
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace trade::cli {
int cmd_features(const CliArgs& args, const trade::Config& config) {
    if (args.symbol.empty()) {
        spdlog::error("--symbol required"); return 1;
    }
    auto scale = args.scale;
    std::transform(scale.begin(), scale.end(), scale.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    trade::FeatureEngine::Config feature_cfg;
    if (scale == "none" || scale == "raw") {
        feature_cfg.standardize = false;
    } else if (scale == "zscore") {
        feature_cfg.standardize = true;
        feature_cfg.standardize_mode = trade::PreprocessorConfig::StandardizeMode::kZScore;
    } else if (scale == "rank" || scale == "quantile" || scale == "quantile_rank") {
        feature_cfg.standardize = true;
        feature_cfg.standardize_mode = trade::PreprocessorConfig::StandardizeMode::kQuantileRank;
    } else {
        spdlog::error("Unsupported --scale '{}'. Use zscore|rank|none", args.scale);
        return 1;
    }

    auto bars = load_bars(args.symbol, config);
    if (bars.empty()) {
        spdlog::error("No data for {}", args.symbol); return 1;
    }
    auto [start, end] = resolve_dates(args, config.ingestion.min_start_date);
    std::vector<trade::Bar> filtered;
    for (const auto& b : bars)
        if (b.date >= start && b.date <= end) filtered.push_back(b);

    spdlog::info("Computing features for {} ({} bars, scale={})",
                 args.symbol, filtered.size(), scale);

    trade::BarSeries series;
    series.symbol = args.symbol;
    series.bars = std::move(filtered);

    std::unordered_map<trade::Symbol, trade::Instrument> instruments;
    instruments[args.symbol] = trade::Instrument{args.symbol};

    trade::FeatureEngine engine(feature_cfg);
    engine.register_calculator(std::make_unique<trade::MomentumCalculator>());
    engine.register_calculator(std::make_unique<trade::VolatilityCalculator>());
    engine.register_calculator(std::make_unique<trade::LiquidityCalculator>());

    auto features = engine.build({series}, instruments);

    std::cout << "Features: " << features.num_features() << " cols x "
              << features.num_observations() << " rows\n\nFeature names:" << std::endl;
    for (int i = 0; i < features.num_features(); ++i)
        std::cout << "  " << features.names[i] << std::endl;

    if (features.num_observations() > 0) {
        int last = features.num_observations() - 1;
        std::cout << "\nLast observation:" << std::endl;
        for (int i = 0; i < features.num_features(); ++i) {
            double v = features.matrix(last, i);
            if (!std::isnan(v))
                std::cout << "  " << std::left << std::setw(30) << features.names[i]
                         << std::fixed << std::setprecision(4) << v << std::endl;
        }
    }
    return 0;
}

// ============================================================================
// train
// ============================================================================
int cmd_train(const CliArgs& args, const trade::Config& config) {
    app::TrainRequest request;
    request.symbol = args.symbol;
    request.model = args.model;
    return app::run_train(request, config);
}

// ============================================================================
// predict
// ============================================================================
int cmd_predict(const CliArgs& args, const trade::Config& config) {
#ifdef HAVE_LIGHTGBM
    if (args.symbol.empty()) { spdlog::error("--symbol required"); return 1; }
    auto bars = load_bars(args.symbol, config);
    if (bars.empty()) { spdlog::error("No data for {}", args.symbol); return 1; }

    trade::StoragePath paths(config.data.data_root);
    trade::LGBMModel model;
    try { model.load(paths.models_dir() + "/lgbm_factor_v1.model"); }
    catch (const std::exception& e) {
        spdlog::error("Load model failed: {} (run 'train' first)", e.what()); return 1;
    }

    trade::BarSeries series{args.symbol, bars};
    std::unordered_map<trade::Symbol, trade::Instrument> instruments;
    instruments[args.symbol] = trade::Instrument{args.symbol};

    trade::FeatureEngine engine;
    engine.register_calculator(std::make_unique<trade::MomentumCalculator>());
    engine.register_calculator(std::make_unique<trade::VolatilityCalculator>());
    engine.register_calculator(std::make_unique<trade::LiquidityCalculator>());
    auto features = engine.build({series}, instruments);
    if (features.num_observations() == 0) { spdlog::error("No features"); return 1; }

    int last = features.num_observations() - 1;
    double pred = model.predict_one(features.matrix.row(last));

    std::cout << "Symbol:    " << args.symbol << "\n"
              << "Date:      " << trade::format_date(bars.back().date) << "\n"
              << "Close:     " << std::fixed << std::setprecision(2)
              << bars.back().close << "\n"
              << "Pred 5d:   " << std::showpos << std::setprecision(4)
              << (pred * 100) << "%\n"
              << "Direction: " << (pred > 0 ? "UP" : "DOWN") << std::endl;
    return 0;
#else
    spdlog::error("LightGBM not available"); return 1;
#endif
}

// ============================================================================
// risk
// ============================================================================
int cmd_risk(const CliArgs& args, const trade::Config& config) {
    if (args.symbol.empty()) { spdlog::error("--symbol required"); return 1; }
    auto bars = load_bars(args.symbol, config);
    if (bars.size() < 60) {
        spdlog::error("Need >=60 bars, got {}", bars.size()); return 1;
    }

    std::vector<double> returns;
    for (size_t i = 1; i < bars.size(); ++i)
        if (bars[i - 1].close > 0)
            returns.push_back((bars[i].close - bars[i - 1].close) / bars[i - 1].close);

    Eigen::VectorXd w(1); w(0) = 1.0;
    Eigen::MatrixXd ret_mat(returns.size(), 1);
    for (size_t i = 0; i < returns.size(); ++i) ret_mat(i, 0) = returns[i];

    trade::CovarianceEstimator cov_est;
    auto cov = cov_est.estimate(ret_mat);

    trade::VaRCalculator var_calc;
    auto var = var_calc.compute(w, cov, ret_mat);

    double mean_r = 0;
    for (double r : returns) mean_r += r;
    mean_r /= returns.size();
    double vol = std::sqrt(cov(0, 0));

    Eigen::VectorXd mu_vec(1); mu_vec(0) = mean_r;
    Eigen::VectorXd sigma_vec(1); sigma_vec(0) = vol;
    Eigen::VectorXd conf_vec(1); conf_vec(0) = 1.0;

    trade::KellyCalculator kelly;
    auto k = kelly.compute_diagnostics(mu_vec, sigma_vec, conf_vec);

    double peak = bars[0].close, max_dd = 0;
    for (const auto& b : bars) {
        peak = std::max(peak, b.close);
        max_dd = std::max(max_dd, (peak - b.close) / peak);
    }

    std::cout << "=== Risk: " << args.symbol << " ===\n"
              << "Period: " << trade::format_date(bars.front().date) << " to "
              << trade::format_date(bars.back().date) << "\n\n"
              << "Daily vol:       " << std::fixed << std::setprecision(2)
              << (vol * 100) << "%\n"
              << "Annual vol:      " << (vol * std::sqrt(252) * 100) << "%\n"
              << "VaR 99% param:   " << (var.parametric.var * 100) << "%\n"
              << "VaR 99% hist:    " << (var.historical.var * 100) << "%\n"
              << "VaR 99% MC:      " << (var.monte_carlo.var * 100) << "%\n"
              << "VaR 99% prod:    " << (var.var_1d_99 * 100) << "%\n"
              << "CVaR 99%:        " << (var.parametric.cvar * 100) << "%\n"
              << "Quarter Kelly:   " << std::setprecision(1)
              << (k.final_weights.size() > 0 ? k.final_weights(0) * 100 : 0.0) << "%\n"
              << "Max drawdown:    " << std::setprecision(2) << (max_dd * 100) << "%\n";
    return 0;
}

// ============================================================================
// backtest
// ============================================================================
int cmd_backtest(const CliArgs& args, const trade::Config& config) {
    if (args.symbol.empty()) { spdlog::error("--symbol required"); return 1; }
    auto [start, end] = resolve_dates(args, "2022-01-01");
    auto bars = load_bars(args.symbol, config);
    if (bars.size() < 120) {
        spdlog::error("Need >=120 bars, got {}", bars.size()); return 1;
    }

    spdlog::info("Backtest {} [{} to {}]", args.symbol,
                 trade::format_date(start), trade::format_date(end));

    // Simple buy-and-hold backtest from price series
    trade::BacktestResult result;
    result.strategy_name = args.strategy.empty() ? "buy_and_hold" : args.strategy;
    result.start_date = start;
    result.end_date = end;
    result.initial_capital = config.backtest.initial_capital;

    double capital = result.initial_capital, shares = 0, peak_nav = capital;
    for (const auto& b : bars) {
        if (b.date < start || b.date > end) continue;
        if (shares == 0 && b.open > 0) {
            shares = std::floor(capital / b.open / 100) * 100;
            capital -= shares * b.open;
        }
        double nav = capital + shares * b.close;
        peak_nav = std::max(peak_nav, nav);

        trade::DailyRecord rec;
        rec.date = b.date;
        rec.nav = nav;
        rec.cash = capital;
        rec.drawdown = (peak_nav - nav) / peak_nav;
        if (!result.daily_records.empty()) {
            double prev = result.daily_records.back().nav;
            rec.daily_return = prev > 0 ? (nav - prev) / prev : 0;
        }
        rec.cumulative_return = result.initial_capital > 0
            ? (nav / result.initial_capital - 1.0) : 0;
        result.daily_records.push_back(rec);
    }
    if (!result.daily_records.empty()) {
        result.final_nav = result.daily_records.back().nav;
        result.trading_days = static_cast<int>(result.daily_records.size());
    }

    trade::PerformanceCalculator calc;
    auto p = calc.compute(result);

    std::cout << "=== Backtest: " << result.strategy_name << " ===\n"
              << "Period:      " << trade::format_date(start) << " to "
              << trade::format_date(end) << " (" << result.trading_days << "d)\n"
              << "Capital:     " << std::fixed << std::setprecision(0)
              << result.initial_capital << " -> " << result.final_nav << "\n"
              << "Return:      " << std::showpos << std::setprecision(2)
              << (p.cumulative_return * 100) << "%\n"
              << "Ann return:  " << (p.annualised_return * 100) << "%\n"
              << std::noshowpos
              << "Sharpe:      " << std::setprecision(3) << p.sharpe_ratio << "\n"
              << "Sortino:     " << p.sortino_ratio << "\n"
              << "Calmar:      " << p.calmar_ratio << "\n"
              << "Max DD:      " << std::setprecision(2)
              << (p.max_drawdown * 100) << "%\n"
              << "DD duration: " << p.max_drawdown_duration << "d\n"
              << "VaR 95%:     " << (p.var_95 * 100) << "%\n"
              << "VaR 99%:     " << (p.var_99 * 100) << "%\n"
              << "Sharpe t:    " << std::setprecision(3) << p.sharpe_t_statistic << "\n"
              << "DSR:         " << p.deflated_sharpe_ratio << "\n";
    return 0;
}

} // namespace trade::cli
