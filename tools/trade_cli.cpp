#include "trade/common/config.h"
#include "trade/common/time_utils.h"
#include "trade/common/types.h"
#include "trade/collector/collector.h"
#include "trade/features/feature_engine.h"
#include "trade/features/momentum.h"
#include "trade/features/volatility.h"
#include "trade/features/liquidity.h"
#include "trade/features/feature_monitor.h"
#include "trade/features/preprocessor.h"
#include "trade/stats/descriptive.h"
#include "trade/stats/correlation.h"
#include "trade/risk/var.h"
#include "trade/risk/kelly.h"
#include "trade/risk/covariance.h"
#include "trade/risk/position_sizer.h"
#include "trade/risk/risk_monitor.h"
#include "trade/regime/regime_detector.h"
#include "trade/backtest/backtest_engine.h"
#include "trade/backtest/performance.h"
#include "trade/sentiment/rss_source.h"
#include "trade/sentiment/rule_sentiment.h"
#include "trade/sentiment/sentiment_factor.h"
#include "trade/sentiment/symbol_linker.h"
#include "trade/sentiment/text_cleaner.h"
#include "trade/decision/decision_report.h"
#include "trade/provider/provider_factory.h"
#include "trade/storage/parquet_reader.h"
#include "trade/storage/storage_path.h"

#ifdef HAVE_LIGHTGBM
#include "trade/ml/lgbm_model.h"
#include "trade/ml/model_trainer.h"
#include "trade/ml/model_evaluator.h"
#endif

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

void print_usage() {
    std::cout << R"(
trade_cli - Quantitative Trading Decision Support System

Usage:
  trade_cli <command> [options]

Commands:
  download    Download market data
  features    Compute features for a symbol
  train       Train ML model
  predict     Generate predictions
  risk        Assess risk for a position
  backtest    Run backtest
  sentiment   Analyze sentiment from RSS feeds
  report      Generate decision report
  info        Show data info

Options:
  --config <path>       Config file path (default: config/config.yaml)
  --symbol <symbol>     Stock symbol (e.g., 600000.SH)
  --start <date>        Start date (YYYY-MM-DD)
  --end <date>          End date (YYYY-MM-DD)
  --provider <name>     Data provider (default: eastmoney)
  --model <name>        Model name (e.g., lgbm)
  --strategy <name>     Strategy name
  --source <name>       Sentiment source (rss, xueqiu, jin10)
  --output <path>       Output file path
  --verbose             Enable verbose logging
  --help                Show this help
)" << std::endl;
}

struct CliArgs {
    std::string command;
    std::string config_path = "config/config.yaml";
    std::string symbol;
    std::string start_date;
    std::string end_date;
    std::string provider = "eastmoney";
    std::string model;
    std::string strategy;
    std::string source;
    std::string output;
    bool verbose = false;
};

CliArgs parse_args(int argc, char* argv[]) {
    CliArgs args;
    if (argc < 2) return args;

    args.command = argv[1];

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) args.config_path = argv[++i];
        else if (arg == "--symbol" && i + 1 < argc) args.symbol = argv[++i];
        else if (arg == "--start" && i + 1 < argc) args.start_date = argv[++i];
        else if (arg == "--end" && i + 1 < argc) args.end_date = argv[++i];
        else if (arg == "--provider" && i + 1 < argc) args.provider = argv[++i];
        else if (arg == "--model" && i + 1 < argc) args.model = argv[++i];
        else if (arg == "--strategy" && i + 1 < argc) args.strategy = argv[++i];
        else if (arg == "--source" && i + 1 < argc) args.source = argv[++i];
        else if (arg == "--output" && i + 1 < argc) args.output = argv[++i];
        else if (arg == "--verbose") args.verbose = true;
        else if (arg == "--help") { print_usage(); std::exit(0); }
    }
    return args;
}

std::pair<trade::Date, trade::Date> resolve_dates(const CliArgs& args,
                                                    const std::string& default_start) {
    auto start = args.start_date.empty()
        ? trade::parse_date(default_start)
        : trade::parse_date(args.start_date);
    auto end = args.end_date.empty()
        ? std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now())
        : trade::parse_date(args.end_date);
    return {start, end};
}

std::vector<trade::Bar> load_bars(const std::string& symbol,
                                   const trade::Config& config) {
    trade::StoragePath paths(config.data.data_root);
    std::vector<trade::Bar> all_bars;
    auto now = std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now());
    int end_year = trade::date_year(now);
    for (int year = 2015; year <= end_year; ++year) {
        auto path = paths.curated_daily(symbol, year);
        try {
            auto bars = trade::ParquetReader::read_bars(path);
            all_bars.insert(all_bars.end(), bars.begin(), bars.end());
        } catch (...) {}
    }
    return all_bars;
}

// ============================================================================
// download
// ============================================================================
int cmd_download(const CliArgs& args, const trade::Config& config) {
    auto provider = trade::ProviderFactory::create(args.provider, config);
    if (!provider->ping()) {
        spdlog::error("Cannot connect to {} provider", args.provider);
        return 1;
    }
    trade::Collector collector(std::move(provider), config);
    auto [start, end] = resolve_dates(args, "2020-01-01");
    if (!args.symbol.empty()) {
        auto report = collector.collect_symbol(args.symbol, start, end);
        std::cout << "Downloaded " << report.total_bars << " bars for " << args.symbol
                  << " (quality: " << std::fixed << std::setprecision(1)
                  << (report.quality_score() * 100) << "%)" << std::endl;
    } else {
        collector.collect_all(start, end,
            [](const trade::Symbol& sym, int cur, int total) {
                std::cout << "\r[" << cur << "/" << total << "] " << sym
                         << "                " << std::flush;
            });
        std::cout << "\nDownload complete." << std::endl;
    }
    return 0;
}

// ============================================================================
// info
// ============================================================================
int cmd_info(const CliArgs& args, const trade::Config& config) {
    if (!args.symbol.empty()) {
        auto bars = load_bars(args.symbol, config);
        std::cout << "Symbol: " << args.symbol << "\nBars: " << bars.size() << std::endl;
        if (!bars.empty()) {
            std::cout << "Range: " << trade::format_date(bars.front().date)
                     << " to " << trade::format_date(bars.back().date) << std::endl;
            std::cout << "Last close: " << std::fixed << std::setprecision(2)
                     << bars.back().close << std::endl;
        }
    } else {
        std::cout << "Data root: " << config.data.data_root << "\nProvider: "
                 << args.provider << std::endl;
    }
    return 0;
}

// ============================================================================
// features
// ============================================================================
int cmd_features(const CliArgs& args, const trade::Config& config) {
    if (args.symbol.empty()) {
        spdlog::error("--symbol required"); return 1;
    }
    auto bars = load_bars(args.symbol, config);
    if (bars.empty()) {
        spdlog::error("No data for {}", args.symbol); return 1;
    }
    auto [start, end] = resolve_dates(args, "2020-01-01");
    std::vector<trade::Bar> filtered;
    for (const auto& b : bars)
        if (b.date >= start && b.date <= end) filtered.push_back(b);

    spdlog::info("Computing features for {} ({} bars)", args.symbol, filtered.size());

    trade::BarSeries series;
    series.symbol = args.symbol;
    series.bars = std::move(filtered);

    std::unordered_map<trade::Symbol, trade::Instrument> instruments;
    instruments[args.symbol] = trade::Instrument{args.symbol};

    trade::FeatureEngine engine;
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
#ifdef HAVE_LIGHTGBM
    if (args.symbol.empty()) { spdlog::error("--symbol required"); return 1; }
    auto bars = load_bars(args.symbol, config);
    if (bars.size() < 252) {
        spdlog::error("Need >=252 bars, got {}", bars.size()); return 1;
    }

    trade::BarSeries series{args.symbol, bars};
    std::unordered_map<trade::Symbol, trade::Instrument> instruments;
    instruments[args.symbol] = trade::Instrument{args.symbol};

    trade::FeatureEngine engine;
    engine.register_calculator(std::make_unique<trade::MomentumCalculator>());
    engine.register_calculator(std::make_unique<trade::VolatilityCalculator>());
    engine.register_calculator(std::make_unique<trade::LiquidityCalculator>());
    auto features = engine.build({series}, instruments);
    spdlog::info("Features: {} x {}", features.num_observations(), features.num_features());

    int n = features.num_observations();
    Eigen::VectorXd labels(n);
    labels.setZero();
    for (int i = 0; i + 5 < static_cast<int>(bars.size()); ++i)
        if (bars[i].close > 0)
            labels(i) = (bars[i + 5].close - bars[i].close) / bars[i].close;

    int split = static_cast<int>(n * 0.8);
    trade::LGBMModel model;
    trade::LGBMParams params;
    params.n_estimators = 300;
    params.learning_rate = 0.05;
    params.num_leaves = 31;

    auto result = model.train(
        features.matrix.topRows(split), labels.head(split), params,
        features.matrix.bottomRows(n - split), labels.tail(n - split));

    spdlog::info("Done: best_iter={} score={:.6f}", result.best_iteration, result.best_score);

    trade::StoragePath paths(config.data.data_root);
    std::string mpath = paths.models_dir() + "/lgbm_factor_v1.model";
    model.save(mpath);
    spdlog::info("Saved to {}", mpath);

    auto imp = model.feature_importance_named(features.names, "gain");
    std::cout << "\nTop features:" << std::endl;
    for (int i = 0; i < std::min(20, static_cast<int>(imp.size())); ++i)
        std::cout << "  " << std::left << std::setw(30) << imp[i].first
                 << std::fixed << std::setprecision(1) << imp[i].second << std::endl;
    return 0;
#else
    spdlog::error("LightGBM not available"); return 1;
#endif
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

// ============================================================================
// sentiment
// ============================================================================
int cmd_sentiment(const CliArgs& args, const trade::Config& config) {
    std::string src = args.source.empty() ? "rss" : args.source;
    spdlog::info("Sentiment (source: {})", src);

    if (src != "rss") {
        spdlog::error("CLI supports 'rss' source; '{}' requires API credentials", src);
        return 1;
    }

    trade::RssSource rss;
    rss.add_feed("https://rsshub.app/cls/telegraph", "CLS");
    rss.add_feed("https://rsshub.app/sina/finance", "Sina");

    auto [start, end] = resolve_dates(args, "2024-01-01");
    auto events = rss.fetch(end);
    if (events.empty()) {
        std::cout << "No events fetched (check network)." << std::endl;
        return 0;
    }

    trade::TextCleaner cleaner;
    for (auto& ev : events)
        ev.clean_text = cleaner.clean(ev.raw_text);

    trade::RuleSentiment model;
    if (!config.sentiment.dict_path.empty())
        model.load_dict(config.sentiment.dict_path);

    int pos = 0, neg = 0, neu = 0;
    std::cout << "=== Sentiment (" << events.size() << " articles) ===\n" << std::endl;
    for (const auto& ev : events) {
        auto r = model.predict(ev.clean_text.empty() ? ev.raw_text : ev.clean_text);
        const char* lbl = "neutral";
        if (r.positive > r.neutral && r.positive > r.negative) { lbl = "positive"; ++pos; }
        else if (r.negative > r.neutral && r.negative > r.positive) { lbl = "negative"; ++neg; }
        else { ++neu; }
        std::cout << "[" << lbl << "] " << ev.title << std::endl;
    }

    int total = pos + neg + neu;
    std::cout << "\nSummary: " << pos << " pos / " << neu << " neu / " << neg << " neg\n";
    if (total > 0)
        std::cout << "Net sentiment: " << std::showpos << std::fixed << std::setprecision(3)
                  << (double(pos - neg) / total) << std::endl;

    if (!args.symbol.empty()) {
        trade::SymbolLinker linker;
        linker.add_alias(args.symbol, args.symbol);
        int matched = 0;
        for (const auto& ev : events)
            if (!linker.link(ev.raw_text).empty()) ++matched;
        std::cout << "Articles linked to " << args.symbol << ": " << matched << std::endl;
    }
    return 0;
}

// ============================================================================
// report
// ============================================================================
int cmd_report(const CliArgs& args, const trade::Config& config) {
    if (args.symbol.empty()) { spdlog::error("--symbol required"); return 1; }
    auto bars = load_bars(args.symbol, config);
    if (bars.size() < 60) { spdlog::error("Need >=60 bars"); return 1; }

    std::vector<double> rets;
    for (size_t i = 1; i < bars.size(); ++i)
        if (bars[i - 1].close > 0)
            rets.push_back((bars[i].close - bars[i - 1].close) / bars[i - 1].close);

    double mr = 0, vol = 0;
    for (double r : rets) mr += r;
    mr /= rets.size();
    for (double r : rets) vol += (r - mr) * (r - mr);
    vol = std::sqrt(vol / rets.size());

    Eigen::VectorXd mu_vec(1); mu_vec(0) = mr;
    Eigen::VectorXd sigma_vec(1); sigma_vec(0) = vol;
    Eigen::VectorXd conf_vec(1); conf_vec(0) = 1.0;

    trade::KellyCalculator kelly;
    auto k = kelly.compute_diagnostics(mu_vec, sigma_vec, conf_vec);

    trade::RegimeDetector detector;
    std::vector<double> prices;
    for (const auto& b : bars) prices.push_back(b.close);
    trade::RegimeDetector::MarketBreadth breadth;
    breadth.total_stocks = 2500;
    breadth.up_stocks = 1500;
    auto regime = detector.update(prices, breadth);

    std::string regime_str = regime.regime_name();

    double full_kelly_val = k.raw_kelly.size() > 0 ? k.raw_kelly(0) : 0.0;
    double quarter_kelly_val = k.final_weights.size() > 0 ? k.final_weights(0) : 0.0;

    nlohmann::json rpt;
    rpt["ticker"]    = args.symbol;
    rpt["date"]      = trade::format_date(bars.back().date);
    rpt["close"]     = bars.back().close;
    rpt["regime"]    = regime_str;
    rpt["risk"] = {
        {"daily_vol", vol}, {"annual_vol", vol * std::sqrt(252)},
        {"mean_daily_return", mr},
        {"full_kelly", full_kelly_val}, {"quarter_kelly", quarter_kelly_val}
    };
    rpt["recommendation"] = {
        {"suggested_weight", quarter_kelly_val},
        {"confidence", 0.6}, {"regime", regime_str}
    };

    std::cout << "=== Report: " << args.symbol << " ===\n\n"
              << rpt.dump(2) << std::endl;

    if (!args.output.empty()) {
        std::ofstream ofs(args.output);
        ofs << rpt.dump(2) << std::endl;
        spdlog::info("Saved to {}", args.output);
    }
    return 0;
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    auto args = parse_args(argc, argv);
    if (args.command.empty() || args.command == "--help") { print_usage(); return 0; }

    auto console = spdlog::stdout_color_mt("console");
    spdlog::set_default_logger(console);
    spdlog::set_level(args.verbose ? spdlog::level::debug : spdlog::level::info);

    trade::Config config;
    try { config = trade::Config::load(args.config_path); }
    catch (...) {
        spdlog::debug("Config not found at {}, using defaults", args.config_path);
        config = trade::Config::defaults();
    }

    try {
        if (args.command == "download")  return cmd_download(args, config);
        if (args.command == "info")      return cmd_info(args, config);
        if (args.command == "features")  return cmd_features(args, config);
        if (args.command == "train")     return cmd_train(args, config);
        if (args.command == "predict")   return cmd_predict(args, config);
        if (args.command == "risk")      return cmd_risk(args, config);
        if (args.command == "backtest")  return cmd_backtest(args, config);
        if (args.command == "sentiment") return cmd_sentiment(args, config);
        if (args.command == "report")    return cmd_report(args, config);
        spdlog::error("Unknown command: {}", args.command);
        print_usage();
        return 1;
    } catch (const std::exception& e) {
        spdlog::error("Error: {}", e.what());
        return 1;
    }
}
