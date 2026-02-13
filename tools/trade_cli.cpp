#include "trade/common/config.h"
#include "trade/common/time_utils.h"
#include "trade/common/types.h"
#include "trade/collector/collector.h"
#include "trade/provider/provider_factory.h"
#include "trade/storage/parquet_reader.h"
#include "trade/storage/storage_path.h"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
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
  risk        Assess risk
  backtest    Run backtest
  sentiment   Analyze sentiment
  report      Generate decision report
  info        Show data info

Options:
  --config <path>       Config file path (default: config/config.yaml)
  --symbol <symbol>     Stock symbol (e.g., 600000.SH)
  --start <date>        Start date (YYYY-MM-DD)
  --end <date>          End date (YYYY-MM-DD)
  --provider <name>     Data provider (default: akshare)
  --model <name>        Model name (e.g., lgbm)
  --strategy <name>     Strategy name
  --source <name>       Sentiment source (rss, xueqiu, jin10)
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
    std::string provider = "akshare";
    std::string model;
    std::string strategy;
    std::string source;
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
        else if (arg == "--verbose") args.verbose = true;
        else if (arg == "--help") { print_usage(); std::exit(0); }
    }
    return args;
}

int cmd_download(const CliArgs& args, const trade::Config& config) {
    auto provider = trade::ProviderFactory::create(args.provider, config);

    if (!provider->ping()) {
        spdlog::error("Cannot connect to {} provider at {}",
                     args.provider, config.akshare.base_url);
        spdlog::info("Make sure the AkShare HTTP server is running:");
        spdlog::info("  pip install aktools && python -m aktools");
        return 1;
    }

    trade::Collector collector(std::move(provider), config);

    if (!args.symbol.empty()) {
        auto start = args.start_date.empty()
            ? trade::parse_date("2020-01-01")
            : trade::parse_date(args.start_date);
        auto end = args.end_date.empty()
            ? std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now())
            : trade::parse_date(args.end_date);

        auto report = collector.collect_symbol(args.symbol, start, end);
        std::cout << "Downloaded " << report.total_bars << " bars for " << args.symbol
                  << " (quality: " << (report.quality_score() * 100) << "%)" << std::endl;
    } else {
        auto start = args.start_date.empty()
            ? trade::parse_date("2020-01-01")
            : trade::parse_date(args.start_date);
        auto end = args.end_date.empty()
            ? std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now())
            : trade::parse_date(args.end_date);

        collector.collect_all(start, end,
            [](const trade::Symbol& sym, int current, int total) {
                std::cout << "\r[" << current << "/" << total << "] " << sym
                         << "                " << std::flush;
            });
        std::cout << std::endl << "Download complete." << std::endl;
    }
    return 0;
}

int cmd_info(const CliArgs& args, const trade::Config& config) {
    trade::StoragePath paths(config.data.data_root);

    if (!args.symbol.empty()) {
        // Show info for a specific symbol
        auto today = std::chrono::floor<std::chrono::days>(
            std::chrono::system_clock::now());
        int year = trade::date_year(today);
        auto path = paths.curated_daily(args.symbol, year);

        try {
            auto count = trade::ParquetReader::row_count(path);
            std::cout << "Symbol: " << args.symbol << std::endl;
            std::cout << "Path: " << path << std::endl;
            std::cout << "Rows: " << count << std::endl;

            auto bars = trade::ParquetReader::read_bars(path);
            if (!bars.empty()) {
                std::cout << "Date range: " << trade::format_date(bars.front().date)
                         << " to " << trade::format_date(bars.back().date) << std::endl;
                std::cout << "Last close: " << bars.back().close << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "No data found for " << args.symbol << ": " << e.what() << std::endl;
        }
    } else {
        std::cout << "Data root: " << config.data.data_root << std::endl;
        std::cout << "Provider: " << args.provider << std::endl;
    }
    return 0;
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    auto args = parse_args(argc, argv);

    if (args.command.empty() || args.command == "--help") {
        print_usage();
        return 0;
    }

    // Setup logging
    auto console = spdlog::stdout_color_mt("console");
    spdlog::set_default_logger(console);
    spdlog::set_level(args.verbose ? spdlog::level::debug : spdlog::level::info);

    // Load config
    auto config = trade::Config::load(args.config_path);

    try {
        if (args.command == "download") return cmd_download(args, config);
        if (args.command == "info") return cmd_info(args, config);
        if (args.command == "features") {
            spdlog::info("Feature computation: TODO");
            return 0;
        }
        if (args.command == "train") {
            spdlog::info("Model training: TODO");
            return 0;
        }
        if (args.command == "predict") {
            spdlog::info("Prediction: TODO");
            return 0;
        }
        if (args.command == "risk") {
            spdlog::info("Risk assessment: TODO");
            return 0;
        }
        if (args.command == "backtest") {
            spdlog::info("Backtesting: TODO");
            return 0;
        }
        if (args.command == "sentiment") {
            spdlog::info("Sentiment analysis: TODO");
            return 0;
        }
        if (args.command == "report") {
            spdlog::info("Decision report: TODO");
            return 0;
        }

        spdlog::error("Unknown command: {}", args.command);
        print_usage();
        return 1;
    } catch (const std::exception& e) {
        spdlog::error("Error: {}", e.what());
        return 1;
    }
}
