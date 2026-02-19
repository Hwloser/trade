#include "trade/cli/args.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace trade::cli {

void print_usage() {
    std::cout << R"(
trade_cli - Quantitative Trading Decision Support System

Usage:
  trade_cli <command> [options]

Commands:
  download    Download market data (incremental by default)
  verify      Verify local/cloud/sql data pipeline
  view        (Paused) Use sql for querying data
  sql         Open DuckDB SQL shell with data pre-loaded
  features    Compute features for a symbol
  train       Train ML model
  predict     Generate predictions
  risk        Assess risk for a position
  backtest    Run backtest
  sentiment   Analyze sentiment from RSS feeds
  report      Generate decision report
  info        (Paused) Use sql for querying metadata

Options:
  --config <path>       Config path (file or dir, default: config)
  --symbol <symbol>     Stock symbol (e.g., 600000.SH)
  --start <date>        Start date (YYYY-MM-DD)
  --end <date>          End date (YYYY-MM-DD)
  --provider <name>     Data provider (default: eastmoney)
  --refresh             Force full re-download (overwrite existing data)
  --file <path>         Parquet file path (for view command)
  --limit <n>           Max rows to display (for view command, default: all)
  --model <name>        Model name (e.g., lgbm)
  --strategy <name>     Strategy name
  --source <name>       Sentiment source (rss, xueqiu, jin10)
  --output <path>       Output file path
  --verbose             Enable verbose logging
  --help                Show this help
)" << std::endl;
}

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
        else if (arg == "--file" && i + 1 < argc) args.file = argv[++i];
        else if (arg == "--limit" && i + 1 < argc) args.limit = std::stoi(argv[++i]);
        else if (arg == "--verbose") args.verbose = true;
        else if (arg == "--refresh") args.refresh = true;
        else if (arg == "--help") {
            print_usage();
            std::exit(0);
        }
    }
    return args;
}

} // namespace trade::cli
