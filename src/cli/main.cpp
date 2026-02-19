#include "trade/cli/args.h"
#include "trade/cli/commands.h"
#include "trade/common/config.h"
#include "trade/storage/parquet_reader.h"
#include "trade/storage/parquet_writer.h"

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    auto args = trade::cli::parse_args(argc, argv);
    if (args.command.empty() || args.command == "--help") {
        trade::cli::print_usage();
        return 0;
    }

    auto console = spdlog::stdout_color_mt("console");
    spdlog::set_default_logger(console);
    spdlog::set_level(args.verbose ? spdlog::level::debug : spdlog::level::info);

    trade::Config config;
    std::vector<std::string> config_search = {args.config_path};
#ifdef TRADE_SOURCE_DIR
    config_search.push_back(std::string(TRADE_SOURCE_DIR) + "/config/config.yaml");
#endif

    bool loaded = false;
    for (const auto& cp : config_search) {
        try {
            config = trade::Config::load(cp);
            loaded = true;
            break;
        } catch (...) {
        }
    }
    if (!loaded) {
        spdlog::debug("Config not found, using defaults");
        config = trade::Config::defaults();
    }

    if (!config.data.data_root.empty() && config.data.data_root[0] != '/') {
#ifdef TRADE_SOURCE_DIR
        config.data.data_root = std::string(TRADE_SOURCE_DIR) + "/" + config.data.data_root;
#endif
    }

    trade::ParquetStore::configure_runtime(config.data, config.storage);
    trade::ParquetReader::configure_runtime(config.data, config.storage);

    try {
        if (args.command == "download") return trade::cli::cmd_download(args, config);
        if (args.command == "verify") return trade::cli::cmd_verify(args, config);
        if (args.command == "view") {
            spdlog::error("Command 'view' is paused. Use 'sql' for querying data.");
            return 1;
        }
        if (args.command == "sql") return trade::cli::cmd_sql(args, config);
        if (args.command == "info") {
            spdlog::error("Command 'info' is paused. Use 'sql' for querying data.");
            return 1;
        }
        if (args.command == "features") return trade::cli::cmd_features(args, config);
        if (args.command == "train") return trade::cli::cmd_train(args, config);
        if (args.command == "predict") return trade::cli::cmd_predict(args, config);
        if (args.command == "risk") return trade::cli::cmd_risk(args, config);
        if (args.command == "backtest") return trade::cli::cmd_backtest(args, config);
        if (args.command == "sentiment") return trade::cli::cmd_sentiment(args, config);
        if (args.command == "report") return trade::cli::cmd_report(args, config);
        spdlog::error("Unknown command: {}", args.command);
        trade::cli::print_usage();
        return 1;
    } catch (const std::exception& e) {
        spdlog::error("Error: {}", e.what());
        return 1;
    }
}
