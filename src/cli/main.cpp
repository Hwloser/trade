#include "trade/cli/args.h"
#include "trade/cli/commands.h"
#include "trade/common/config.h"
#include "trade/storage/parquet_reader.h"
#include "trade/storage/parquet_writer.h"

#include <algorithm>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

namespace {

using CommandHandler = int (*)(const trade::cli::CliArgs&, const trade::Config&);

struct CommandEntry {
    const char* name;
    CommandHandler handler;
    bool paused;
    const char* paused_message;
};

constexpr CommandEntry kCommandRegistry[] = {
    {"collect", trade::cli::cmd_collect, false, nullptr},
    {"silver", trade::cli::cmd_silver, false, nullptr},
    {"cleanup", trade::cli::cmd_cleanup, false, nullptr},
    {"verify", trade::cli::cmd_verify, false, nullptr},
    {"view", nullptr, true, "Command 'view' is paused. Use 'sql' for querying data."},
    {"sql", trade::cli::cmd_sql, false, nullptr},
    {"info", nullptr, true, "Command 'info' is paused. Use 'sql' for querying data."},
    {"features", trade::cli::cmd_features, false, nullptr},
    {"train", trade::cli::cmd_train, false, nullptr},
    {"predict", trade::cli::cmd_predict, false, nullptr},
    {"risk", trade::cli::cmd_risk, false, nullptr},
    {"backtest", trade::cli::cmd_backtest, false, nullptr},
    {"sentiment", trade::cli::cmd_sentiment, false, nullptr},
    {"account", trade::cli::cmd_account, false, nullptr},
    {"report", trade::cli::cmd_report, false, nullptr},
};

const CommandEntry* find_command(const std::string& name) {
    const auto it = std::find_if(std::begin(kCommandRegistry), std::end(kCommandRegistry),
                                 [&](const CommandEntry& entry) { return name == entry.name; });
    if (it == std::end(kCommandRegistry)) return nullptr;
    return it;
}

} // namespace

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
        const CommandEntry* command = find_command(args.command);
        if (command) {
            if (command->paused) {
                spdlog::error("{}", command->paused_message ? command->paused_message : "Command is paused.");
                return 1;
            }
            if (!command->handler) {
                spdlog::error("Command '{}' has no handler bound", args.command);
                return 1;
            }
            return command->handler(args, config);
        }
        spdlog::error("Unknown command: {}", args.command);
        trade::cli::print_usage();
        return 1;
    } catch (const std::exception& e) {
        spdlog::error("Error: {}", e.what());
        return 1;
    }
}
