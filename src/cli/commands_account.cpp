#include "trade/cli/commands.h"

#include "trade/storage/metadata_store.h"
#include "trade/storage/storage_path.h"

#include <iostream>
#include <spdlog/spdlog.h>

namespace trade::cli {
namespace {

int cmd_account_bind(const CliArgs& /*args*/, MetadataStore& /*metadata*/) {
    std::cout << "Account management not available in this version" << std::endl;
    return 0;
}

int cmd_account_list(const CliArgs& /*args*/, MetadataStore& /*metadata*/) {
    std::cout << "Account management not available in this version" << std::endl;
    return 0;
}

int cmd_account_show(const CliArgs& /*args*/, MetadataStore& /*metadata*/) {
    std::cout << "Account management not available in this version" << std::endl;
    return 0;
}

int cmd_account_import(const CliArgs& args, MetadataStore& /*metadata*/) {
    if (args.file.empty()) {
        spdlog::error("--file <snapshot.json> required for account import");
        return 1;
    }
    std::cout << "Account management not available in this version" << std::endl;
    return 0;
}

int cmd_account_sync(const CliArgs& /*args*/, const Config& /*config*/, MetadataStore& /*metadata*/) {
    std::cout << "Account management not available in this version" << std::endl;
    return 0;
}

} // namespace

int cmd_account(const CliArgs& args, const Config& config) {
    StoragePath paths(config.data.data_root);
    MetadataStore metadata(paths.metadata_db());

    std::string action = args.action;
    if (action.empty()) action = args.account_id.empty() ? "list" : "show";

    if (action == "bind") return cmd_account_bind(args, metadata);
    if (action == "list") return cmd_account_list(args, metadata);
    if (action == "show") return cmd_account_show(args, metadata);
    if (action == "import") return cmd_account_import(args, metadata);
    if (action == "sync") return cmd_account_sync(args, config, metadata);

    spdlog::error("Unknown account action: {} (supported: bind|list|show|import|sync)", action);
    return 1;
}

} // namespace trade::cli
