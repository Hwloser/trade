#include "trade/cli/commands.h"

#include "trade/common/time_utils.h"
#include "trade/provider/provider_factory.h"
#include "trade/storage/metadata_store.h"
#include "trade/storage/storage_path.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace trade::cli {
namespace {

Date today_date() {
    return std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now());
}

Side parse_side(const std::string& s) {
    if (s == "sell" || s == "SELL" || s == "Sell") return Side::kSell;
    return Side::kBuy;
}

double json_number(const nlohmann::json& j, const char* key, double def = 0.0) {
    if (!j.contains(key)) return def;
    if (j[key].is_number()) return j[key].get<double>();
    if (j[key].is_string()) {
        try {
            return std::stod(j[key].get<std::string>());
        } catch (...) {
            return def;
        }
    }
    return def;
}

int64_t json_int64(const nlohmann::json& j, const char* key, int64_t def = 0) {
    if (!j.contains(key)) return def;
    if (j[key].is_number_integer()) return j[key].get<int64_t>();
    if (j[key].is_number()) return static_cast<int64_t>(j[key].get<double>());
    if (j[key].is_string()) {
        try {
            return std::stoll(j[key].get<std::string>());
        } catch (...) {
            return def;
        }
    }
    return def;
}

void persist_snapshot(MetadataStore& metadata,
                      const AccountSnapshot& snapshot,
                      const std::string& source) {
    if (!snapshot.account.account_id.empty()) {
        metadata.upsert_broker_account(snapshot.account);
    }
    if (snapshot.cash) {
        metadata.upsert_account_cash(*snapshot.cash, source);
    }
    for (const auto& p : snapshot.positions) {
        metadata.upsert_account_position(p, source);
    }
    for (const auto& t : snapshot.trades) {
        metadata.upsert_account_trade(t, source);
    }
}

int cmd_account_bind(const CliArgs& args, MetadataStore& metadata) {
    if (args.account_id.empty()) {
        spdlog::error("--account-id required for account bind");
        return 1;
    }

    BrokerAccount account;
    account.account_id = args.account_id;
    account.broker = args.broker.empty() ? "manual" : args.broker;
    account.account_name = args.account_name.empty() ? args.account_id : args.account_name;
    account.auth_payload = args.auth_payload;
    account.is_active = true;
    metadata.upsert_broker_account(account);

    std::cout << "Bound account " << account.account_id
              << " (broker=" << account.broker << ")" << std::endl;
    return 0;
}

int cmd_account_list(const CliArgs& args, MetadataStore& metadata) {
    auto accounts = metadata.list_broker_accounts(!args.all);
    if (accounts.empty()) {
        std::cout << "No broker accounts found." << std::endl;
        return 0;
    }

    std::cout << "Accounts (" << accounts.size() << "):" << std::endl;
    for (const auto& a : accounts) {
        std::cout << "  " << a.account_id
                  << "  broker=" << a.broker
                  << "  name=" << a.account_name
                  << "  active=" << (a.is_active ? "yes" : "no")
                  << std::endl;
    }
    return 0;
}

int cmd_account_show(const CliArgs& args, MetadataStore& metadata) {
    if (args.account_id.empty()) {
        spdlog::error("--account-id required for account show");
        return 1;
    }

    auto account = metadata.get_broker_account(args.account_id);
    if (!account) {
        spdlog::error("Account {} not found", args.account_id);
        return 1;
    }

    std::cout << "=== Account ===" << std::endl;
    std::cout << "id: " << account->account_id << "\n"
              << "broker: " << account->broker << "\n"
              << "name: " << account->account_name << "\n"
              << "active: " << (account->is_active ? "yes" : "no") << "\n";

    auto cash = metadata.latest_account_cash(args.account_id);
    if (cash) {
        std::cout << "\n=== Cash (" << format_date(cash->as_of_date) << ") ===" << std::endl;
        std::cout << std::fixed << std::setprecision(2)
                  << "total_asset: " << cash->total_asset << "\n"
                  << "cash: " << cash->cash << "\n"
                  << "available_cash: " << cash->available_cash << "\n"
                  << "frozen_cash: " << cash->frozen_cash << "\n"
                  << "market_value: " << cash->market_value << std::endl;
    }

    auto positions = metadata.latest_account_positions(args.account_id);
    std::cout << "\n=== Positions (" << positions.size() << ") ===" << std::endl;
    for (const auto& p : positions) {
        std::cout << "  " << p.symbol
                  << " qty=" << p.quantity
                  << " avail=" << p.available_quantity
                  << " cost=" << std::fixed << std::setprecision(3) << p.cost_price
                  << " last=" << p.last_price
                  << " mv=" << std::setprecision(2) << p.market_value
                  << " pnl=" << p.unrealized_pnl
                  << " pnl%=" << std::setprecision(2) << (p.unrealized_pnl_ratio * 100.0) << "%"
                  << std::endl;
    }

    int limit = args.limit > 0 ? args.limit : 20;
    auto trades = metadata.list_account_trades(args.account_id, limit);
    std::cout << "\n=== Trades (latest " << trades.size() << ") ===" << std::endl;
    for (const auto& t : trades) {
        std::cout << "  " << format_date(t.trade_date)
                  << " " << t.trade_id
                  << " " << t.symbol
                  << " " << (t.side == Side::kSell ? "sell" : "buy")
                  << " qty=" << t.quantity
                  << " px=" << std::fixed << std::setprecision(3) << t.price
                  << " amt=" << std::setprecision(2) << t.amount
                  << " fee=" << t.fee
                  << std::endl;
    }
    return 0;
}

int cmd_account_import(const CliArgs& args, MetadataStore& metadata) {
    if (args.file.empty()) {
        spdlog::error("--file <snapshot.json> required for account import");
        return 1;
    }
    std::ifstream ifs(args.file);
    if (!ifs.is_open()) {
        spdlog::error("Cannot open {}", args.file);
        return 1;
    }

    nlohmann::json root;
    try {
        ifs >> root;
    } catch (const std::exception& e) {
        spdlog::error("Invalid json {}: {}", args.file, e.what());
        return 1;
    }

    AccountSnapshot snapshot;
    if (root.contains("account") && root["account"].is_object()) {
        const auto& a = root["account"];
        snapshot.account.account_id = a.value("account_id", args.account_id);
        snapshot.account.broker = a.value("broker", args.broker);
        snapshot.account.account_name = a.value("account_name", args.account_name);
        snapshot.account.auth_payload = a.value("auth_payload", args.auth_payload);
        snapshot.account.is_active = a.value("is_active", true);
    } else {
        snapshot.account.account_id = args.account_id;
        snapshot.account.broker = args.broker;
        snapshot.account.account_name = args.account_name;
        snapshot.account.auth_payload = args.auth_payload;
        snapshot.account.is_active = true;
    }

    if (snapshot.account.account_id.empty()) {
        spdlog::error("account_id missing in json/account args");
        return 1;
    }
    if (snapshot.account.broker.empty()) {
        snapshot.account.broker = "manual";
    }
    if (snapshot.account.account_name.empty()) {
        snapshot.account.account_name = snapshot.account.account_id;
    }

    Date snapshot_date = today_date();
    if (root.contains("cash") && root["cash"].is_object()) {
        const auto& c = root["cash"];
        AccountCashSnapshot cash;
        cash.account_id = snapshot.account.account_id;
        if (c.contains("as_of_date")) snapshot_date = parse_date(c.value("as_of_date", format_date(snapshot_date)));
        cash.as_of_date = snapshot_date;
        cash.total_asset = json_number(c, "total_asset", 0.0);
        cash.cash = json_number(c, "cash", 0.0);
        cash.available_cash = json_number(c, "available_cash", cash.cash);
        cash.frozen_cash = json_number(c, "frozen_cash", 0.0);
        cash.market_value = json_number(c, "market_value", 0.0);
        snapshot.cash = cash;
    }

    if (root.contains("positions") && root["positions"].is_array()) {
        for (const auto& p : root["positions"]) {
            if (!p.is_object()) continue;
            AccountPositionSnapshot pos;
            pos.account_id = snapshot.account.account_id;
            pos.as_of_date = p.contains("as_of_date")
                ? parse_date(p.value("as_of_date", format_date(snapshot_date)))
                : snapshot_date;
            pos.symbol = p.value("symbol", "");
            if (pos.symbol.empty()) continue;
            pos.quantity = json_int64(p, "quantity", 0);
            pos.available_quantity = json_int64(p, "available_quantity", pos.quantity);
            pos.cost_price = json_number(p, "cost_price", 0.0);
            pos.last_price = json_number(p, "last_price", 0.0);
            pos.market_value = json_number(p, "market_value", pos.quantity * pos.last_price);
            pos.unrealized_pnl = json_number(
                p, "unrealized_pnl", pos.market_value - pos.cost_price * static_cast<double>(pos.quantity));
            const double cost_basis = pos.cost_price * static_cast<double>(pos.quantity);
            pos.unrealized_pnl_ratio = json_number(
                p, "unrealized_pnl_ratio",
                cost_basis > 0 ? pos.unrealized_pnl / cost_basis : 0.0);
            snapshot.positions.push_back(std::move(pos));
        }
    }

    if (root.contains("trades") && root["trades"].is_array()) {
        for (const auto& t : root["trades"]) {
            if (!t.is_object()) continue;
            AccountTradeRecord tr;
            tr.account_id = snapshot.account.account_id;
            tr.trade_id = t.value("trade_id", "");
            if (tr.trade_id.empty()) continue;
            tr.trade_date = t.contains("trade_date")
                ? parse_date(t.value("trade_date", format_date(snapshot_date)))
                : snapshot_date;
            tr.symbol = t.value("symbol", "");
            tr.side = parse_side(t.value("side", "buy"));
            tr.price = json_number(t, "price", 0.0);
            tr.quantity = json_int64(t, "quantity", 0);
            tr.amount = json_number(t, "amount", tr.price * static_cast<double>(tr.quantity));
            tr.fee = json_number(t, "fee", 0.0);
            snapshot.trades.push_back(std::move(tr));
        }
    }

    persist_snapshot(metadata, snapshot, "import_json");
    std::cout << "Imported account snapshot: account=" << snapshot.account.account_id
              << " positions=" << snapshot.positions.size()
              << " trades=" << snapshot.trades.size()
              << " cash=" << (snapshot.cash ? "yes" : "no")
              << std::endl;
    return 0;
}

int cmd_account_sync(const CliArgs& args, const Config& config, MetadataStore& metadata) {
    if (args.account_id.empty()) {
        spdlog::error("--account-id required for account sync");
        return 1;
    }
    auto account = metadata.get_broker_account(args.account_id);
    if (!account) {
        spdlog::error("Account {} not found (run account --action bind first)", args.account_id);
        return 1;
    }

    const std::string provider_name = args.provider.empty() ? account->broker : args.provider;
    auto provider = ProviderFactory::create(provider_name, config);
    if (!provider->supports_account_snapshot()) {
        spdlog::error("Provider '{}' does not support account snapshot API yet", provider_name);
        return 1;
    }
    auto snap = provider->fetch_account_snapshot(account->account_id, account->auth_payload);
    if (!snap) {
        spdlog::error("Provider '{}' returned empty account snapshot", provider_name);
        return 1;
    }

    persist_snapshot(metadata, *snap, provider_name);
    std::cout << "Synced account snapshot from provider " << provider_name
              << " for " << account->account_id << std::endl;
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

