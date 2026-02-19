#pragma once

#include <string>

namespace trade::cli {

struct CliArgs {
    std::string command;
    std::string config_path = "config";
    std::string symbol;
    std::string start_date;
    std::string end_date;
    std::string provider = "eastmoney";
    std::string model;
    std::string strategy;
    std::string source;
    std::string output;
    std::string file;
    std::string action;
    std::string account_id;
    std::string broker;
    std::string account_name;
    std::string auth_payload;
    bool verbose = false;
    bool refresh = false;
    bool all = false;
    int limit = 0;
};

void print_usage();
CliArgs parse_args(int argc, char* argv[]);

} // namespace trade::cli
