#pragma once

#include "trade/cli/args.h"
#include "trade/common/config.h"

namespace trade::cli {

int cmd_verify(const CliArgs& args, const Config& config);
int cmd_download(const CliArgs& args, const Config& config);
int cmd_cleanup(const CliArgs& args, const Config& config);
int cmd_info(const CliArgs& args, const Config& config);
int cmd_view(const CliArgs& args, const Config& config);
int cmd_sql(const CliArgs& args, const Config& config);

int cmd_features(const CliArgs& args, const Config& config);
int cmd_train(const CliArgs& args, const Config& config);
int cmd_predict(const CliArgs& args, const Config& config);
int cmd_risk(const CliArgs& args, const Config& config);
int cmd_backtest(const CliArgs& args, const Config& config);

int cmd_sentiment(const CliArgs& args, const Config& config);
int cmd_account(const CliArgs& args, const Config& config);

int cmd_report(const CliArgs& args, const Config& config);

} // namespace trade::cli
