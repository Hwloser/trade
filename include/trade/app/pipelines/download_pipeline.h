#pragma once

#include "trade/common/config.h"
#include "trade/common/types.h"
#include <optional>
#include <string>

namespace trade::app {

struct DownloadRequest {
    std::string symbol; // single symbol or comma-separated symbol list
    std::optional<Date> start;
    std::optional<Date> end;
    std::string provider = "eastmoney";
    bool refresh = false;
    bool use_checkpoint = false; // when true, resume from stream checkpoint (Flink-like)
};

int run_download(const DownloadRequest& request, const Config& config);

} // namespace trade::app
