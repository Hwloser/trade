#pragma once

#include "trade/common/config.h"
#include "trade/common/types.h"
#include <optional>
#include <string>

namespace trade::app {

struct SentimentRequest {
    std::string symbol;
    std::string source;
    std::optional<Date> start;
    std::optional<Date> end;
};

int run_sentiment(const SentimentRequest& request, const Config& config);

} // namespace trade::app
