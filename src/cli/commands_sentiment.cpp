#include "trade/cli/commands.h"

#include "trade/app/pipelines/sentiment_pipeline.h"
#include "trade/common/time_utils.h"

namespace trade::cli {

int cmd_sentiment(const CliArgs& args, const trade::Config& config) {
    app::SentimentRequest request;
    request.symbol = args.symbol;
    request.source = args.source;
    if (!args.start_date.empty()) {
        request.start = parse_date(args.start_date);
    }
    if (!args.end_date.empty()) {
        request.end = parse_date(args.end_date);
    }
    return app::run_sentiment(request, config);
}

} // namespace trade::cli
