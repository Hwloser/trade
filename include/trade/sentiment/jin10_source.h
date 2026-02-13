#pragma once

// ============================================================================
// Phase 2J placeholder -- Jin10 (金十) flash news collector
// ============================================================================
// Jin10 provides real-time financial flash news (快讯) via its open API at
// open.jin10.com.  Each flash is a short paragraph (usually < 200 chars)
// with a category tag and timestamp.
//
// Authorization:
//   - Requires a valid API key (apply at open.jin10.com).
//   - The key is sent as an Authorization header or query parameter.
//

#include "trade/sentiment/text_source.h"
#include "trade/common/types.h"

#include <string>
#include <vector>

namespace trade {

class Jin10Source : public ITextSource {
public:
    struct Config {
        std::string api_key;                                // authorization key
        std::string base_url = "https://open.jin10.com";    // API base
        int timeout_ms = 10000;                             // HTTP timeout
        int rate_limit_ms = 1000;                           // min interval
        int retry_count = 3;
        int max_items_per_request = 100;                    // page size
    };

    explicit Jin10Source(Config cfg = {});
    ~Jin10Source() override;

    // -- ITextSource ---------------------------------------------------------
    std::string name() const override { return "jin10"; }
    std::vector<TextEvent> fetch(Date date) override;
    std::vector<TextEvent> fetch_range(Date start, Date end) override;
    bool is_available() const override;

    // -- Configuration -------------------------------------------------------
    void set_api_key(const std::string& key);
    const Config& config() const { return config_; }

private:
    // Fetch one page of flash news from the API.
    std::vector<TextEvent> fetch_page(Date date, int page);

    // Parse Jin10 JSON response into TextEvents.
    std::vector<TextEvent> parse_response(const std::string& json);

    // HTTP GET with Authorization header and retry logic.
    std::string http_get(const std::string& url);

    Config config_;
};

} // namespace trade
