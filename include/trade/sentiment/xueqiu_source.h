#pragma once

// ============================================================================
// EXPERIMENTAL -- Phase 2J placeholder
// ============================================================================
// XueqiuSource collects posts and comments from Xueqiu (雪球), a popular
// Chinese financial social platform.
//
// Anti-crawl notes:
//   - Xueqiu requires a valid cookie obtained by visiting the homepage.
//   - Cookies expire periodically; the source must detect 40x responses and
//     transparently refresh the cookie.
//   - Request frequency should stay low to avoid IP bans.
//   - Consider rotating user agents / proxies for production use.
//

#include "trade/sentiment/text_source.h"
#include "trade/common/types.h"

#include <string>
#include <vector>

namespace trade {

class XueqiuSource : public ITextSource {
public:
    struct Config {
        std::string cookie;                                // session cookie
        std::string user_agent = "Mozilla/5.0";            // browser UA
        int timeout_ms = 15000;                            // HTTP timeout
        int rate_limit_ms = 3000;                          // min interval
        int retry_count = 2;
        int max_pages = 5;                                 // pages per fetch
    };

    XueqiuSource();
    explicit XueqiuSource(Config cfg);
    ~XueqiuSource() override;

    // -- ITextSource ---------------------------------------------------------
    std::string name() const override { return "xueqiu"; }
    std::vector<TextEvent> fetch(Date date) override;
    std::vector<TextEvent> fetch_range(Date start, Date end) override;
    bool is_available() const override;

    // -- Cookie management ---------------------------------------------------

    // Refresh the session cookie by visiting the Xueqiu homepage.
    // Returns true on success.
    bool refresh_cookie();

    // Set cookie manually (e.g. from config file).
    void set_cookie(const std::string& cookie);

    // -- Configuration -------------------------------------------------------
    const Config& config() const { return config_; }

private:
    // Fetch a single page of hot posts / topic timeline.
    std::vector<TextEvent> fetch_page(int page);

    // Parse JSON response into TextEvents.
    std::vector<TextEvent> parse_response(const std::string& json);

    // HTTP GET with cookie and retry logic.
    std::string http_get(const std::string& url);

    Config config_;
};

} // namespace trade
