#pragma once

#include "trade/sentiment/text_source.h"
#include "trade/common/types.h"

#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace trade {

// ============================================================================
// RssSource: RSS feed collector using libcurl + pugixml
// ============================================================================
// Collects financial news from configurable RSS feed URLs.
//
// Implementation notes:
//   - HTTP fetching via libcurl (handles HTTPS, redirect, gzip).
//   - XML parsing via pugixml (supports RSS 2.0 and Atom feeds).
//   - Encoding: auto-detects UTF-8 / GBK; converts to UTF-8 internally.
//   - Retries: configurable retry count with exponential back-off.
//   - Rate limiting: configurable minimum interval between requests to the
//     same host to avoid being blocked.
//   - Deduplication: content_hash (SHA-256 of title + raw_text) is set on
//     each TextEvent so downstream layers can discard duplicates.
//
class RssSource : public ITextSource {
public:
    // Per-feed configuration
    struct FeedInfo {
        std::string url;       // RSS/Atom feed URL
        std::string name;      // Human-readable feed name
    };

    // Global settings
    struct Config {
        int timeout_ms = 15000;                          // HTTP timeout
        int retry_count = 3;                             // max retries per feed
        int retry_delay_ms = 2000;                       // initial delay (doubles)
        int rate_limit_ms = 1000;                        // min interval per host
        std::string user_agent = "trade-rss-bot/1.0";   // HTTP User-Agent
    };

    RssSource();
    explicit RssSource(Config cfg);
    ~RssSource() override;

    // -- ITextSource ---------------------------------------------------------
    std::string name() const override { return "rss"; }
    std::vector<TextEvent> fetch(Date date) override;
    std::vector<TextEvent> fetch_range(Date start, Date end) override;
    bool is_available() const override;

    // -- Feed management -----------------------------------------------------

    // Add a feed to the list.
    void add_feed(const std::string& url, const std::string& feed_name);

    // Add multiple feeds at once.
    void add_feeds(const std::vector<FeedInfo>& feeds);

    // Remove a previously added feed by URL.
    void remove_feed(const std::string& url);

    // Get current list of feeds.
    const std::vector<FeedInfo>& feeds() const { return feeds_; }

    // -- Configuration -------------------------------------------------------
    const Config& config() const { return config_; }

private:
    // Download a single feed URL and return parsed TextEvents.
    std::vector<TextEvent> fetch_feed(const FeedInfo& feed);

    // Parse RSS 2.0 XML document into TextEvents.
    std::vector<TextEvent> parse_rss(const std::string& xml,
                                     const FeedInfo& feed);

    // Parse Atom XML document into TextEvents.
    std::vector<TextEvent> parse_atom(const std::string& xml,
                                      const FeedInfo& feed);

    // HTTP GET with retries.  Returns response body or empty on failure.
    std::string http_get(const std::string& url);

    // Enforce rate limit for the given host.
    void throttle(const std::string& host);

    Config config_;
    std::vector<FeedInfo> feeds_;

    // Last request time per host for rate limiting.
    std::unordered_map<std::string, std::chrono::steady_clock::time_point>
        last_request_time_;
};

} // namespace trade
