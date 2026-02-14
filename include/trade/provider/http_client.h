#pragma once

#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace trade {

class HttpClient {
public:
    struct Config {
        int timeout_ms = 30000;
        int retry_count = 3;
        int retry_delay_ms = 1000;
        int rate_limit_ms = 200;
        std::string user_agent =
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36";
    };

    HttpClient();
    explicit HttpClient(Config cfg);
    ~HttpClient();

    HttpClient(const HttpClient&) = delete;
    HttpClient& operator=(const HttpClient&) = delete;

    // GET request with automatic retry + rate limiting
    std::optional<std::string> get(
        const std::string& url,
        const std::unordered_map<std::string, std::string>& params = {},
        const std::unordered_map<std::string, std::string>& headers = {}) const;

private:
    Config config_;
    mutable std::chrono::steady_clock::time_point last_request_;
    mutable std::mutex mutex_;

    void rate_limit() const;
    static std::string build_url(
        const std::string& url,
        const std::unordered_map<std::string, std::string>& params);
};

} // namespace trade
