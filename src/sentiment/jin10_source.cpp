#include "trade/sentiment/jin10_source.h"
#include "trade/sentiment/text_cleaner.h"
#include "trade/common/time_utils.h"

#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <chrono>
#include <thread>

namespace trade {

// ---------------------------------------------------------------------------
// libcurl write callback
// ---------------------------------------------------------------------------
static size_t jin10_write_cb(char* ptr, size_t size, size_t nmemb, void* ud) {
    auto* buf = static_cast<std::string*>(ud);
    size_t total = size * nmemb;
    buf->append(ptr, total);
    return total;
}

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------
Jin10Source::Jin10Source() : config_{} {}
Jin10Source::Jin10Source(Config cfg) : config_(std::move(cfg)) {}

Jin10Source::~Jin10Source() = default;

// ---------------------------------------------------------------------------
// ITextSource
// ---------------------------------------------------------------------------
std::vector<TextEvent> Jin10Source::fetch(Date date) {
    std::vector<TextEvent> all;
    for (int page = 1; page <= 10; ++page) {
        auto page_events = fetch_page(date, page);
        if (page_events.empty()) break;
        for (auto& ev : page_events) {
            all.push_back(std::move(ev));
        }
        // Respect rate limit between pages
        std::this_thread::sleep_for(
            std::chrono::milliseconds(config_.rate_limit_ms));
    }
    return all;
}

std::vector<TextEvent> Jin10Source::fetch_range(Date start, Date end) {
    std::vector<TextEvent> all;
    for (auto d = start; d <= end; d += std::chrono::days(1)) {
        auto day_events = fetch(d);
        all.insert(all.end(),
                   std::make_move_iterator(day_events.begin()),
                   std::make_move_iterator(day_events.end()));
    }
    return all;
}

bool Jin10Source::is_available() const {
    return !config_.api_key.empty();
}

void Jin10Source::set_api_key(const std::string& key) {
    config_.api_key = key;
}

// ---------------------------------------------------------------------------
// Fetch one page of flash news for a specific date
// ---------------------------------------------------------------------------
std::vector<TextEvent> Jin10Source::fetch_page(Date date, int page) {
    // Build the API URL
    // Jin10 open API: GET /flash?date=YYYY-MM-DD&page=N&page_size=M
    std::string date_str = format_date(date);
    std::string url = config_.base_url + "/flash"
                      "?date=" + date_str +
                      "&page=" + std::to_string(page) +
                      "&page_size=" + std::to_string(config_.max_items_per_request);

    std::string json_str = http_get(url);
    if (json_str.empty()) return {};

    return parse_response(json_str);
}

// ---------------------------------------------------------------------------
// Parse Jin10 JSON response
// ---------------------------------------------------------------------------
std::vector<TextEvent> Jin10Source::parse_response(const std::string& json_str) {
    std::vector<TextEvent> events;

    try {
        auto root = nlohmann::json::parse(json_str);

        // Jin10 API typically returns { "data": [ { ... }, ... ] }
        nlohmann::json items;
        if (root.contains("data") && root["data"].is_array()) {
            items = root["data"];
        } else if (root.is_array()) {
            items = root;
        } else {
            spdlog::debug("[Jin10Source] unexpected JSON structure");
            return {};
        }

        for (const auto& item : items) {
            TextEvent ev;
            ev.source = "jin10";

            // Content / text
            if (item.contains("data") && item["data"].contains("content") &&
                item["data"]["content"].is_string()) {
                ev.raw_text = item["data"]["content"].get<std::string>();
            } else if (item.contains("content") && item["content"].is_string()) {
                ev.raw_text = item["content"].get<std::string>();
            }

            // Title (Jin10 flash may not have separate title)
            if (item.contains("title") && item["title"].is_string()) {
                ev.title = item["title"].get<std::string>();
            }

            // URL / ID
            if (item.contains("id")) {
                ev.url = config_.base_url + "/flash/" +
                         std::to_string(item["id"].get<int64_t>());
            }

            // Timestamp
            if (item.contains("time") && item["time"].is_string()) {
                // Jin10 uses ISO 8601 or "YYYY-MM-DD HH:MM:SS" format
                ev.timestamp = std::chrono::system_clock::now(); // default
                // Parse the time string
                auto time_str = item["time"].get<std::string>();
                auto parsed = parse_timestamp(time_str);
                if (parsed != Timestamp{}) ev.timestamp = parsed;
            } else if (item.contains("created_at") && item["created_at"].is_number()) {
                auto secs = item["created_at"].get<int64_t>();
                ev.timestamp = Timestamp(std::chrono::seconds(secs));
            } else {
                ev.timestamp = std::chrono::system_clock::now();
            }

            // Clean
            ev.title = TextCleaner::remove_html_tags(ev.title);
            ev.raw_text = TextCleaner::remove_html_tags(ev.raw_text);
            ev.content_hash = TextCleaner::content_hash(ev.title + ev.raw_text);

            if (!ev.raw_text.empty()) {
                events.push_back(std::move(ev));
            }
        }
    } catch (const nlohmann::json::exception& e) {
        spdlog::warn("[Jin10Source] JSON parse error: {}", e.what());
    }

    return events;
}

// ---------------------------------------------------------------------------
// HTTP GET with API key auth and retry
// ---------------------------------------------------------------------------
std::string Jin10Source::http_get(const std::string& url) {
    for (int attempt = 0; attempt <= config_.retry_count; ++attempt) {
        if (attempt > 0) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(config_.rate_limit_ms * (1 << attempt)));
        }

        std::string response;
        CURL* curl = curl_easy_init();
        if (!curl) continue;

        struct curl_slist* headers = nullptr;
        std::string auth_header = "Authorization: Bearer " + config_.api_key;
        headers = curl_slist_append(headers, auth_header.c_str());
        headers = curl_slist_append(headers, "Accept: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, jin10_write_cb);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS,
                         static_cast<long>(config_.timeout_ms));
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "");
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

        CURLcode res = curl_easy_perform(curl);
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        curl_easy_cleanup(curl);
        curl_slist_free_all(headers);

        if (res == CURLE_OK && http_code >= 200 && http_code < 300) {
            return response;
        }

        if (http_code == 401 || http_code == 403) {
            spdlog::error("[Jin10Source] auth error (HTTP {}) - check API key", http_code);
            return "";  // no point retrying auth failures
        }

        spdlog::warn("[Jin10Source] HTTP GET {} failed: curl={} http={}",
                     url, static_cast<int>(res), http_code);
    }

    return "";
}

} // namespace trade
