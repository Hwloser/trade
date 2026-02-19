#include "trade/sentiment/rss_source.h"
#include "trade/sentiment/text_cleaner.h"

#include <curl/curl.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <optional>
#include <sstream>
#include <thread>

namespace trade {

RssSource::RssSource() : config_{} {}

RssSource::RssSource(Config cfg) : config_(std::move(cfg)) {}

RssSource::~RssSource() = default;

std::vector<TextEvent> RssSource::fetch(Date date) {
    std::vector<TextEvent> events;
    for (const auto& feed : feeds_) {
        auto feed_events = fetch_feed(feed);
        // Filter events by date: keep only those matching the requested date
        for (auto& ev : feed_events) {
            auto ev_date = std::chrono::floor<std::chrono::days>(ev.timestamp);
            if (ev_date == date) {
                events.push_back(std::move(ev));
            }
        }
    }
    return events;
}

std::vector<TextEvent> RssSource::fetch_range(Date start, Date end) {
    std::vector<TextEvent> events;
    for (const auto& feed : feeds_) {
        auto feed_events = fetch_feed(feed);
        // Filter events by date range [start, end]
        for (auto& ev : feed_events) {
            auto ev_date = std::chrono::floor<std::chrono::days>(ev.timestamp);
            if (ev_date >= start && ev_date <= end) {
                events.push_back(std::move(ev));
            }
        }
    }
    return events;
}

bool RssSource::is_available() const {
    return !feeds_.empty();
}

void RssSource::add_feed(const std::string& url, const std::string& feed_name) {
    feeds_.push_back({url, feed_name});
}

void RssSource::add_feeds(const std::vector<FeedInfo>& feeds) {
    feeds_.insert(feeds_.end(), feeds.begin(), feeds.end());
}

void RssSource::remove_feed(const std::string& url) {
    feeds_.erase(
        std::remove_if(feeds_.begin(), feeds_.end(),
                        [&url](const FeedInfo& f) { return f.url == url; }),
        feeds_.end());
}

// ---------------------------------------------------------------------------
// Helper: extract hostname from URL for rate-limiting
// ---------------------------------------------------------------------------
static std::string extract_host(const std::string& url) {
    // Find "://" to skip the scheme
    auto scheme_end = url.find("://");
    size_t host_start = (scheme_end != std::string::npos) ? scheme_end + 3 : 0;
    // Host ends at the next '/' or end of string
    auto host_end = url.find('/', host_start);
    if (host_end == std::string::npos) host_end = url.size();
    return url.substr(host_start, host_end - host_start);
}

// ---------------------------------------------------------------------------
// Helper: find XML tag content (simple non-recursive extraction)
// Returns content between <tag> and </tag>, or empty if not found.
// ---------------------------------------------------------------------------
static std::string xml_tag_content(const std::string& xml, const std::string& tag,
                                    size_t start_pos = 0, size_t* found_end = nullptr) {
    std::string open_tag = "<" + tag + ">";
    std::string open_tag_attr = "<" + tag + " "; // tag with attributes
    std::string close_tag = "</" + tag + ">";

    size_t open_pos = xml.find(open_tag, start_pos);
    if (open_pos == std::string::npos) {
        open_pos = xml.find(open_tag_attr, start_pos);
        if (open_pos == std::string::npos) {
            if (found_end) *found_end = std::string::npos;
            return "";
        }
        // Find the end of the opening tag
        auto tag_end = xml.find('>', open_pos);
        if (tag_end == std::string::npos) {
            if (found_end) *found_end = std::string::npos;
            return "";
        }
        open_pos = tag_end + 1;
    } else {
        open_pos += open_tag.size();
    }

    size_t close_pos = xml.find(close_tag, open_pos);
    if (close_pos == std::string::npos) {
        if (found_end) *found_end = std::string::npos;
        return "";
    }

    if (found_end) *found_end = close_pos + close_tag.size();
    return xml.substr(open_pos, close_pos - open_pos);
}

// ---------------------------------------------------------------------------
// Helper: find Atom link href (link tag is self-closing in Atom)
// ---------------------------------------------------------------------------
static std::string atom_link_href(const std::string& xml, size_t start_pos = 0) {
    size_t pos = xml.find("<link", start_pos);
    if (pos == std::string::npos) return "";
    auto href_pos = xml.find("href=\"", pos);
    if (href_pos == std::string::npos) return "";
    href_pos += 6; // skip href="
    auto end_pos = xml.find('"', href_pos);
    if (end_pos == std::string::npos) return "";
    return xml.substr(href_pos, end_pos - href_pos);
}

static std::string trim(std::string s) {
    auto l = s.find_first_not_of(" \t\r\n");
    if (l == std::string::npos) return "";
    auto r = s.find_last_not_of(" \t\r\n");
    return s.substr(l, r - l + 1);
}

static Timestamp parse_feed_time(const std::string& raw) {
    auto now = std::chrono::system_clock::now();
    std::string s = trim(raw);
    if (s.empty()) return now;

    // RFC-822 with timezone suffix, e.g. "Thu, 20 Feb 2026 08:30:00 GMT"
    auto try_parse_tm = [&](const std::string& t, const char* fmt) -> std::optional<std::tm> {
        std::tm tm{};
        std::istringstream iss(t);
        iss >> std::get_time(&tm, fmt);
        if (!iss.fail()) return tm;
        return std::nullopt;
    };

    std::string no_tz = s;
    auto last_space = s.find_last_of(' ');
    if (last_space != std::string::npos) {
        std::string tz = s.substr(last_space + 1);
        bool tz_token = false;
        if (tz == "GMT" || tz == "UTC" || tz == "CST") tz_token = true;
        if (!tz.empty() && (tz[0] == '+' || tz[0] == '-')) tz_token = true;
        if (tz_token) no_tz = s.substr(0, last_space);
    }

    if (auto tm = try_parse_tm(no_tz, "%a, %d %b %Y %H:%M:%S")) {
        return std::chrono::system_clock::from_time_t(std::mktime(&*tm));
    }
    if (auto tm = try_parse_tm(no_tz, "%Y-%m-%dT%H:%M:%S")) {
        return std::chrono::system_clock::from_time_t(std::mktime(&*tm));
    }
    if (auto tm = try_parse_tm(no_tz, "%Y-%m-%d %H:%M:%S")) {
        return std::chrono::system_clock::from_time_t(std::mktime(&*tm));
    }

    return now;
}

std::vector<TextEvent> RssSource::fetch_feed(const FeedInfo& feed) {
    // Enforce rate limit for this host
    std::string host = extract_host(feed.url);
    throttle(host);

    // Fetch the XML
    std::string xml = http_get(feed.url);
    if (xml.empty()) return {};

    // Detect format: Atom feeds contain "<feed", RSS feeds contain "<rss" or "<channel"
    if (xml.find("<feed") != std::string::npos) {
        return parse_atom(xml, feed);
    } else {
        return parse_rss(xml, feed);
    }
}

std::vector<TextEvent> RssSource::parse_rss(const std::string& xml,
                                             const FeedInfo& feed) {
    std::vector<TextEvent> events;

    // Find each <item>...</item> block
    std::string item_open = "<item>";
    std::string item_open_attr = "<item ";
    std::string item_close = "</item>";

    size_t pos = 0;
    while (pos < xml.size()) {
        // Find next <item> or <item ...>
        size_t item_start = xml.find(item_open, pos);
        size_t item_start_attr = xml.find(item_open_attr, pos);

        // Pick the earlier one
        size_t actual_start = std::string::npos;
        if (item_start != std::string::npos && item_start_attr != std::string::npos) {
            actual_start = std::min(item_start, item_start_attr);
        } else if (item_start != std::string::npos) {
            actual_start = item_start;
        } else if (item_start_attr != std::string::npos) {
            actual_start = item_start_attr;
        }

        if (actual_start == std::string::npos) break;

        size_t item_end = xml.find(item_close, actual_start);
        if (item_end == std::string::npos) break;
        item_end += item_close.size();

        std::string item_xml = xml.substr(actual_start, item_end - actual_start);

        TextEvent ev;
        ev.source = "rss";
        ev.title = xml_tag_content(item_xml, "title");
        ev.url = xml_tag_content(item_xml, "link");
        ev.raw_text = xml_tag_content(item_xml, "description");
        ev.timestamp = std::chrono::system_clock::now(); // Default fallback
        auto pub_date = xml_tag_content(item_xml, "pubDate");
        if (!pub_date.empty()) {
            ev.timestamp = parse_feed_time(pub_date);
        }

        // Strip HTML from title and description
        ev.title = TextCleaner::remove_html_tags(ev.title);
        ev.raw_text = TextCleaner::remove_html_tags(ev.raw_text);

        // Compute content hash for dedup
        ev.content_hash = TextCleaner::content_hash(ev.title + ev.raw_text);

        events.push_back(std::move(ev));
        pos = item_end;
    }

    return events;
}

std::vector<TextEvent> RssSource::parse_atom(const std::string& xml,
                                              const FeedInfo& feed) {
    std::vector<TextEvent> events;

    // Find each <entry>...</entry> block
    std::string entry_open = "<entry>";
    std::string entry_open_attr = "<entry ";
    std::string entry_close = "</entry>";

    size_t pos = 0;
    while (pos < xml.size()) {
        size_t entry_start = xml.find(entry_open, pos);
        size_t entry_start_attr = xml.find(entry_open_attr, pos);

        size_t actual_start = std::string::npos;
        if (entry_start != std::string::npos && entry_start_attr != std::string::npos) {
            actual_start = std::min(entry_start, entry_start_attr);
        } else if (entry_start != std::string::npos) {
            actual_start = entry_start;
        } else if (entry_start_attr != std::string::npos) {
            actual_start = entry_start_attr;
        }

        if (actual_start == std::string::npos) break;

        size_t entry_end = xml.find(entry_close, actual_start);
        if (entry_end == std::string::npos) break;
        entry_end += entry_close.size();

        std::string entry_xml = xml.substr(actual_start, entry_end - actual_start);

        TextEvent ev;
        ev.source = "rss";
        ev.title = xml_tag_content(entry_xml, "title");
        ev.url = atom_link_href(entry_xml);

        // Atom uses <summary> or <content> for the body
        ev.raw_text = xml_tag_content(entry_xml, "summary");
        if (ev.raw_text.empty()) {
            ev.raw_text = xml_tag_content(entry_xml, "content");
        }

        ev.timestamp = std::chrono::system_clock::now();
        auto updated = xml_tag_content(entry_xml, "updated");
        if (!updated.empty()) {
            ev.timestamp = parse_feed_time(updated);
        } else {
            auto published = xml_tag_content(entry_xml, "published");
            if (!published.empty()) {
                ev.timestamp = parse_feed_time(published);
            }
        }

        // Strip HTML from title and body
        ev.title = TextCleaner::remove_html_tags(ev.title);
        ev.raw_text = TextCleaner::remove_html_tags(ev.raw_text);

        // Compute content hash for dedup
        ev.content_hash = TextCleaner::content_hash(ev.title + ev.raw_text);

        events.push_back(std::move(ev));
        pos = entry_end;
    }

    return events;
}

// libcurl write callback: appends received data to a std::string
static size_t curl_write_cb(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* buf = static_cast<std::string*>(userdata);
    size_t total = size * nmemb;
    buf->append(ptr, total);
    return total;
}

std::string RssSource::http_get(const std::string& url) {
    std::string response;

    int delay_ms = config_.retry_delay_ms;
    for (int attempt = 0; attempt <= config_.retry_count; ++attempt) {
        if (attempt > 0) {
            spdlog::debug("[RssSource] retry {}/{} for {} (delay {}ms)",
                          attempt, config_.retry_count, url, delay_ms);
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            delay_ms *= 2; // exponential backoff
        }

        response.clear();
        CURL* curl = curl_easy_init();
        if (!curl) {
            spdlog::error("[RssSource] curl_easy_init failed");
            continue;
        }

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(config_.timeout_ms));
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, 5000L);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 5L);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, config_.user_agent.c_str());
        curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, ""); // auto decompress
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);        // thread safety

        CURLcode res = curl_easy_perform(curl);
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        curl_easy_cleanup(curl);

        if (res == CURLE_OK && http_code >= 200 && http_code < 300) {
            spdlog::debug("[RssSource] fetched {} ({} bytes, HTTP {})",
                          url, response.size(), http_code);
            return response;
        }

        spdlog::warn("[RssSource] HTTP GET {} failed: curl={} http={}",
                     url, static_cast<int>(res), http_code);
    }

    spdlog::error("[RssSource] all retries exhausted for {}", url);
    return "";
}

void RssSource::throttle(const std::string& host) {
    auto now = std::chrono::steady_clock::now();
    auto it = last_request_time_.find(host);
    if (it != last_request_time_.end()) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - it->second);
        if (elapsed.count() < config_.rate_limit_ms) {
            auto sleep_duration = std::chrono::milliseconds(
                config_.rate_limit_ms - static_cast<int>(elapsed.count()));
            std::this_thread::sleep_for(sleep_duration);
        }
    }
    last_request_time_[host] = std::chrono::steady_clock::now();
}

} // namespace trade
