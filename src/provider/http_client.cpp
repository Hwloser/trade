#include "trade/provider/http_client.h"
#include <curl/curl.h>
#include <spdlog/spdlog.h>
#include <thread>

namespace trade {

namespace {

size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    auto* str = static_cast<std::string*>(userp);
    str->append(static_cast<char*>(contents), size * nmemb);
    return size * nmemb;
}

} // namespace

HttpClient::HttpClient() : HttpClient(Config{}) {}

HttpClient::HttpClient(Config cfg) : config_(std::move(cfg)) {}

HttpClient::~HttpClient() = default;

void HttpClient::rate_limit() const {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_request_).count();
    if (elapsed < config_.rate_limit_ms) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(config_.rate_limit_ms - elapsed));
    }
    last_request_ = std::chrono::steady_clock::now();
}

std::string HttpClient::build_url(
    const std::string& url,
    const std::unordered_map<std::string, std::string>& params) {
    if (params.empty()) return url;

    std::string result = url;
    char separator = (url.find('?') != std::string::npos) ? '&' : '?';
    for (const auto& [key, value] : params) {
        result += separator;
        result += key;
        result += '=';
        // Simple URL encoding for common characters
        for (char c : value) {
            if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~' ||
                c == ':' || c == '+' || c == ',') {
                result += c;
            } else {
                char buf[4];
                std::snprintf(buf, sizeof(buf), "%%%02X", static_cast<unsigned char>(c));
                result += buf;
            }
        }
        separator = '&';
    }
    return result;
}

std::optional<std::string> HttpClient::get(
    const std::string& url,
    const std::unordered_map<std::string, std::string>& params,
    const std::unordered_map<std::string, std::string>& headers) const {

    std::string full_url = build_url(url, params);

    for (int attempt = 0; attempt <= config_.retry_count; ++attempt) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            rate_limit();
        }

        CURL* curl = curl_easy_init();
        if (!curl) {
            spdlog::error("Failed to initialize curl");
            return std::nullopt;
        }

        std::string response_body;
        curl_easy_setopt(curl, CURLOPT_URL, full_url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(config_.timeout_ms));
        curl_easy_setopt(curl, CURLOPT_USERAGENT, config_.user_agent.c_str());
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
        curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "gzip, deflate");

        // Custom headers
        struct curl_slist* header_list = nullptr;
        for (const auto& [key, value] : headers) {
            std::string h = key + ": " + value;
            header_list = curl_slist_append(header_list, h.c_str());
        }
        if (header_list) {
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list);
        }

        CURLcode res = curl_easy_perform(curl);

        if (res == CURLE_OK) {
            long http_code = 0;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
            curl_slist_free_all(header_list);
            curl_easy_cleanup(curl);

            if (http_code == 200) {
                return response_body;
            }
            spdlog::warn("HTTP {} for {} (attempt {})", http_code, full_url, attempt);
        } else {
            spdlog::warn("curl error for {} (attempt {}): {}",
                         full_url, attempt, curl_easy_strerror(res));
            curl_slist_free_all(header_list);
            curl_easy_cleanup(curl);
        }

        if (attempt < config_.retry_count) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(config_.retry_delay_ms));
        }
    }

    spdlog::error("All retries exhausted for {}", full_url);
    return std::nullopt;
}

} // namespace trade
