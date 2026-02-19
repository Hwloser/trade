#include "trade/storage/baidu_netdisk_client.h"

#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <thread>

namespace trade {
namespace {

struct HttpResponse {
    long status_code = 0;
    std::string body;
    std::string error;
};

void ensure_curl_ready() {
    static const int kInit = []() {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        return 0;
    }();
    (void)kInit;
}

size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    auto* out = static_cast<std::string*>(userp);
    out->append(static_cast<char*>(contents), size * nmemb);
    return size * nmemb;
}

std::string url_encode(const std::string& value) {
    ensure_curl_ready();
    CURL* curl = curl_easy_init();
    if (!curl) return value;
    char* encoded = curl_easy_escape(curl, value.c_str(), static_cast<int>(value.size()));
    std::string out = encoded ? encoded : value;
    if (encoded) {
        curl_free(encoded);
    }
    curl_easy_cleanup(curl);
    return out;
}

HttpResponse http_post_form(const std::string& url,
                            const std::string& body,
                            int timeout_ms) {
    ensure_curl_ready();
    HttpResponse resp;

    CURL* curl = curl_easy_init();
    if (!curl) {
        resp.error = "curl_easy_init failed";
        return resp;
    }

    std::string response_body;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/x-www-form-urlencoded");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(timeout_ms));
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);

    const CURLcode code = curl_easy_perform(curl);
    if (code != CURLE_OK) {
        resp.error = curl_easy_strerror(code);
    } else {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &resp.status_code);
    }
    resp.body = std::move(response_body);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return resp;
}

HttpResponse http_post_multipart_bytes(const std::string& url,
                                       const std::vector<uint8_t>& payload,
                                       int timeout_ms) {
    ensure_curl_ready();
    HttpResponse resp;

    CURL* curl = curl_easy_init();
    if (!curl) {
        resp.error = "curl_easy_init failed";
        return resp;
    }

    std::string response_body;
    curl_mime* mime = curl_mime_init(curl);
    curl_mimepart* part = curl_mime_addpart(mime);
    curl_mime_name(part, "file");
    curl_mime_filename(part, "part.parquet");
    curl_mime_data(part,
                   reinterpret_cast<const char*>(payload.data()),
                   static_cast<size_t>(payload.size()));

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(timeout_ms));
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);

    const CURLcode code = curl_easy_perform(curl);
    if (code != CURLE_OK) {
        resp.error = curl_easy_strerror(code);
    } else {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &resp.status_code);
    }
    resp.body = std::move(response_body);

    curl_mime_free(mime);
    curl_easy_cleanup(curl);
    return resp;
}

HttpResponse http_get(const std::string& url, int timeout_ms) {
    ensure_curl_ready();
    HttpResponse resp;

    CURL* curl = curl_easy_init();
    if (!curl) {
        resp.error = "curl_easy_init failed";
        return resp;
    }

    std::string response_body;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(timeout_ms));
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);

    const CURLcode code = curl_easy_perform(curl);
    if (code != CURLE_OK) {
        resp.error = curl_easy_strerror(code);
    } else {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &resp.status_code);
    }
    resp.body = std::move(response_body);
    curl_easy_cleanup(curl);
    return resp;
}

bool is_token_expired_json(const std::string& body) {
    try {
        const auto j = nlohmann::json::parse(body);
        int code = 0;
        if (j.contains("error_code")) code = j["error_code"].get<int>();
        if (j.contains("errno")) code = j["errno"].get<int>();
        if (code == 111 || code == 110) return true;
        if (j.contains("error")) {
            const auto e = j["error"].get<std::string>();
            if (e.find("expired") != std::string::npos ||
                e.find("invalid_token") != std::string::npos) {
                return true;
            }
        }
        if (j.contains("error_msg")) {
            const auto m = j["error_msg"].get<std::string>();
            if (m.find("expired") != std::string::npos ||
                m.find("invalid") != std::string::npos) {
                return true;
            }
        }
    } catch (...) {
    }
    return false;
}

std::string normalize_rel_path(std::string path) {
    std::replace(path.begin(), path.end(), '\\', '/');
    while (!path.empty() && path.front() == '/') path.erase(path.begin());
    return path;
}

std::string normalize_root(std::string root) {
    std::replace(root.begin(), root.end(), '\\', '/');
    if (root.empty()) root = "/apps/trade";
    if (root.front() != '/') root = "/" + root;
    while (root.size() > 1 && root.back() == '/') root.pop_back();
    return root;
}

} // namespace

BaiduNetdiskClient::BaiduNetdiskClient(Config cfg) : cfg_(std::move(cfg)) {}

bool BaiduNetdiskClient::upload_bytes(const std::string& remote_rel_path,
                                      const std::vector<uint8_t>& payload) {
    if (cfg_.access_token.empty()) {
        spdlog::error("[baidu] empty access token");
        return false;
    }
    if (payload.empty()) {
        spdlog::warn("[baidu] skip upload: payload is empty for {}", remote_rel_path);
        return true;
    }

    std::string full = build_full_path(remote_rel_path);
    std::string resp_body;
    const int attempts = std::max(1, cfg_.retry_count + 1);
    for (int i = 0; i < attempts; ++i) {
        if (upload_bytes_once(full, payload, &resp_body)) {
            spdlog::info("[baidu] uploaded {} bytes to {}", payload.size(), full);
            return true;
        }

        const bool token_expired = is_token_expired_json(resp_body);
        if (token_expired && refresh_access_token()) {
            spdlog::warn("[baidu] access token refreshed, retrying upload for {}", full);
            continue;
        }

        if (i + 1 < attempts) {
            std::this_thread::sleep_for(std::chrono::milliseconds(300 * (i + 1)));
        }
    }

    spdlog::error("[baidu] upload failed after retries: {}", full);
    return false;
}

bool BaiduNetdiskClient::download_bytes(const std::string& remote_rel_path,
                                        std::vector<uint8_t>* payload_out) {
    if (!payload_out) return false;
    payload_out->clear();

    if (cfg_.access_token.empty()) {
        spdlog::error("[baidu] empty access token");
        return false;
    }

    std::string full = build_full_path(remote_rel_path);
    std::string resp_body;
    const int attempts = std::max(1, cfg_.retry_count + 1);
    for (int i = 0; i < attempts; ++i) {
        if (download_bytes_once(full, payload_out, &resp_body)) {
            spdlog::debug("[baidu] downloaded {} bytes from {}", payload_out->size(), full);
            return true;
        }

        const bool token_expired = is_token_expired_json(resp_body);
        if (token_expired && refresh_access_token()) {
            spdlog::warn("[baidu] access token refreshed, retrying download for {}", full);
            continue;
        }
        if (i + 1 < attempts) {
            std::this_thread::sleep_for(std::chrono::milliseconds(300 * (i + 1)));
        }
    }

    spdlog::warn("[baidu] download failed after retries: {}", full);
    return false;
}

bool BaiduNetdiskClient::upload_bytes_once(const std::string& remote_full_path,
                                           const std::vector<uint8_t>& payload,
                                           std::string* response_out) {
    const std::string url =
        "https://d.pcs.baidu.com/rest/2.0/pcs/file?method=upload"
        "&ondup=overwrite"
        "&path=" + url_encode(remote_full_path) +
        "&access_token=" + url_encode(cfg_.access_token);

    HttpResponse resp = http_post_multipart_bytes(url, payload, cfg_.timeout_ms);
    if (response_out) *response_out = resp.body;

    if (!resp.error.empty()) {
        spdlog::warn("[baidu] upload http error: {}", resp.error);
        return false;
    }
    if (resp.status_code != 200) {
        spdlog::warn("[baidu] upload status {} body: {}", resp.status_code, resp.body);
        return false;
    }

    try {
        auto j = nlohmann::json::parse(resp.body);
        int err = 0;
        if (j.contains("errno")) err = j["errno"].get<int>();
        if (j.contains("error_code")) err = j["error_code"].get<int>();
        if (err != 0) {
            spdlog::warn("[baidu] upload api error {}: {}", err, resp.body);
            return false;
        }
    } catch (...) {
        // Some successful responses are small JSON payloads; if parsing fails,
        // rely on HTTP status only.
    }
    return true;
}

bool BaiduNetdiskClient::download_bytes_once(const std::string& remote_full_path,
                                             std::vector<uint8_t>* payload_out,
                                             std::string* response_out) {
    const std::string url =
        "https://d.pcs.baidu.com/rest/2.0/pcs/file?method=download"
        "&path=" + url_encode(remote_full_path) +
        "&access_token=" + url_encode(cfg_.access_token);

    HttpResponse resp = http_get(url, cfg_.timeout_ms);
    if (response_out) *response_out = resp.body;

    if (!resp.error.empty()) {
        spdlog::warn("[baidu] download http error: {}", resp.error);
        return false;
    }
    if (resp.status_code != 200) {
        spdlog::warn("[baidu] download status {} body: {}", resp.status_code, resp.body);
        return false;
    }

    if (is_token_expired_json(resp.body)) {
        return false;
    }

    payload_out->assign(resp.body.begin(), resp.body.end());
    return true;
}

bool BaiduNetdiskClient::refresh_access_token() {
    if (cfg_.refresh_token.empty() || cfg_.app_key.empty() || cfg_.app_secret.empty()) {
        return false;
    }

    const std::string url = "https://openapi.baidu.com/oauth/2.0/token";
    const std::string body =
        "grant_type=refresh_token"
        "&refresh_token=" + url_encode(cfg_.refresh_token) +
        "&client_id=" + url_encode(cfg_.app_key) +
        "&client_secret=" + url_encode(cfg_.app_secret);

    HttpResponse resp = http_post_form(url, body, cfg_.timeout_ms);
    if (!resp.error.empty()) {
        spdlog::warn("[baidu] refresh token http error: {}", resp.error);
        return false;
    }
    if (resp.status_code != 200) {
        spdlog::warn("[baidu] refresh token status {} body: {}", resp.status_code, resp.body);
        return false;
    }

    try {
        auto j = nlohmann::json::parse(resp.body);
        if (!j.contains("access_token")) {
            spdlog::warn("[baidu] refresh token response missing access_token: {}", resp.body);
            return false;
        }
        cfg_.access_token = j["access_token"].get<std::string>();
        if (j.contains("refresh_token")) {
            cfg_.refresh_token = j["refresh_token"].get<std::string>();
        }
        return true;
    } catch (const std::exception& e) {
        spdlog::warn("[baidu] refresh token parse failed: {}, body={}", e.what(), resp.body);
        return false;
    }
}

std::string BaiduNetdiskClient::build_full_path(const std::string& remote_rel_path) const {
    const std::string root = normalize_root(cfg_.root_path);
    const std::string rel = normalize_rel_path(remote_rel_path);
    if (rel.empty()) return root;
    return root + "/" + rel;
}

} // namespace trade
