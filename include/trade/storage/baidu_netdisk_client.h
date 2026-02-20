#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trade {

class BaiduNetdiskClient {
public:
    struct Config {
        std::string access_token;
        std::string refresh_token;
        std::string app_key;
        std::string app_secret;
        std::string app_id;
        std::string sign_key;
        std::string root_path = "/apps/trade";
        int timeout_ms = 30000;
        int retry_count = 2;
    };

    explicit BaiduNetdiskClient(Config cfg);

    // Upload bytes to Netdisk path rooted at config.root_path.
    // remote_rel_path example: "raw/cn_a/daily/2026/600000.SH.parquet"
    bool upload_bytes(const std::string& remote_rel_path,
                      const std::vector<uint8_t>& payload);

    // Download bytes from Netdisk path rooted at config.root_path.
    bool download_bytes(const std::string& remote_rel_path,
                        std::vector<uint8_t>* payload_out);

    // Delete one or more files from Netdisk under config.root_path.
    bool delete_path(const std::string& remote_rel_path);
    bool delete_paths(const std::vector<std::string>& remote_rel_paths);

private:
    Config cfg_;

    bool upload_bytes_once(const std::string& remote_full_path,
                           const std::vector<uint8_t>& payload,
                           std::string* response_out);
    bool download_bytes_once(const std::string& remote_full_path,
                             std::vector<uint8_t>* payload_out,
                             std::string* response_out);
    bool delete_paths_once(const std::vector<std::string>& remote_full_paths,
                           std::string* response_out);
    bool refresh_access_token();
    std::string build_full_path(const std::string& remote_rel_path) const;
};

} // namespace trade
