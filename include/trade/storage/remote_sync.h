#pragma once

#include <string>

namespace trade {

class RemoteSync {
public:
    static bool rclone_exists(const std::string& rclone_bin = "rclone");

    // Sync local directory to Baidu Netdisk remote via rclone.
    // remote_spec example: "baidu_trade" (configured by `rclone config`).
    static bool sync_to_baidu(const std::string& local_dir,
                              const std::string& rclone_bin,
                              const std::string& remote_spec,
                              const std::string& remote_path,
                              bool dry_run = false);
};

} // namespace trade
