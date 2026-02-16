#include "trade/storage/remote_sync.h"

#include <cstdlib>
#include <iostream>

namespace trade {

bool RemoteSync::rclone_exists(const std::string& rclone_bin) {
    std::string cmd = "command -v \"" + rclone_bin + "\" > /dev/null 2>&1";
    return std::system(cmd.c_str()) == 0;
}

bool RemoteSync::sync_to_baidu(const std::string& local_dir,
                               const std::string& rclone_bin,
                               const std::string& remote_spec,
                               const std::string& remote_path,
                               bool dry_run) {
    if (remote_spec.empty()) {
        std::cerr << "[offload] storage.baidu_remote is empty in config" << std::endl;
        return false;
    }

    std::string full_remote = remote_spec + ":" + remote_path;
    std::string cmd =
        "\"" + rclone_bin + "\" sync "
        "\"" + local_dir + "\" "
        "\"" + full_remote + "\" "
        "--create-empty-src-dirs --transfers 8 --checkers 8";

    if (dry_run) {
        cmd += " --dry-run";
    }

    std::cout << "[offload] running: " << cmd << std::endl;
    return std::system(cmd.c_str()) == 0;
}

} // namespace trade
