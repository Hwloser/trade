#include "trade/storage/data_compactor.h"

#include "trade/common/time_utils.h"
#include "trade/storage/baidu_netdisk_client.h"
#include "trade/storage/metadata_store.h"
#include "trade/storage/parquet_reader.h"
#include "trade/storage/parquet_writer.h"
#include "trade/storage/storage_path.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <spdlog/spdlog.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trade {
namespace {

struct DailyCandidateFile {
    MetadataStore::DatasetFileRecord file;
    int year = 0;
    int bucket = 0;
    bool bucket_layout = false;
};

struct GroupKey {
    int year = 0;
    int bucket = 0;

    bool operator<(const GroupKey& other) const {
        if (year != other.year) return year < other.year;
        return bucket < other.bucket;
    }
};

bool cloud_mode_enabled(const Config& config) {
    return config.storage.enabled &&
        (config.storage.backend == "baidu_netdisk" || config.storage.backend == "baidu");
}

std::vector<std::string> split_path(const std::string& path) {
    std::vector<std::string> out;
    std::string token;
    for (char ch : path) {
        if (ch == '/') {
            if (!token.empty()) {
                out.push_back(token);
                token.clear();
            }
        } else {
            token.push_back(ch);
        }
    }
    if (!token.empty()) out.push_back(token);
    return out;
}

std::vector<std::string> split_dataset_id(const std::string& dataset_id) {
    std::vector<std::string> out;
    std::string token;
    for (char ch : dataset_id) {
        if (ch == '.') {
            if (!token.empty()) {
                out.push_back(token);
                token.clear();
            }
        } else {
            token.push_back(ch);
        }
    }
    if (!token.empty()) out.push_back(token);
    return out;
}

bool parse_year_part(const std::string& part, int* year_out) {
    if (!year_out) return false;
    if (part.rfind("year=", 0) == 0) {
        try {
            *year_out = std::stoi(part.substr(5));
            return true;
        } catch (...) {
            return false;
        }
    }
    try {
        *year_out = std::stoi(part);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_bucket_part(const std::string& part, int* bucket_out) {
    if (!bucket_out) return false;
    if (part.rfind("bucket=", 0) == 0) {
        try {
            *bucket_out = std::stoi(part.substr(7));
            return true;
        } catch (...) {
            return false;
        }
    }
    return false;
}

std::string build_target_rel_path(const std::string& path_prefix,
                                  int year,
                                  int bucket) {
    const std::string bucket_dir = std::string("bucket=") +
        (bucket < 10 ? "0" : "") + std::to_string(bucket);
    return (std::filesystem::path(path_prefix) /
            std::to_string(year) /
            bucket_dir /
            "part-000.parquet")
        .generic_string();
}

std::optional<DailyCandidateFile> parse_daily_candidate(
    const MetadataStore::DatasetFileRecord& file,
    const std::vector<std::string>& dataset_parts,
    int bucket_count) {
    if (dataset_parts.size() != 3) return std::nullopt;
    if (bucket_count <= 0) bucket_count = 1;

    const auto segs = split_path(file.file_path);
    if (segs.size() < 5) return std::nullopt;
    if (segs[0] != dataset_parts[0] ||
        segs[1] != dataset_parts[1] ||
        segs[2] != dataset_parts[2]) {
        return std::nullopt;
    }
    if (std::filesystem::path(file.file_path).extension() != ".parquet") return std::nullopt;

    DailyCandidateFile out;
    out.file = file;

    if (!parse_year_part(segs[3], &out.year)) return std::nullopt;

    int parsed_bucket = 0;
    if (segs.size() >= 6 && parse_bucket_part(segs[4], &parsed_bucket)) {
        out.bucket_layout = true;
        out.bucket = std::max(0, parsed_bucket);
        return out;
    }

    const std::string symbol = std::filesystem::path(file.file_path).stem().string();
    if (symbol.empty()) return std::nullopt;
    out.bucket = StoragePath::bucket_for_symbol(symbol, bucket_count);
    out.bucket_layout = false;
    return out;
}

std::string bar_key(const Bar& bar) {
    return bar.symbol + "|" + format_date(bar.date);
}

std::optional<Date> max_event_date(const std::vector<Bar>& bars) {
    if (bars.empty()) return std::nullopt;
    Date max_date = bars.front().date;
    for (const auto& bar : bars) {
        if (bar.date > max_date) {
            max_date = bar.date;
        }
    }
    return max_date;
}

std::vector<uint8_t> read_local_file_bytes(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return {};
    ifs.seekg(0, std::ios::end);
    const std::streamoff sz = ifs.tellg();
    if (sz <= 0) return {};
    ifs.seekg(0, std::ios::beg);

    std::vector<uint8_t> out(static_cast<size_t>(sz));
    ifs.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
    if (!ifs.good()) return {};
    return out;
}

std::optional<BaiduNetdiskClient> make_cloud_client(const Config& config) {
    if (!cloud_mode_enabled(config)) return std::nullopt;
    if (config.storage.baidu_access_token.empty()) return std::nullopt;
    return BaiduNetdiskClient({
        .access_token = config.storage.baidu_access_token,
        .refresh_token = config.storage.baidu_refresh_token,
        .app_key = config.storage.baidu_app_key,
        .app_secret = config.storage.baidu_app_secret,
        .app_id = config.storage.baidu_app_id,
        .sign_key = config.storage.baidu_sign_key,
        .root_path = config.storage.baidu_root,
        .timeout_ms = config.storage.baidu_timeout_ms,
        .retry_count = config.storage.baidu_retry_count,
    });
}

} // namespace

DataCompactor::DataCompactor(const Config& config) : config_(config) {}

CompactionStats DataCompactor::compact_daily_dataset(const std::string& dataset_id,
                                                     CompactionMode mode,
                                                     const CompactionOptions& options,
                                                     CompactionProgress progress) {
    CompactionStats stats;

    auto dataset_parts = split_dataset_id(dataset_id);
    if (dataset_parts.size() != 3 || dataset_parts[2] != "daily") {
        spdlog::warn("Compaction currently supports *.daily datasets only: {}", dataset_id);
        return stats;
    }

    const int bucket_count = std::max(1, options.bucket_count);
    const int small_file_rows = std::max(1, options.small_file_row_threshold);

    StoragePath paths(config_.data.data_root);
    MetadataStore metadata(paths.metadata_db());
    auto datasets = metadata.list_datasets();

    std::string path_prefix;
    for (const auto& ds : datasets) {
        if (ds.dataset_id == dataset_id) {
            path_prefix = ds.path_prefix;
            break;
        }
    }
    if (path_prefix.empty()) {
        path_prefix = (std::filesystem::path(dataset_parts[0]) /
                       dataset_parts[1] /
                       dataset_parts[2]).generic_string();
    }

    auto files = metadata.list_dataset_files(dataset_id);
    if (files.empty()) {
        return stats;
    }

    std::map<GroupKey, std::vector<DailyCandidateFile>> groups;
    for (const auto& file : files) {
        auto candidate = parse_daily_candidate(file, dataset_parts, bucket_count);
        if (!candidate.has_value()) continue;
        groups[{candidate->year, candidate->bucket}].push_back(std::move(*candidate));
        ++stats.source_files_seen;
    }
    stats.groups_total = static_cast<int>(groups.size());
    if (groups.empty()) return stats;

    auto cloud_client = make_cloud_client(config_);
    const bool can_hydrate_remote = cloud_client.has_value();
    const bool major_mode = mode == CompactionMode::kMajor;

    int group_idx = 0;
    for (auto& [key, source_group] : groups) {
        ++group_idx;
        const std::string target_rel = build_target_rel_path(path_prefix, key.year, key.bucket);
        const std::string detail = "dataset=" + dataset_id +
            " year=" + std::to_string(key.year) +
            " bucket=" + std::to_string(key.bucket);
        if (progress) {
            progress("scan", group_idx, stats.groups_total, detail);
        }

        bool should_compact = major_mode;
        if (!major_mode) {
            if (source_group.size() > 1) {
                should_compact = true;
            } else if (!source_group.empty()) {
                const auto& only = source_group.front();
                should_compact = (only.file.file_path != target_rel &&
                                  only.file.row_count <= small_file_rows);
            }
        }
        if (!should_compact) continue;

        std::sort(source_group.begin(), source_group.end(),
                  [](const DailyCandidateFile& a, const DailyCandidateFile& b) {
                      if (a.bucket_layout != b.bucket_layout) {
                          return a.bucket_layout && !b.bucket_layout;
                      }
                      return a.file.file_path < b.file.file_path;
                  });

        std::unordered_map<std::string, Bar> dedup;
        std::set<std::string> already_deleted_sources;
        int64_t group_rows_in = 0;
        for (const auto& source : source_group) {
            const std::filesystem::path abs_path =
                std::filesystem::path(config_.data.data_root) / source.file.file_path;
            const bool exists_local = std::filesystem::exists(abs_path);
            if (!major_mode && !exists_local) {
                continue;
            }
            if (major_mode && !exists_local && !can_hydrate_remote) {
                if (!options.dry_run) {
                    metadata.delete_dataset_file(dataset_id,
                                                 source.file.file_path,
                                                 "major_compaction_missing_source");
                    ++stats.source_files_deleted;
                    already_deleted_sources.insert(source.file.file_path);
                }
                continue;
            }

            std::vector<Bar> bars;
            try {
                bars = ParquetReader::read_bars(abs_path.string());
            } catch (const std::exception& e) {
                spdlog::warn("Failed to read {} during compaction: {}",
                             abs_path.string(), e.what());
                continue;
            }
            group_rows_in += static_cast<int64_t>(bars.size());
            for (auto& bar : bars) {
                dedup[bar_key(bar)] = std::move(bar);
            }
        }

        if (dedup.empty()) continue;

        std::vector<Bar> merged;
        merged.reserve(dedup.size());
        for (auto& [_, bar] : dedup) {
            merged.push_back(std::move(bar));
        }
        std::sort(merged.begin(), merged.end(),
                  [](const Bar& a, const Bar& b) {
                      if (a.date != b.date) return a.date < b.date;
                      return a.symbol < b.symbol;
                  });

        stats.rows_in += group_rows_in;
        stats.rows_out += static_cast<int64_t>(merged.size());
        if (group_rows_in > static_cast<int64_t>(merged.size())) {
            stats.rows_deduped += group_rows_in - static_cast<int64_t>(merged.size());
        }

        if (progress) {
            progress("write", group_idx, stats.groups_total, detail);
        }

        if (!options.dry_run) {
            const std::filesystem::path target_abs =
                std::filesystem::path(config_.data.data_root) / target_rel;
            ParquetStore::write_bars(target_abs.string(),
                                     merged,
                                     ParquetStore::MergeMode::kReplace,
                                     max_event_date(merged));
            if (std::filesystem::exists(target_abs)) {
                ++stats.target_files_written;
            } else {
                spdlog::warn("Compaction target write missing: {}", target_abs.string());
                continue;
            }

            if (major_mode && cloud_client.has_value()) {
                auto bytes = read_local_file_bytes(target_abs.string());
                if (!bytes.empty() && cloud_client->upload_bytes(target_rel, bytes)) {
                    ++stats.target_files_uploaded;
                } else {
                    spdlog::warn("Major compaction upload failed: {}", target_rel);
                }
            }

            std::vector<std::string> remote_deleted;
            for (const auto& source : source_group) {
                if (source.file.file_path == target_rel) continue;
                if (already_deleted_sources.find(source.file.file_path) !=
                    already_deleted_sources.end()) {
                    continue;
                }

                const std::filesystem::path source_abs =
                    std::filesystem::path(config_.data.data_root) / source.file.file_path;
                std::error_code ec;
                std::filesystem::remove(source_abs, ec);
                if (ec) {
                    spdlog::debug("Compaction local remove skipped {}: {}",
                                  source_abs.string(), ec.message());
                }

                metadata.delete_dataset_file(dataset_id,
                                             source.file.file_path,
                                             major_mode ? "major_compaction_merge"
                                                        : "minor_compaction_merge");
                ++stats.source_files_deleted;

                if (major_mode) {
                    remote_deleted.push_back(source.file.file_path);
                }
            }

            if (major_mode && cloud_client.has_value() && !remote_deleted.empty()) {
                if (progress) {
                    progress("remote_delete", group_idx, stats.groups_total, detail);
                }
                cloud_client->delete_paths(remote_deleted);
            }
        }

        ++stats.groups_compacted;
    }

    if (major_mode && !options.dry_run) {
        if (progress) {
            progress("tombstone", 1, 1, "purge tombstones");
        }
        stats.tombstones_purged = metadata.purge_dataset_tombstones(
            dataset_id,
            std::max(0, options.tombstone_retention_days));
    }

    return stats;
}

} // namespace trade
