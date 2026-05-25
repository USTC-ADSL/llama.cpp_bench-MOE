#include "qnn-lib-path.hpp"

#include <algorithm>
#include <filesystem>
#include <system_error>
#include <vector>

namespace qnn {

namespace {

bool non_empty(const char * value) {
    return value != nullptr && value[0] != '\0';
}

void add_candidate(std::vector<std::string> & candidates, const char * value) {
    if (!non_empty(value)) {
        return;
    }

    std::string normalized = normalize_qnn_lib_search_path(value);
    if (normalized.empty()) {
        return;
    }

    if (std::find(candidates.begin(), candidates.end(), normalized) == candidates.end()) {
        candidates.emplace_back(std::move(normalized));
    }
}

void add_path_list(std::vector<std::string> & candidates, const char * value, char separator) {
    if (!non_empty(value)) {
        return;
    }

    const std::string paths(value);
    size_t            start = 0;
    while (start <= paths.size()) {
        const size_t end = paths.find(separator, start);
        add_candidate(candidates, paths.substr(start, end == std::string::npos ? std::string::npos : end - start).c_str());
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
}

}  // namespace

std::string normalize_qnn_lib_search_path(const std::string & path) {
    std::string normalized = path;
    while (normalized.size() > 1 && (normalized.back() == '/' || normalized.back() == '\\')) {
        normalized.pop_back();
    }
    return normalized;
}

std::string resolve_qnn_lib_search_path(
        const char * explicit_path,
        const char * env_override,
        const char * remote_lib_dir,
        const char * remote_bin_dir,
        const char * loader_path,
        const char * current_working_dir,
        const char * compile_default,
        char         loader_path_separator,
        const qnn_path_probe & path_has_qnn_libs) {
    if (non_empty(explicit_path)) {
        return normalize_qnn_lib_search_path(explicit_path);
    }

    if (non_empty(env_override)) {
        return normalize_qnn_lib_search_path(env_override);
    }

    std::vector<std::string> candidates;
    add_candidate(candidates, remote_lib_dir);
    add_candidate(candidates, remote_bin_dir);
    add_path_list(candidates, loader_path, loader_path_separator);
    add_candidate(candidates, current_working_dir);

    for (const auto & candidate : candidates) {
        if (path_has_qnn_libs && path_has_qnn_libs(candidate)) {
            return candidate;
        }
    }

    return non_empty(compile_default) ? normalize_qnn_lib_search_path(compile_default) : std::string();
}

std::string qnn_current_working_directory() {
    std::error_code ec;
    const auto      cwd = std::filesystem::current_path(ec);
    return ec ? std::string() : cwd.string();
}

}  // namespace qnn
