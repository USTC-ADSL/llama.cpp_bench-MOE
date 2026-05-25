#pragma once

#include <functional>
#include <string>

namespace qnn {

using qnn_path_probe = std::function<bool(const std::string & path)>;

std::string normalize_qnn_lib_search_path(const std::string & path);

std::string resolve_qnn_lib_search_path(
        const char * explicit_path,
        const char * env_override,
        const char * remote_lib_dir,
        const char * remote_bin_dir,
        const char * loader_path,
        const char * current_working_dir,
        const char * compile_default,
        char         loader_path_separator,
        const qnn_path_probe & path_has_qnn_libs);

std::string qnn_current_working_directory();

}  // namespace qnn
