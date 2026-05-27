#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="native"
BUILD_DIR="${ROOT_DIR}/build-clean-min"
BUILD_TYPE="Release"
ANDROID_ABI="arm64-v8a"
ANDROID_PLATFORM="android-31"
ANDROID_NDK="${ANDROID_NDK_ROOT:-}"
OPENCL_SDK="${OPENCL_SDK_ROOT:-}"
QNN_SDK="${QNN_SDK_PATH:-${QNN_SDK_ROOT:-}}"
HEXAGON_SDK="${HEXAGON_SDK_ROOT:-${ROOT_DIR}/hexagon-sdk}"
HEXAGON_TOOLS="${HEXAGON_TOOLS_ROOT:-${HEXAGON_SDK}/tools/HEXAGON_Tools/19.0.04}"
VULKAN_GLSLC=""
VULKAN_INCLUDE=""

CLEAN=0
CONFIGURE_ONLY=0
RUN_TESTS=0
BUILD_TESTS="ON"
BUILD_TOOLS="ON"
BUILD_EXAMPLES="ON"

OPENCL=""
QNN=""
HEXAGON=""
VULKAN=""
QNN_CPU_BACKEND="ON"
QNN_HEXAGON_BACKEND="OFF"
PROFILING="OFF"

TARGETS=()

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

info() {
    printf '[build] %s\n' "$*"
}

usage() {
    cat <<'EOF'
Usage:
  scripts/build.sh [options]

Common options:
  --native                         Configure a host build (default).
  --android-snapdragon             Configure an Android arm64 Snapdragon build.
  --build-dir <dir>                Build directory. Default: build-clean-min.
  --type <Release|Debug|RelWithDebInfo|MinSizeRel>
  --clean                          Remove the build directory before configuring.
  --configure-only                 Configure but do not build.
  --run-tests                      Run ctest after a native build.
  --target <name>                  Build one target. Repeat for multiple targets.
  --no-tests / --tests             Disable or enable LLAMA_BUILD_TESTS.
  --no-tools / --tools             Disable or enable LLAMA_BUILD_TOOLS.
  --no-examples / --examples       Disable or enable LLAMA_BUILD_EXAMPLES.

Backend options:
  --with-opencl / --without-opencl
  --with-qnn / --without-qnn
  --qnn-sdk <path>                 QNN or QAIRT SDK root. Also reads QNN_SDK_PATH/QNN_SDK_ROOT.
  --with-qnn-cpu-backend / --without-qnn-cpu-backend
  --with-qnn-hexagon-backend / --without-qnn-hexagon-backend
  --with-hexagon / --without-hexagon
  --hexagon-sdk <path>
  --hexagon-tools <path>
  --with-vulkan / --without-vulkan
  --vulkan-glslc <path>
  --vulkan-include <path>
  --with-profiling / --without-profiling

Android/Snapdragon options:
  --android-ndk <path>             Android NDK root. Also reads ANDROID_NDK_ROOT.
  --android-abi <abi>              Default: arm64-v8a.
  --android-platform <api>         Default: android-31.
  --opencl-sdk <path>              Optional OpenCL SDK prefix. Also reads OPENCL_SDK_ROOT.

Examples:
  scripts/build.sh --native --clean --run-tests
  scripts/build.sh --native --build-dir build-opencl --with-opencl
  scripts/build.sh --android-snapdragon --build-dir build-android-opencl --with-opencl
  scripts/build.sh --android-snapdragon --build-dir build-qnn-opencl \
      --with-opencl --with-qnn --qnn-sdk /path/to/qairt
EOF
}

require_arg() {
    [ $# -ge 2 ] || die "$1 requires an argument"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --native)
            MODE="native"
            shift
            ;;
        --android-snapdragon)
            MODE="android-snapdragon"
            shift
            ;;
        --build-dir)
            require_arg "$@"
            BUILD_DIR="$2"
            shift 2
            ;;
        --type)
            require_arg "$@"
            BUILD_TYPE="$2"
            shift 2
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        --configure-only)
            CONFIGURE_ONLY=1
            shift
            ;;
        --run-tests)
            RUN_TESTS=1
            shift
            ;;
        --target)
            require_arg "$@"
            TARGETS+=("$2")
            shift 2
            ;;
        --tests)
            BUILD_TESTS="ON"
            shift
            ;;
        --no-tests)
            BUILD_TESTS="OFF"
            shift
            ;;
        --tools)
            BUILD_TOOLS="ON"
            shift
            ;;
        --no-tools)
            BUILD_TOOLS="OFF"
            shift
            ;;
        --examples)
            BUILD_EXAMPLES="ON"
            shift
            ;;
        --no-examples)
            BUILD_EXAMPLES="OFF"
            shift
            ;;
        --with-opencl)
            OPENCL="ON"
            shift
            ;;
        --without-opencl|--no-opencl)
            OPENCL="OFF"
            shift
            ;;
        --with-qnn)
            QNN="ON"
            shift
            ;;
        --without-qnn|--no-qnn)
            QNN="OFF"
            shift
            ;;
        --qnn-sdk)
            require_arg "$@"
            QNN_SDK="$2"
            shift 2
            ;;
        --with-qnn-cpu-backend)
            QNN_CPU_BACKEND="ON"
            shift
            ;;
        --without-qnn-cpu-backend|--no-qnn-cpu-backend)
            QNN_CPU_BACKEND="OFF"
            shift
            ;;
        --with-qnn-hexagon-backend)
            QNN="ON"
            QNN_HEXAGON_BACKEND="ON"
            shift
            ;;
        --without-qnn-hexagon-backend|--no-qnn-hexagon-backend)
            QNN_HEXAGON_BACKEND="OFF"
            shift
            ;;
        --with-hexagon)
            HEXAGON="ON"
            shift
            ;;
        --without-hexagon|--no-hexagon)
            HEXAGON="OFF"
            shift
            ;;
        --hexagon-sdk)
            require_arg "$@"
            HEXAGON_SDK="$2"
            shift 2
            ;;
        --hexagon-tools)
            require_arg "$@"
            HEXAGON_TOOLS="$2"
            shift 2
            ;;
        --with-vulkan)
            VULKAN="ON"
            shift
            ;;
        --without-vulkan|--no-vulkan)
            VULKAN="OFF"
            shift
            ;;
        --vulkan-glslc)
            require_arg "$@"
            VULKAN_GLSLC="$2"
            shift 2
            ;;
        --vulkan-include)
            require_arg "$@"
            VULKAN_INCLUDE="$2"
            shift 2
            ;;
        --with-profiling)
            PROFILING="ON"
            shift
            ;;
        --without-profiling|--no-profiling)
            PROFILING="OFF"
            shift
            ;;
        --android-ndk)
            require_arg "$@"
            ANDROID_NDK="$2"
            shift 2
            ;;
        --android-abi)
            require_arg "$@"
            ANDROID_ABI="$2"
            shift 2
            ;;
        --android-platform)
            require_arg "$@"
            ANDROID_PLATFORM="$2"
            shift 2
            ;;
        --opencl-sdk)
            require_arg "$@"
            OPENCL_SDK="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

case "${BUILD_DIR}" in
    /*) ;;
    *) BUILD_DIR="${ROOT_DIR}/${BUILD_DIR}" ;;
esac

if [ -z "${OPENCL}" ]; then
    if [ "${MODE}" = "android-snapdragon" ]; then
        OPENCL="ON"
    else
        OPENCL="OFF"
    fi
fi
[ -n "${QNN}" ] || QNN="OFF"
[ -n "${HEXAGON}" ] || HEXAGON="OFF"
[ -n "${VULKAN}" ] || VULKAN="OFF"

[ "${MODE}" = "native" ] || [ "${MODE}" = "android-snapdragon" ] || die "unsupported mode: ${MODE}"

if [ "${RUN_TESTS}" -eq 1 ] && [ "${MODE}" != "native" ]; then
    die "--run-tests is only supported for native builds; Android binaries must be tested on device separately"
fi

if [ "${QNN}" = "ON" ]; then
    [ -n "${QNN_SDK}" ] || die "QNN is enabled but no SDK was provided; use --qnn-sdk or QNN_SDK_PATH"
    [ -d "${QNN_SDK}" ] || die "QNN SDK directory does not exist: ${QNN_SDK}"
    QNN_SDK="$(cd "${QNN_SDK}" && pwd)"
fi

if [ "${HEXAGON}" = "ON" ] || [ "${QNN_HEXAGON_BACKEND}" = "ON" ]; then
    [ -d "${HEXAGON_SDK}" ] || die "Hexagon SDK directory does not exist: ${HEXAGON_SDK}"
    [ -d "${HEXAGON_TOOLS}" ] || die "Hexagon tools directory does not exist: ${HEXAGON_TOOLS}"
    HEXAGON_SDK="$(cd "${HEXAGON_SDK}" && pwd)"
    HEXAGON_TOOLS="$(cd "${HEXAGON_TOOLS}" && pwd)"
fi

if [ "${MODE}" = "android-snapdragon" ]; then
    [ -n "${ANDROID_NDK}" ] || die "ANDROID_NDK_ROOT is not set; use --android-ndk"
    [ -d "${ANDROID_NDK}" ] || die "Android NDK directory does not exist: ${ANDROID_NDK}"
    ANDROID_NDK="$(cd "${ANDROID_NDK}" && pwd)"
fi

if [ "${CLEAN}" -eq 1 ]; then
    info "removing ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
fi

cmake_args=(
    -S "${ROOT_DIR}"
    -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DLLAMA_BUILD_TESTS="${BUILD_TESTS}"
    -DLLAMA_BUILD_TOOLS="${BUILD_TOOLS}"
    -DLLAMA_BUILD_EXAMPLES="${BUILD_EXAMPLES}"
    -DGGML_OPENCL="${OPENCL}"
    -DGGML_QNN="${QNN}"
    -DGGML_HEXAGON="${HEXAGON}"
    -DGGML_VULKAN="${VULKAN}"
)

if [ "${MODE}" = "android-snapdragon" ]; then
    cmake_args+=(
        -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake"
        -DANDROID_ABI="${ANDROID_ABI}"
        -DANDROID_PLATFORM="${ANDROID_PLATFORM}"
        -DCMAKE_C_FLAGS="-march=armv8.7a+fp16 -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE"
        -DCMAKE_CXX_FLAGS="-march=armv8.7a+fp16 -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE"
        -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG"
        -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG"
        -DCMAKE_C_FLAGS_RELWITHDEBINFO="-O3 -DNDEBUG -g"
        -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3 -DNDEBUG -g"
        -DGGML_OPENMP="OFF"
        -DGGML_LLAMAFILE="OFF"
        -DLLAMA_OPENSSL="OFF"
    )
    if [ -n "${OPENCL_SDK}" ]; then
        cmake_args+=(-DCMAKE_PREFIX_PATH="${OPENCL_SDK}")
    fi
fi

if [ "${QNN}" = "ON" ]; then
    cmake_args+=(
        -DGGML_QNN_SDK_PATH="${QNN_SDK}"
        -DGGML_QNN_ENABLE_CPU_BACKEND="${QNN_CPU_BACKEND}"
        -DGGML_QNN_ENABLE_HEXAGON_BACKEND="${QNN_HEXAGON_BACKEND}"
    )
fi

if [ "${HEXAGON}" = "ON" ] || [ "${QNN_HEXAGON_BACKEND}" = "ON" ]; then
    cmake_args+=(
        -DHEXAGON_SDK_ROOT="${HEXAGON_SDK}"
        -DHEXAGON_TOOLS_ROOT="${HEXAGON_TOOLS}"
    )
fi

if [ "${PROFILING}" = "ON" ]; then
    cmake_args+=(-DGGML_OPENCL_PROFILING="ON")
fi

if [ "${VULKAN}" = "ON" ]; then
    [ -z "${VULKAN_GLSLC}" ] || cmake_args+=(-DVulkan_GLSLC_EXECUTABLE="${VULKAN_GLSLC}")
    [ -z "${VULKAN_INCLUDE}" ] || cmake_args+=(-DVulkan_INCLUDE_DIR="${VULKAN_INCLUDE}")
fi

info "mode=${MODE}"
info "build_dir=${BUILD_DIR}"
info "type=${BUILD_TYPE}"
info "features: opencl=${OPENCL}, qnn=${QNN}, hexagon=${HEXAGON}, vulkan=${VULKAN}, profiling=${PROFILING}"

cmake "${cmake_args[@]}"

if [ "${CONFIGURE_ONLY}" -eq 1 ]; then
    info "configure-only requested"
    exit 0
fi

build_args=(--build "${BUILD_DIR}")
if [ "${#TARGETS[@]}" -gt 0 ]; then
    build_args+=(--target "${TARGETS[@]}")
fi

cmake "${build_args[@]}" --parallel

if [ "${RUN_TESTS}" -eq 1 ]; then
    ctest --test-dir "${BUILD_DIR}" --output-on-failure
fi

info "done"
