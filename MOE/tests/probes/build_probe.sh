#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SOURCE_DIR="$ROOT/MOE"
BUILD_DIR=${BUILD_DIR:-"$SOURCE_DIR/build/android-raw"}
ANDROID_NDK=${ANDROID_NDK:-/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d}
HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT:-/mnt/sda1/pzw/HeteroCompute/Qualcomm/Hexagon_SDK/6.4.0.0}
HEXAGON_TOOLS_ROOT=${HEXAGON_TOOLS_ROOT:-"$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04"}
JOBS=${JOBS:-$(nproc)}

for required in \
    "$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    "$HEXAGON_SDK_ROOT/build/cmake/hexagon_fun.cmake" \
    "$HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-clang"; do
    if [[ ! -e "$required" ]]; then
        echo "missing required build input: $required" >&2
        exit 2
    fi
done

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-31 \
    -DHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT" \
    -DHEXAGON_TOOLS_ROOT="$HEXAGON_TOOLS_ROOT" \
    -DPREBUILT_LIB_DIR=toolv19_v79 \
    -DBUFFER_CAPACITY_BUILD_HTP=ON \
    -DMOE_BUILD_DEVICE_PROFILES=ON

cmake --build "$BUILD_DIR" --parallel "$JOBS"

for output in \
    "$BUILD_DIR/bin/buffer-capacity-probe" \
    "$BUILD_DIR/bin/htp-map-latency-probe" \
    "$BUILD_DIR/bin/shared-buffer-demo" \
    "$BUILD_DIR/bin/shared-buffer-device-tests" \
    "$BUILD_DIR/bin/buffer-lifecycle-profile" \
    "$BUILD_DIR/bin/expert-buffer-profile" \
    "$BUILD_DIR/tests/probes/libbuffer-capacity-htp-v79.so"; do
    if [[ ! -f "$output" ]]; then
        echo "expected build output is missing: $output" >&2
        exit 3
    fi
done

echo "Android programs: $BUILD_DIR/bin"
echo "HTP v79 test skel: $BUILD_DIR/tests/probes/libbuffer-capacity-htp-v79.so"
