#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SOURCE_DIR="$ROOT/MOE/test/expert-miss"
BUILD_DIR=${BUILD_DIR:-"$SOURCE_DIR/build-android-cpu"}
ANDROID_NDK=${ANDROID_NDK:-/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d}
JOBS=${JOBS:-$(nproc)}

TOOLCHAIN="$ANDROID_NDK/build/cmake/android.toolchain.cmake"
if [[ ! -f "$TOOLCHAIN" ]]; then
    echo "missing Android NDK toolchain: $TOOLCHAIN" >&2
    exit 2
fi

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-31 \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DEXPERT_MISS_BUILD_GENERATOR=OFF

cmake --build "$BUILD_DIR" --target expert-ufs-probe --parallel "$JOBS"

if [[ ! -f "$BUILD_DIR/expert-ufs-probe" ]]; then
    echo "expected Android probe is missing: $BUILD_DIR/expert-ufs-probe" >&2
    exit 3
fi

echo "Android CPU probe: $BUILD_DIR/expert-ufs-probe"

