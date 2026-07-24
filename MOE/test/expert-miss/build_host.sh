#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SOURCE_DIR="$ROOT/MOE/test/expert-miss"
BUILD_DIR=${BUILD_DIR:-"$SOURCE_DIR/build-host"}
JOBS=${JOBS:-$(nproc)}

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DEXPERT_MISS_BUILD_GENERATOR=ON

cmake --build "$BUILD_DIR" --target expert-pack-gen expert-ufs-probe --parallel "$JOBS"

echo "Host generator: $BUILD_DIR/expert-pack-gen"
echo "Host CPU probe: $BUILD_DIR/expert-ufs-probe"

