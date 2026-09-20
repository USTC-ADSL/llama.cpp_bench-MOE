#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_DIR="$ROOT/MOE"
BUILD_DIR=${BUILD_DIR:-"$SOURCE_DIR/build/host"}
JOBS=${JOBS:-$(nproc)}

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DMOE_WITH_GGML=ON

cmake --build "$BUILD_DIR" --parallel "$JOBS"

echo "Host generator: $BUILD_DIR/bin/expert-pack-gen"
echo "Host CPU profile: $BUILD_DIR/bin/expert-io-profile"
