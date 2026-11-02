#!/bin/bash
set -euo pipefail
root=/home/rkuo/strix-dispatch-investigation
rocm_root=/home/rkuo/.local/opt/rocm-7.14.1
deps_root=$root/dependencies/gufo-dev
export CPLUS_INCLUDE_PATH="$deps_root/usr/include/x86_64-linux-gnu${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LD_LIBRARY_PATH="$deps_root/usr/lib/x86_64-linux-gnu:$rocm_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PKG_CONFIG_SYSROOT_DIR="$deps_root"
export PKG_CONFIG_LIBDIR="$deps_root/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/share/pkgconfig"
exec 9>"$root/overnight/results/model-matrix.lock"
flock -n 9
cd "$root/tp-cache-retention/upstream-ffn-hc-baseline-worktree"
revision=upstream-ffn-hc-baseline-$(git rev-parse --short=12 HEAD)
cmake --preset release -B "$root/tp-cache-retention/upstream-ffn-hc-baseline-release-build" \
 -DGUFO_BUILD_TOOLS=OFF -DBUILD_TESTING=OFF -DGUFO_REVISION="$revision" \
 -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
 -DCMAKE_HIP_COMPILER="$rocm_root/lib/llvm/bin/clang++" \
 -DCMAKE_C_COMPILER="$rocm_root/bin/amdclang" \
 -DCMAKE_CXX_COMPILER="$rocm_root/bin/amdclang++" \
 -DPKG_CONFIG_EXECUTABLE="$deps_root/usr/bin/pkgconf" \
 -DCMAKE_PREFIX_PATH="$rocm_root;$deps_root/usr"
cmake --build "$root/tp-cache-retention/upstream-ffn-hc-baseline-release-build" --target gufo --parallel 4

touch "$root/tp-cache-retention/upstream-ffn-hc-v1/baseline-release-build-complete"
