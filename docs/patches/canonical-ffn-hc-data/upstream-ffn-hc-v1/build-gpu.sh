#!/bin/bash
set -euo pipefail
root=/home/rkuo/strix-dispatch-investigation
out="$root/tp-cache-retention/upstream-ffn-hc-v1/operator"
source_tree="$root/tp-cache-retention/upstream-ffn-hc-worktree"
build_dir="$root/tp-cache-retention/upstream-ffn-hc-gpu-build"
rocm_root=/home/rkuo/.local/opt/rocm-7.14.1
deps_root="$root/dependencies/gufo-dev"
mkdir "$out"
exec 9>"$root/overnight/results/model-matrix.lock"
flock -n 9
export CPLUS_INCLUDE_PATH="$deps_root/usr/include/x86_64-linux-gnu${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LD_LIBRARY_PATH="$deps_root/usr/lib/x86_64-linux-gnu:$rocm_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PKG_CONFIG_SYSROOT_DIR="$deps_root"
export PKG_CONFIG_LIBDIR="$deps_root/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/share/pkgconfig"
unset LD_PRELOAD ROCP_TOOL_ATTACH STRIX_READ_OUTPUT STRIX_FINISH_TRACE GUFO_TP_TRACE_PREFIX
cd "$source_tree"
cmake --preset gpu-test -B "$build_dir" -DGUFO_BUILD_TOOLS=OFF  \
 -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DCMAKE_HIP_COMPILER="$rocm_root/lib/llvm/bin/clang++" \
 -DCMAKE_C_COMPILER="$rocm_root/bin/amdclang" -DCMAKE_CXX_COMPILER="$rocm_root/bin/amdclang++" \
 -DPKG_CONFIG_EXECUTABLE="$deps_root/usr/bin/pkgconf" -DCMAKE_PREFIX_PATH="$rocm_root;$deps_root/usr" > "$out/build.log" 2>&1
cmake --build "$build_dir" --target gufo qwen38_flash_next_hc_mix_ops_test --parallel 4 >> "$out/build.log" 2>&1
ctest --test-dir "$build_dir" -R '^qwen38_flash_next\.hc_mix_ops$' --output-on-failure --no-tests=error --output-junit "$out/operator-tests.xml" > "$out/tests.log" 2>&1
python3 - "$out" <<'PY'
from pathlib import Path
import sys,xml.etree.ElementTree as E,json
out=Path(sys.argv[1]);tree=E.parse(out/'operator-tests.xml');cases=tree.findall('.//testcase');assert len(cases)==1 and all(c.find('skipped')is None and c.find('failure')is None and c.find('error')is None for c in cases)
(out/'validation.json').write_text(json.dumps({'real_GPU_operator_tests':1,'skips':0,'scope':'Freshcanonical8ab headers/objects, originalHCfixture+48batchreference conditions, HIP/assertionsenabled. No fork/privateABI/transportdependency. Modelstate/performance qualification separate.'},indent=2)+'\n')
PY
touch "$out/complete"
