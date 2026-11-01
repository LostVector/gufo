# Current-upstream RDMA integration checkpoint

This experimental integration reconciles `gufo-org/gufo` main at
`8f5aece0c27ca35e3a0831da9ff960e8be78b30e` with the RDMA source at
`0b6978aa1709f2ef863f5407e4ca7261191e9046`. The main-side parent also contains
the separately reviewed cache-retention and MTP-backoff patches. Canonical
upstream remains **gufo-org/gufo**. The measured optimization branches are
preserved separately.

This is an integration checkpoint, not a new performance release or a claim
that the complete optimized TP2 stack has been consolidated. In particular,
the separate snapshot-handoff/fairness, ordinary target-head, dynamic host-window
allocation and batched target-head review branches are not all included here.
The old fixed-address host-window limitation is still present until the allocator
patch is integrated.

## Reconciliation

Twenty-nine files conflicted with current upstream. Resolution preserves:

- Upstream's shared serving library, generated model/test inventory, extension
  include facades, ROCm package selection and Windows benchmark process cleanup.
- RDMA bootstrap/control/executor hooks, failed-peer cache-reuse checks,
  rank-local attention/recurrent/expert dimensions and explicit reductions.
- Upstream's register-owned 128-row routed prefill kernels and side-stream
  execution, alongside existing paired Q8 and 64-row distributed shapes.
- Independent convolution-channel extent and side-stream row-tile offset.
- Non-owner shared-expert output clearing on its producing stream, before its
  completion event; placing that clear on the main stream would introduce a
  race with the new overlap.
- Existing unknown-link rejection in the dependency graph. Static private
  system libraries are now classified like their plain counterparts; a real
  CMake regression checks both a retained project edge and rejected unknown
  dependency. The imported verbs target is visible to the root graph audit.

The first routed-expert suite rejected a Q5_1 TP down projection: 3000 tokens,
64 experts, m=2560, k=320, 64-row tiles, 80 tiles. The combined switch had
omitted the old 64-row case while retaining the new 128-row specialization.
Restoring the existing generic 64-row dispatch fixed that failure without
removing the new 128-row path. The failure and diagnostic run are retained in
the investigation, rather than discarded as noise.

## Validation

Environment: two 128 GB Strix Halo hosts, gfx1151, Linux 7.3.0-strix-test,
private ROCm 7.14.1 / Clang 23; approved native driver unchanged. No kernel,
boot, driver, system-library, ROCm or service configuration was changed.

On the final source, all six selected GPU operator suites passed together:
MTP, HC mixing, GDN, attention, projection and routed WMMA. The projection
suite covers full and TP-half fused SSM convolution, output/state equality
and side-stream replay. Total correctness-run wall time was 140.01 seconds;
this is not an inference speed result. The source manifest stayed unchanged
through acquisition. Six focused CPU tests also passed: TP control, TP disk
store, TP executor, n-gram reads, TP partition and GID selection.

The complete `gufo` server builds with HIP and native RDMA enabled. Version and
serving-help checks pass, including TP and MTP options. The validated executable
has SHA256 `992f2c0dd16a5ddb4f570c0cd27e3860c62efb20cf6ba2ea4fb12f593d156e11`.
It is a `gpu-test` / RelWithDebInfo build with assertions, not a headline release
benchmark. It was built before the merge checkpoint was committed; source
manifests associate its exact files with this integration, independently of the
embedded pre-commit revision string.

An entry-point replacement linked against those production objects performed a
real native-provider, two-host GPU all-reduce. Both processes exited zero and
verified all 2560 float values equal three after exchanging 10 KiB. Checked
transport error/retry/drop deltas stayed zero, rails ended unbound, final baseline
inventories passed and checked kernel windows were clean. This verifies a basic
collective, not model throughput or a target-head gather feature.

The first attempt could not start Corsair's process because the newly linked
upstream audio dependency `libsoxr.so.0` was absent there. A SHA-matched copy in
the experiment's private runtime directory resolved that loader failure. No
system package/library was installed. The controller was also corrected to poll
every process, avoiding a delayed report of a peer exit. A later final assertion
expected the wrong success string; independent analysis validated the saved
successful run without repeating it. The reused probe's printed target-head
feature label was hard-coded and is explicitly not evidence of negotiation.

Format (839 C++ files), documentation, generated infrastructure, actual HIP link
graph and all thirteen graph regression checks pass. Model/core and serving
include boundaries, target ownership and the family graph pass. Nix package
builds were not run on these hosts.

## Focused reproduction

Use the documented source-build dependencies and the same toolchain:

```sh
cmake --preset gpu-test -DGUFO_ENABLE_TP2_RDMA=ON
cmake --build --preset gpu-test --parallel 4 --target gufo \
  qwen38_flash_next_mtp_ops_test qwen38_flash_next_hc_mix_ops_test \
  qwen38_flash_next_gdn_ops_test qwen38_flash_next_attention_ops_test \
  qwen38_flash_next_projection_ops_test qwen38_flash_next_routed_wmma_ops_test
ctest --test-dir build/gpu-test --output-on-failure --timeout 600 -j 1 \
  -R '^qwen38_flash_next\.(mtp_ops|hc_mix_ops|gdn_ops|attention_ops|projection_ops|routed_wmma_ops)$'
python3 tools/ci/check-target-graph.py --report build/gpu-test/gufo-link-graph.json
python3 tests/tools/check_target_graph_test.py build/gpu-test
```

For CPU checks use a separate `cpu-test` build with native RDMA disabled; it
requires HIP when enabled. Build `tp_control_test`, `tp_disk_store_test`,
`tp_executor_test`, `qwen38_flash_next_ngram_test`,
`qwen38_flash_next_tp_partition_test` and `qwen38_flash_next_rdma_gid_test`.

Local raw logs, exact private commands, source manifests, retained failures and
transport snapshots are under
`/home/rkuo/strix-dispatch-investigation/tp-cache-retention/fork-integration-v1/`.

## Remaining gates

Full two-node model/HTTP correctness, long-context/cache/cancellation behavior,
and matched release prefill/decode measurements have not been run on this
integrated engine. Historical downstream speedups do not transfer automatically
to it. Consolidate the remaining reviewed changes with their dependencies and
run those checks before changing the operational default or merging this entire
integration into the fork's main branch.
