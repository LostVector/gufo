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

## Optimized-branch follow-up: scheduling and confidence compatibility

The optimized integration branch incorporates the MTP snapshot-handoff and
recursive AR-fairness source from fork PRs #1 and #3. The exact gate header
matches the reviewed PR; integrated gate and TP-executor CPU tests pass after
each merge. Their historical speed results remain historical, not measurements
of this newer upstream engine.

Inspection then found a compatibility bug in the base checkpoint: its legacy
TP full-vocabulary greedy draft-head branch wrote/copied only the token, while
new upstream `MtpForward` also reads a confidence word. That path could leave
confidence stale. The optimized branch now computes rank-local stable softmax
denominators in the unused padded exchange slots, completes the global token
and probability after the existing all-reduce, and copies both words back.
The newer upstream confidence policy is retained. There is no extra collective
or new option. Transport version 18 rejects old token-only payload peers.

The focused GPU test checks 60 fixtures and 120 split-versus-full results against an
independent F64 probability oracle at vocabularies 2, 10, 256, 99328 and 248320,
including ties, nonfinite values, empty rank support, large shifts and near-gate
probabilities. Output guards and foreign exchange slots are checked. The
predeclared tolerance is 2e-6 times max(reference probability, 1e-5); worst observed
absolute error was 7.75991e-8. A token-only finish negative control fails on the
poisoned probability word. This control covers the kernel write contract, not
the production `MtpHead` host-transfer route by itself.

A real two-node native test then exercised the same pack/all-reduce/finish/
two-word-download path using 248320-logit fixtures. Both ranks returned token 0
and probability 1/248320 for uniform logits, token 124160 and probability .90001
for a near-threshold fixture, and token 124160 and probability 1 with no finite logits
on rank 0. The independent 10 KiB GPU sum also passed. Version 18/8 and 8/18 reject
before inference. Both nodes returned to the approved idle driver baseline.
This is an actual transport/component check, not full-model qualification or a
throughput result. Files and raw outputs are in `confidence-protocol-v1` under
the investigation directory above.

The base binary is preserved for diagnosis. Because its MTP confidence is
known to be stale on this branch, it must not serve as the performance reference
for later optimizer gains. Apply this compatibility correction to both arms.
Full-model/cache/cancellation qualification and the remaining target-head and
lab-optimization integration are still outstanding.

## Target-head and host-window ports

The optimized branch now also ports PR #5's ordinary two-row target-head gather,
PR #6's OS-selected registered host windows, and PR #7's batched gather. The
ordinary path retains its original two-row MTP guard; packed all-MTP batches
retain the 2–7-row guard, with other paths falling back to the existing full
projection. The newer upstream confidence policy and its corrected distributed
probability output stay intact. Older policy prototypes were not imported.

Transport versions 19, 20 and 21 identify the successive integration stages;
the combined port uses 21. All peers must use matching builds. The archived
measurements and wire versions in the individual PR documents describe their
original reviewed revisions, not a new speed result for this branch.

The complete combined server builds. The allocator's CPU regression passes,
using production mapping helpers and a simulated HIP registrar. It is supported
by a separate real two-node test on the unchanged native driver: GPU sum and
confidence checks plus gathers at 1, 2, 3, 7 and 12 rows. Every gathered word
matches exactly, including negative zero and NaN payload bits. Helper coverage
at 1/12 rows does not expand production eligibility beyond the guards above.

The same test passes with the legacy fixed address deliberately occupied on
both processes. The new host windows select unused addresses; an old fixed-VA
build fails with EEXIST as expected. Mixed version 21/18 and 18/21 peers reject.
Across the normal and occupied positive runs, 24,832,000 float values are compared
bitwise on the two hosts. The actual transport reports mapped host windows;
DMA-BUF registration is unsupported here, so this is not a zero-copy claim.
Checked driver error/retry/drop deltas remain zero and final inventories are
idle. Raw fixed-sequence results are in `head-protocol-v1` in the investigation.

The imported statistical data still reconstructs its original paired estimates
and timing flags. CSV line endings were normalized only in this integration
copy; raw compressed requests/prompts and numeric rows were retained. Full-model
projection/state/caching/failure qualification and matched release performance
measurements remain required. Cooperative PLE reads, peer-combine and subset-head
sharding from the older lab baseline remain separately planned ports.

## Upstream refresh before model qualification

Canonical upstream advanced to `bdf800de477e980904c26aad483123af7dfa6583`
after the initial integration, adding the Qwen prefill GDN fusion and DeepSeek
decode/sampling changes. The optimized branch merges those two commits without
source conflicts. The earlier component checkpoint remains pinned separately;
its old binary is preserved rather than relabeled as the new engine.

On the refreshed source, the complete server builds. Selected MTP, GDN,
attention and projection GPU suites pass together; sampling, logit-sampler,
TP-executor and snapshot-gate CPU suites pass. Extension metadata checks pass
with the existing private dependency prefix. The initial standalone metadata
run could not find OpenSSL development files in its nested CMake process; that
failure is retained and was resolved through test-process environment settings,
without installing a system dependency or weakening the check. These are still
component/integration checks, not full-model or performance qualification.

Future optimizer comparisons must use the same refreshed upstream, corrected
confidence path and dynamic host windows in both arms. The older preserved
binaries serve provenance and diagnosis; comparing across upstream changes
would conflate their effects with target-head or driver improvements.
