# Maintained TP2 integration

The fork's working branch is **`tp2`**. It combines canonical
[gufo-org/gufo](https://github.com/gufo-org/gufo), the RDMA implementation
originating in [Neuhaus](https://github.com/neuhaus/gufo/tree/rdma), and the
validated downstream fixes. Fork `main` mirrors canonical upstream; dated
integration and measurement branches remain historical references. Review new
canonical updates and relevant Neuhaus RDMA changes on `tp2`; qualify changes
before advancing deployed binaries. Do not merge unrelated fork development
solely to match a branch tip.

## November 1 checkpoint

Production source **`4e8a2dc0`** incorporates canonical **`f25f4744`** and the
current Neuhaus `rdma` head **`0b6978aa`**, plus the source ports from fork
PRs #1–#8. Existing PRs retain their original measured revisions; a source port
is not a formal GitHub merge. The current transport is **wire 21**, using
OS-selected registered **host** windows. Both peers need matching qualified
builds. This checkpoint does not establish a DMA-BUF performance gain.

The model owns native transport and TP runner creation through an optional,
host-only `TextProvider::create_parallel`/`TextParallelRuntime` interface.
Serving retains scheduling, mirroring, control messages and verdicts without
model-private includes. Canonical permanent API records and generated versioning
are retained; the distinct TP record produces API **0.1.12**. All six
include/ownership/family gates pass without exceptions.

On the latest binary: production HIP build, format/docs/API policy, isolation,
18 discovery cases, metadata and nine focused CPU tests pass. A fresh real TP2
start passes all twelve standard output/prompt/cache-work identities; greedy and
sampled keep/discard cache cases and cancellation in both slots preserve the
live width-two survivor and subsequent replay. Both hosts finish idle with no
checked transport error delta.

The preceding extension migration at **`739eb86a`** has the fuller numerical,
materialized-state, snapshot, injected-failure/recovery and actual disk-restart
coverage. The latest refresh changes build/API discovery rather than model math:
77 Flash-Next/Qwen-family translation-unit sources are identical; eleven host
commands add only the generated public-header include directory and normalized
HIP commands match. This comparison is not a fresh full numerical/fault/disk
sweep of `4e8a2dc0`. Revision attribution is retained in the
[exported reports](canonical-refresh-data/manifest.json).

## Migration measurements

Ten fixed matched pairs on `739eb86a`, twenty fresh process-pair starts and
240 exact request identities compare against the preserved pre-migration
EndRequest-fixed release. Same UD-Q4_K_XL/shared-Q8_0 MTP, 65,536 context
capacity, two slots, 2,048 prefill chunk, 1 GiB RAM cache and crossed 40+20 Gb/s
native links. The driver remains 81C6/TX32, four rails.

| Co-primary metric | Before | After | Paired change | Marginal 95% interval |
|---|---:|---:|---:|---:|
| C1 server-accounted decode | 66.7232 tok/s | 66.7240 tok/s | +0.0011% | [-0.2095%, +0.2003%] |
| C2 short complete client wave | 71.2399 tok/s | 71.5597 tok/s | +0.4489% | [-0.3701%, +1.3246%] |

Both results are **inconclusive**, not equivalence. C1 uses server-accounted
decode duration; C2 includes admission, prefill and terminal waits. C1 speculative
work matches; 32 C2 comparisons have adaptive-work differences. All 52 strict
timing entries over 20 request comparisons remain visible. No C1 flag occurs.
The largest coding wall delay (+328.229 ms) overlaps changed speculative work,
which does not prove complete attribution. Short-C2 TTFT has approximately
265/330 ms modes in both arms; their cause remains unresolved.

[Paired figure](canonical-refresh-data/before-after.png),
[statistics](canonical-refresh-data/statistical-summary.json),
[all flags](canonical-refresh-data/timing-flags.csv),
[flag review](canonical-refresh-data/timing-flag-review.json) and
[portable data verification](canonical-refresh-data/README.md) are retained.
These measurements do not test the later `f25f4744` refresh or establish a new
optimization gain. Existing earlier optimization studies remain separately
attributed; do not add their percentages or pool the studies.

## Upstreaming strategy

Keep one usable combined branch while submitting independently reviewable
changes to canonical upstream:

1. Maintain the generic cache-retention [PR #902](https://github.com/gufo-org/gufo/pull/902)
   and MTP-backoff [PR #912](https://github.com/gufo-org/gufo/pull/912) separately
   from the transport. Both were open when this checkpoint was prepared.
2. Separate the migrated GPU/session-test dependency and typed-loader repairs;
   these can be reviewed without a two-node runtime.
3. Offer the neutral optional parallel-provider contracts with CPU contract and
   registration tests. A mock-provider test proves interfaces and scheduling
   contracts, not real distributed performance.
4. Propose RDMA implementation pieces with default-off dependencies, CPU
   bootstrap/control/executor tests and reproducible two-host evidence. Keep
   hardware-dependent support in `tp2` when upstream cannot validate it yet.

The aggregate fork PR is the combined-branch review view. It does not require
canonical upstream to accept the full distributed stack, and does not replace
the focused upstream PRs. The historical checkpoints below preserve their
original scope and limitations.

# Historical bdf800de integration checkpoint

The working integration branch is `integrate/tp2-optimized-current-upstream`.
Canonical upstream is **gufo-org/gufo**, pinned here to
`bdf800de477e980904c26aad483123af7dfa6583`. Neuhaus supplies the RDMA lineage,
not the canonical upstream cadence. Source ports from fork PRs #1–#7 are now
included, together with the distributed draft-confidence compatibility fix.
Existing PRs remain drafts until the combined engine passes its remaining gates.
No operational default has changed and no new release speedup is claimed.

The current transport is version 21, using OS-selected registered host windows.
Both nodes must use the same qualified build. The original fixed-address/token-only
checkpoints below are retained for diagnosis and must not be deployed or used
as optimizer performance references.

## Historical bdf800de full-model checks

The first actual model load exposed an incompatibility missed by component tests:
upstream's 16,384-row chunk plus 128-row tail exceeds the native transport's
48 MiB partial-sum window at 2,560 floats per row. `PlanPrefillRows` preserves
upstream's local chunk and selects a smaller whole power-of-two chunk, including
tail space, when the communicator requires it. On this transport each trunk
batch holds 4,096 chunk rows plus up to 128 tail rows; existing split reduction
still permits two trunk batches per prefill step. The bounds remain checked at
load. Focused capacity tests include small contexts, insufficient windows and
transport limits. This is a load-compatibility fix, not a measured speedup.

With that correction, real TP2 model replay passes at short and actual 32,768-token
prefixes (65,536-token capacity): 72 paired cases, 576 session-stage records,
115 actual gathered heads and 144 complete materialized state comparisons per rank.
Gathered and full projection arms match tokens, complete finite-logit hashes,
positions, sampler RNG, draft/acceptance work and state bytes. Cross-rank records
also match. All eligible packed widths 2–7 actually execute, as does width 8's
fallback, plus mixed-AR and pure-AR fallbacks. Production confidence gating is
retained: requested draft caps are recorded separately from actual packed rows.
The old fixed-width fixture's early stop is retained as an invalid assumption,
not presented as a model mismatch. Driver/kernel windows and final idle checks pass.

This uses a test-only host predicate toggle/counters linked against production
model/GPU/transport objects. It establishes equality within its exercised paths,
not release throughput, HTTP lifecycle behavior or arbitrary long-context shapes.
Raw scripts, frozen source hashes, build commands, records and checks are in
`tp-cache-retention/integrated-head-quality-v3` in the investigation directory.
The ordinary two-row path also passes against full projection and an independent
AR teacher at short and actual 32K contexts: 160 records and 80 gathered heads per
rank, with exact full logits, greedy choices, seeded sampled replay and cross-session
snapshot continuation. Four materialized end-state pairs match byte for byte.
Supplementary real-model tests exercise actual all-MTP fallback widths 9, 10 and 12
using eight supported sessions with zero/one-proposal caps: 48 records and 24 full
state pairs per rank match, and no gathered head runs. These wider fallback cases
use short, distinct prefixes and 256-token session capacity; they are not 32K cases.
Confidence gating remains unchanged. A preliminary fixture exceeded the supported
eight-session limit and was rejected at load; its failure is retained separately.
Normal HTTP cache/cancellation checks also pass on the actual production server:
four greedy/sampled keep/discard interrupted-assistant cache cases and nine
cancellation/control records (37 inference POSTs plus a readiness GET). Cached continuations
match fresh controls; both victim roles leave a live width-two survivor matching
its solo result, and post-cancellation replay matches. All substantive completed
requests exercise MTP. SHA-checked binaries/settings, frozen sources, task-owned
process cleanup, both final idle inventories and checked driver/kernel windows
pass. This is candidate-only correctness, not matched request-time equivalence.
Borrowed same-session and cross-session restoration, serialized cross-session
restoration and snapshot-file round trips also pass at short and actual 32K
contexts for greedy and sampled requests: 104 records, 44 gathered heads and
32 complete post-replay state comparisons per rank. Tokens, full logits, RNG,
positions and proposal work match the full-projection reference. File round trips
use the same loaded model process; this does not establish HTTP disk-cache restart
or peer-loss recovery.

## Current failure and disk-restart checks

The refreshed fault harness pins the exact production local-copy return PC and
requires a live outer `ForwardBatch` stack frame. Seven CPU-only arming/range/
geometry/ancestor/one-shot controls pass. Injected HIP copy API failures on each
rank and a task-owned rank-1 process exit all produce explicit non-normal errors
on both HTTP streams within the predeclared ten-second bound, without a socket
error or transport timeout being accepted as a pass. Complete request times are
0.61, 0.90 and 0.93 seconds; these include admission and prefill, not just abort
handling. Each fault is followed by a fresh process pair: all nine recovery
outputs and proposal/acceptance work match a newly acquired current-engine
control. All fourteen rank/phase counter windows have zero checked error deltas;
final PID, idle, boot, topology and kernel checks pass. This establishes bounded
failure containment and fresh restart recovery, not transparent reconnection,
a physical GPU failure or a cable-loss guarantee.

The current upstream disk-startup fixture also passes with a real two-host
lifecycle and only its TP2 cache log signature adapted. Both processes are
restarted with independent, initially empty task-owned identity/cache directories.
A cold 2,294-token prompt evaluates all tokens, RAM reuse evaluates none, and
restoration after restart reports an actual disk hit with 2,287 cached tokens
and seven evaluated tokens. All three outputs match exactly, with 18 draft
proposals and 15 accepted on each request. Both ranks' checkpoint inventories
and changed process identities are verified; checked counters stay clean.
The two worker-control-close errors occur during deliberate head shutdown and
are classified explicitly, with no unknown server errors. This is greedy disk
restore coverage, not sampled restore, cancellation during restore or physical
storage-failure coverage. Initial artifact hashing is normal engine identity
validation; no separate n-gram-table warming or system change is performed.

Raw frozen protocols and outputs are in `integrated-head-faults-v1` and
`integrated-disk-restart-v1` alongside the earlier qualification directories.
Matched release performance remains unmeasured on this integration.

## Historical initial reconciliation

The initial checkpoint combined upstream
`8f5aece0c27ca35e3a0831da9ff960e8be78b30e` with RDMA source
`0b6978aa1709f2ef863f5407e4ca7261191e9046`, plus cache-retention and MTP-backoff
patches. The following sections describe those earlier checkpoints; later source
ports and corrected confidence supersede their limitations.

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

## Initial checkpoint validation

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

Build the release profile and measure matched prefill/decode
with identical upstream, confidence correction and allocator in both arms.
Historical downstream speedups do not transfer automatically. Cooperative PLE
reads, peer-combine and subset-head sharding remain separately planned ports.
Finish qualification before changing the operational default or merging the
complete integration into the fork's main branch.

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
