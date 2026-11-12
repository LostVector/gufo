# HC C2 follow-up, November 8, 2026

PR16 is merged at `59855a97a52cb678d66ce3edc3964ceaac6b974e`.
PR14 and PR15 remain drafts, unmerged. This follow-up does not establish a
serving gain, equivalence, or a candidate-specific transport defect. No small-gain
cutoff was used. Historical component/GPU correctness evidence is retained and
is not pooled with this experiment.

## Production comparison

All three binaries use official CMake release preset (RelWithDebInfo, tests off),
private ROCm 7.14.1, gfx1151 and RDMA. Base is the PR16 merge above; candidate
production patches are independently cherry-picked `f5e9290c` (PR14) or
`74f385dc` (PR15), never stacked. Only production batch.cpp differs. All 375
normalized compiler commands match, normalizing only GUFO_REVISION. Each build
log explicitly contains compilation of its changed batch.cpp; source patches,
exact revision, binary hashes and compiler inventories are archived.

Driver B6C332F34D27E163651EE38 on both nodes; two existing cables, TX32,
RX progress 1ms, active polling off. Per-launch inventory records boot IDs,
cmdlines, GPU-independent driver parameters and link state. None changed.

Target UD-Q4_K_XL plus shared Q8_0 MTP, greedy, thinking off, 65,536 context
capacity, 2,048 prefill chunk, 1 GiB RAM cache; cache_prompt=false. PR14 uses
2 slots, PR15 8 slots, both tested with two active clients. Natural prompts
request 128 generated tokens; coding prompts 256. Actual prompt/cache token
counts and all adaptive MTP work are in the raw records: capacity is not occupied
context length. C1 2,048-prompt controls precede/follow the concurrent waves.

The fixed acquisition has eight fresh starts, two process pairs per candidate,
four waves per workload per start: 168 total requests, including 128 measured
C2 requests. Outputs, prompts, prefill/cache work and completion counts match in
every comparison. Actual MTP proposals occur in every measured C2 request.
No profiler or admin sampler ran during these production comparisons.

| Candidate | Pair | Natural C2 wave throughput change | Coding C2 change |
| --- | --- | ---: | ---: |
| PR14 | 1 | -0.293% | +0.343% |
| PR14 | 2 | -0.473% | -2.317% |
| PR15 | 1 | -0.521% | -2.156% |
| PR15 | 2 | -0.479% | +0.097% |

These are descriptive geometric paired changes in aggregate client wave
throughput, including admission/prefill/decode, not isolated decode-only gains.
Waves in one process are not independent process replicates. Two process pairs
are insufficient for a significance or equivalence conclusion.

The unchanged predeclared gate (candidate slower by both 5% and 3ms) flags
34 correlated phase entries. The largest is PR14 pair2 coding actor1:
+326.9ms decode, +324.7ms wall. There are 128 adaptive-work field differences;
equal output does not mean equal proposal counts, acceptance or rounds.
All flags/work differences are retained in analysis.json and phase-flags.csv.

### Transport confounding

Three candidate launches record recovered errors, while all four baseline
launches and PR15 candidate2 are clean:

| Launch | EVO deltas | Corsair deltas |
| --- | --- | --- |
| PR14 candidate1 | lost4, bad-frame3, CRC3, overrun1 | retransmit2 |
| PR15 candidate1 | retransmit1 | lost2, bad-frame1, CRC1 |
| PR14 candidate2 | lost1, bad-frame1, CRC1 | retransmit1 |

All requests complete correctly. These are launch-level counters, not exact
per-request error timestamps. The small sample does not establish that an HC
change caused the errors or that cabling explains them. Affected launches remain
confounded; no samples were discarded or replaced.

## Source observation

A separate, bounded observer was triggered by the retained phase flags and
>80ms first-followup gaps. It records same-host source timestamps and flushes
only at orderly exit; no cross-host timestamp subtraction. Its data are never
used for throughput or gain estimates. Dispatch/completion wall time includes
GPU execution, synchronization, transport and CPU work; it is not pure GPU time.

The first observer candidate builds incorrectly reused baseline objects because
copy2 preserved source mtimes. Those raw files are retained but invalid for
candidate attribution. V2 builds have the same issue and were not executed.
V3 uses copyfile plus touch, and asserts every instrumented translation unit
was compiled for each role. This does not affect the untraced production builds,
whose changed-file compilation is independently present in their logs.

The server uses peer identity rather than X-Client-ID here. Source/client joins
use increasing server request IDs against serialized waves with staggered actor
admission, checking each publication against the request's monotonic interval.

The verified observer confirms PR14 executes 59,539 eligible Q8 producer calls
per rank; PR15 executes 28,032 eligible fused-branch calls per rank. These are
host-path observations over the whole cohort (including capture/setup), not a
GPU replay kernel count, a consumer-hit proof, or an end-to-end saving estimate.
Both paths really execute under the current model/settings.

The new source trace attributes the ordinary initial C2 gap to serial first
native-MTP decode calls. The second actor's snapshot finishes before the first
actor's decode returns, but its future is consumed only afterward. One native
MTP call then produces the second actor's next publication. Baseline and both
candidates exhibit the same sequence. In the four natural waves of the first
baseline observer, the first calls propose three tokens and accept two (actor0)
or three (actor1): these are not zero-acceptance rounds. All source work records
are retained; do not infer proposal behavior from SSE chunk spacing alone.

A dispatch wall span includes model/GPU/RDMA/wait work; eliminating its CPU
wait does not eliminate that work. These source observations explain the common
first-followup pattern, not every intermittent production timing regression or
recovered transport error. No arbitrary HIP-launch attribution is warranted.

### Valid observer timing and transport results

Each row below is the median of four natural-C2 waves, for actor1 (the second
admitted request). All values are milliseconds. These are source diagnostics,
not comparisons suitable for serving-gain statistics.

| Launch | First-followup publication gap | Snapshot worker | Worker end → future get | Overlap of that delay with model dispatch | Own first decode call |
| --- | ---: | ---: | ---: | ---: | ---: |
| p14-b | 89.631 | 2.430 | 43.631 | 43.623 | 43.544 |
| p14-c | 90.181 | 2.289 | 43.831 | 43.825 | 43.835 |
| p15-c | 89.492 | 2.411 | 43.537 | 43.529 | 43.524 |
| p15-b | 90.210 | 2.286 | 43.590 | 43.583 | 44.174 |

The first-followup publication gap is covered by model-dispatch wall spans for
87.3–88.4ms at these launch medians. Future get itself takes median 1.5–3.7µs.
The 43.5–43.8ms worker-ready-to-consumption delay overlaps almost entirely with
another request's model call, rather than an idle scheduler pause.

All 84 diagnostic requests complete with paired output/prompt/cache identities
matching; 80 requests have joined source observations (the four eight-token
warmups are deliberately outside source scope), with no ring overflow.

Recovered transport errors also occur in the **diagnostic baseline** p15-b:
EVO lost4/bad-frame2/CRC2/overrun2, Corsair retransmit1. Diagnostic PR14 has EVO
retransmit1/lost7/bad-frame4/CRC4/overrun3 and Corsair retransmit2/lost2/bad-frame2/CRC2.
Diagnostic PR15 has EVO lost4/bad-frame4/CRC4 and Corsair retransmit2. The two-slot
diagnostic baseline is clean. These observations show candidate-only errors are
not universal; observer and untraced counts remain separate, and neither set
establishes an error cause or exact per-request latency attribution.

## Decision and next experiment

Keep PR14 and PR15 unmerged. Their previously measured component improvements
remain valid within their original scopes, but this fresh current-base serving
sample does not clear the latency flags or demonstrate a gain. The reason is
unresolved evidence, not that a statistically significant small gain would be
uninteresting. No candidate sample has been removed or replaced.

The next scheduling experiment should examine forming the first two-request
native-MTP batch once the already-submitted snapshot completes, rather than
immediately committing to a blocking single-request decode. Preserve TP snapshot
gate ordering, dependency correctness and C1 admission latency; use an isolated
probe and measure batch cost before claiming any saved milliseconds. The current
trace shows serial work, not 44ms of idle CPU time that can simply be removed.
Separately, correlate recovered transport errors with driver event timestamps in
a fixed acquisition before assigning candidate-specific timing effects. Do not
rerun until green or pool these data with the older driver/source studies.

## Reproduction and artifacts

The accompanying archive contains the fixed PROTOCOL.md, exact production
patches, sources for the private observer, complete client content/timestamps,
per-launch driver/link/boot inventories and counters, all build logs and compiler
inventories, analysis scripts, and all invalid/valid observer attempts clearly
separated. Executables and models are excluded; their SHA256 identities remain.
The evidence branches used to build the production candidates are local;
reconstruct them from public base 59855a97 and source-pr14.patch or
source-pr15.patch. Never apply both to reproduce these separate comparisons.

On the recorded lab layout, build-variants.py creates/freeze the three standard
CMake production builds; run-locked.sh acquires the shared model lock, pauses the
trial, runs run.py, and restores native MTP on exit. prepare-trace.py plus
build-trace-v3.py constructs the checked observer builds; trace-v3-run-locked.sh
runs the separate diagnostic. Existing acquisitions intentionally refuse
overwrite: use a fresh lab/output path, updating the absolute paths consistently.
Run analyze.py, analyze-trace-v3.py and observer-summary.py offline afterward.
This is a record of reproducible lab commands, not a turnkey deployment script
for arbitrary remote hardware.

The archive checker validates hashes, retained identities/flags and observer
coverage. It does not rerun a GPU, prove mathematical equivalence, qualify the
live hosts or establish statistical significance.

[Raw archive](hc-c2-followup-2026-11-08-data/acquisition.zip) and
[hashed inventory](hc-c2-followup-2026-11-08-data/archive.json) retain all 392 entries.
Run `python3 docs/patches/hc-c2-followup-2026-11-08-data/check.py` for the offline
integrity/coverage check.
