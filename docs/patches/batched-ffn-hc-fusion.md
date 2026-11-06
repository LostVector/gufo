# Eligible target-batch FFN HC fusion (review draft)

Small target batches currently launch the FFN HC combine, two projections,
activation, epilogue and quantization separately. This change routes the
immediate combine/mix pair through the existing fused decode implementation
when its geometry, Q8 format, row and residual-state guards permit it. The fused
implementation also supplies its existing ready-Q8 output. Wider shapes retain
their current route.

**Component saving is established; serving benefit remains unproven.** This
branch is for review and is not adopted into maintained `tp2`. It introduces
no runtime flag, public API, allocation, driver or transport change.

## Source and validation boundaries

- Production change: `74f385dc46e68e06be8975c6ab41682d7591add4`, on maintained
  `d038b2e8` (qualified production code `9c715720`). Only the immediate target
  FFN combine/mix pair changes; the final/head combine and predictor route retain
  their original scheduling.
- Test head: `4749d941f44a03a2fb8351f16bea63e739de15ff`. Production
  `src`, `include`, `cmake` and `CMakeLists.txt` are identical to `74f385dc`.
  The test preserves the original scalar, projection, injection and CPU-reference
  checks and adds 48 batch-reference conditions (rows 1–8, injection on/off,
  scales 0.0001/1/10000). Exact intermediate/active-byte, alias, repeated-launch
  and graph checks pass with real HIP and assertions enabled. Shared formatting
  passes for 895 C++ files.
- Full fresh production release: official release preset, RelWithDebInfo,
  `BUILD_TESTING=OFF`, RDMA enabled, private ROCm 7.14.1/gfx1151. All 336 normalized
  compilation commands match baseline `9c715720`; only `batch.cpp` source differs.
  The executable identifies `74f385dc` and has SHA256
  `dab3df75b08b02f48cd605ac0e8f23b2cdfa540f0738f3c6a4950cab69649767`.
- Strict numerical/state checks: 104 records and 32 complete post-replay state
  pairs per rank, then 1,992 wider records and 498 complete state pairs per rank.
  Short and actual 32K histories, greedy/sampled/mixed sampling, C2–C8, mixed AR,
  borrowed same/cross-session, serialized/file restore and ragged 9/12-row
  fallbacks retain full logits, tokens, positions, RNG and proposal work.
  Snapshot comparisons assert complete materialized bytes within rank.
- Serving: smoke 12, cache 4, cancellation 9, initial B/C/B 60, fresh full-release
  C8/ragged 29 and final fixed 624 records pass their output/work/runtime checks.
  Actual MTP, binary/source/settings, PID exit, unchanged boot/topology/module,
  error counters and final idle state are checked. No transport error changes.
  File roundtrips in the state probe are within one process; this is not a new
  disk-cache restart or physical-fault qualification.

The earlier numerical binaries replace one host object in the qualified release
and retain parent version metadata. Their source/SHA proofs remain in the archive;
they are not relabeled as full-release performance artifacts. Operator validation
uses unchanged qualified GPU/header library bytes and does not itself test engine
routing. Real-model routing and full-release tests provide that separate evidence.

## Actual dispatch reduction

Fresh, equally instrumented standard rocprofiler-SDK captures of baseline and
candidate retain identical ordered work in all eight rank/cycle comparisons.
Same-host public SDK clock calibration passes; peer clocks are never subtracted.

| Target rows | Baseline dispatches, rank 0 | Candidate | Removed | Actual fused calls |
| ---: | ---: | ---: | ---: | ---: |
| 9 | 2,993 | 2,993 | 0 | 0 |
| 8 | 2,228 | 1,844 | 384 | 48 |
| 3 | 2,230 | 1,846 | 384 | 48 |

Both nodes remove exactly eight dispatches per eligible layer, about 17.2% of
all dispatches in these eligible passes. Both builds still have one final host
stream synchronization. The nine-row fallback is unchanged. This is fusion plus
existing producer reuse, not pure launch-only savings. Eligible instrumented
spans are 0.76–1.41 ms shorter in this one paired capture; that is not significant
unprofiled throughput or evidence that host waits are wholly avoidable.

GPU CSVs in the bundle contain every original-field row overlapping the four
measured target-pass windows per host/arm. The complete original captures remain
local; their SHA256, byte/record counts, exact windows and selection code are
included. These are disclosed derivatives, not the full profiler streams. No
independent SDK loss counter exists. API and GPU intervals overlap and must not
be added as separate costs.

## Fixed unprofiled serving comparison

Twelve balanced randomized pairs, 24 fresh starts, 624 requests. Methods, seed,
order, sample count and statistics were fixed before execution. No exclusions,
replacement starts, interim significance, historical pooling or magnitude cutoff.

UD-Q4_K_XL plus shared-Q8 MTP, greedy/thinking off, eight sessions, 65,536 context
capacity, chunk 2,048, 1 GiB RAM cache, approved native driver/rails unchanged.
Concurrent prompts are shallow; C1 controls use actual 2,048-token prompts and
request at most 128 output tokens. Capacity is not actual prompt depth.

| Co-primary | Baseline wave tok/s | Candidate | Paired change | Marginal 95% paired bootstrap CI | Holm3 p |
| --- | ---: | ---: | ---: | --- | ---: |
| C2 | 72.292 | 72.559 | +0.3694% | [-0.000628%, +0.758381%] | 0.282715 |
| C4 | 97.227 | 97.432 | +0.2110% | [-0.075363%, +0.501275%] | 0.386719 |
| C8 | 123.779 | 123.851 | +0.0581% | [-0.825238%, +0.926523%] | 0.898438 |

Rates shown are geometric means across pairs. Complete client-wave rate is total
completed output tokens divided by earliest request start to latest finish;
it includes queue, prefill and stream completion. It is not the sum of individual
request decode rates or pure GPU decode. Holm3 uses exact paired sign flips under
a sign-symmetry assumption; balanced-assignment sensitivity is also retained.
All three estimates are **inconclusive**, not equivalence or rejection by size.

There are **228 phase-flag entries across 64 paired request comparisons**, at the
fixed 5% AND 3 ms threshold, and **291 adaptive-work field comparisons** (114 draft
counts, 91 accepted counts, 86 round counts). Correlated client/server/raw fields
are not independent regressions. C1/solo speculative work matches. Every flag
and changed-work field is retained; whole-wave rates do not waive request delays.

A posthoc descriptive audit finds high/low startup modes in unchanged controls
as well: C4 actor 2's first gap exceeds 180 ms in 7/12 baselines and 5/12 candidates;
C8 late-actor TTFT ranges overlap. This does not clear the candidate, establish a
scheduler cause or prove equal per-cycle work from aggregate counts. Timings
remain unqualified. No further confidence repetitions are proposed to seek a
positive result.

## Evidence and reproduction

[Acquisition bundle](batched-ffn-hc-fusion-data/acquisition.zip),
[manifest and scope](batched-ffn-hc-fusion-data/archive.json),
[offline audit](batched-ffn-hc-fusion-data/check.py),
[paired figure](batched-ffn-hc-fusion-data/paired-performance.png),
[PDF figure](batched-ffn-hc-fusion-data/paired-performance.pdf).

```sh
python3 docs/patches/batched-ffn-hc-fusion-data/check.py
```

The audit checks all 929 entry hashes, 624 raw identities, paired statistics,
flags/work, state/operator/lifecycle records and the scoped GPU-window unions,
counts and actual fused launches. It does not rerun GPUs, live PID/boot/kernel
checks, inference or physical faults. Original harnesses, exact commands,
protocol hashes, stdout/stderr, raw request events and failure notes are retained.
Models, binaries, objects, control tokens and multi-GB state files are excluded.
The initial copy-working-directory failure and unavailable system-Python plotting
package are retained; corrected preparation and an existing project environment
were used, without rerunning measurements or relaxing assertions/tolerances.

This fork evidence is TP2-specific. A common canonical port is prepared but has
not received independent canonical TP1 qualification. The separate Q8-reuse PR
is not included. Maintained `tp2` remains unchanged pending review and evidence.
