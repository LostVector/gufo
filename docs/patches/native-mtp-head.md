# Native MTP TP2 head-sharing result

The isolated candidate improves C1 native MTP decode speed by **1.487%** on the
predeclared five-task workload (95% paired-process bootstrap CI **[1.298, 1.684]%**,
exact balanced-order two-sided **p = 0.007937**). All ten process-pair estimates
are positive. All 120 requests match in output and non-timing work counters.
No pilot data, exclusions, replacement runs or optional stopping are used.

The change extends the existing lossless TP2 target-head gather from native
two-row verification to guarded native widths 3–8. Both ranks previously computed
the full vocabulary head at those widths. There is no new kernel, allocation,
public flag, arithmetic, sampler/controller policy or driver change. AR, prefill,
existing two-row native MTP and external DFlash routes retain their behavior.

## Fixed production study

Twenty fresh starts, ten independent process pairs with five baseline-first and
five candidate-first orders, five tasks clustered within each pair, sampled
seeds 2001–2010. One primary hypothesis; 20,000-draw order-stratified bootstrap
and all 252 balanced-order randomizations. Intervals are conditional on this
workload and machine pair. Per-task numbers below are descriptive, not five
separately qualified hypotheses.

| Task | Baseline decode tok/s | Candidate | Paired change |
| --- | ---: | ---: | ---: |
| Short, greedy | 70.25 | 71.95 | +2.411% |
| Short, sampled | 64.54 | 65.30 | +1.169% |
| Actual 2,048-token prompt, greedy | 67.77 | 69.06 | +1.900% |
| Actual 2,048-token prompt, sampled | 59.60 | 60.28 | +1.145% |
| Prose, sampled | 56.08 | 56.54 | +0.819% |

Client wall throughput improves **1.219%**, CI [0.994, 1.452]%, as a secondary
outcome. No separate multiplicity-controlled claim is made for it.

Settings: two Strix Halo machines, approved B6 Thunderbolt driver, existing dual
cables, UD-Q4_K_XL target / shared Q8_0 native MTP, adaptive maximum seven drafts,
65,536 context capacity, one slot, prefill chunk 2,048, 1 GiB RAM cache, no prompt
cache/persistence or table warming, neutral penalties, thinking off. Sampled
requests use temperature 0.8/top-k 40/top-p 0.95. Natural EOS is retained.
These absolute rates must not be compared directly to older prose tasks with
different penalties, prompts or seeds. The live service uses two slots/chunk 512.

## Retained latency and transport limits

Six correlated timing flags across four paired requests remain under the strict
5%/3 ms rule. Two maximum-inter-token increases are 4.48 and 4.15 ms (below the
repository's 10 ms absolute threshold). Two sampled prefill/first-token pairs
are slower by about 134/135 ms and 67/67 ms. All four entries above 10 ms occur
in those two prefill/first-token pairs.

Four starts have recovered transport errors: two baseline and two candidate.
Both prefill-flagged candidate starts have errors; one paired baseline also has
errors. Before/after counters do not locate those errors in a request or prove
they caused a timing flag. The earlier DFlash diagnostic reproduced errors in
shared prefill on both paths, but that is not a per-command trace of these new
runs. The source guard excludes prefill; this alone does not clear observed
serving latency flags. No sample was removed or replaced. This establishes mean
decode gain, not uniformly improved tails or transport reliability.

## Qualification and attribution

A separate source-only observer shows target execution occupies roughly 86–91%
of decode-cycle wall time, including real GPU, transport and synchronization
work. Many calls have three rows (66 of 120 prose decode passes); they previously
duplicated the head. Per-case decode starts at the first `PrepareDecode` scope,
excluding small prefill tails. Nested phase totals are not additive. Observer
results are not pooled into production timing, nor called pure CPU dispatch.

Per node, 64 real-weight component cases check 71,516,160 finite bitwise logits
and allocation guards at every width 1–8. The actual engine probe checks 294
replays and 28 complete serialized post-replay state comparisons at short,
16K and 32K actual depths. All widths 1–8 are exercised. Verification/frontier
logits, tokens, RNG including deferred draws, history, proposal work and complete
state bytes match between baseline and candidate, including rejection, rollback,
borrowed-object, serialized and file restores. Cross-node full logit hashes match.
Borrowed objects are materialized before mutation; the diagnostic adds GPU
synchronization and does not independently qualify asynchronous lifetime safety.

Unmodified HTTP fixtures pass stops, long-context histories, state edges,
cancellation/resume, isolated cache spacing and disk persistence/restores.
The isolated cache ownership prevents the previously identified legacy aggregate
checkpoint contamination; no assertions or thresholds were weakened.

Source and binary hashes are pinned in `production-provenance.json`. All 338
normalized production compiler commands match. The baseline artifact is source-
equivalent to merged ff485f17 but retains its earlier build revision string.
The candidate is an ordinary uninstrumented release-preset build. Format checks
cover 915 C++ files; extension API and strict model/core/serving/family/ownership
checks pass. Numerical diagnostics are separately built and hashed.

## Concurrent serving screen

The four-start ABBA screen preserves all 52 output/prompt/cache identities.
Two active clients, eight slots, greedy neutral-penalty requests, two waves each
of natural text and coding per process. It does **not** establish a C2 speedup or
equivalence: only two independent process pairs are available, with 38 adaptive
work-field differences. This is not the equal-work C1 qualification above.

| C2 workload | Baseline summed decode tok/s | Candidate | Change | Client wave throughput change |
| --- | ---: | ---: | ---: | ---: |
| Natural text | 89.23 | 89.63 | +0.454% | +0.418% |
| Coding | 95.95 | 95.51 | −0.455% | −0.328% |

Nine correlated phase flags remain, including about 133 ms slower prefill/first
content for one C1 control request inside the C2 screen. That candidate start has
recovered CRC/loss/retransmit counters on both nodes. Counter snapshots cannot
prove a per-request cause. No run or wave was excluded. Concurrent performance
and tail latency remain unqualified; do not describe these mixed point estimates
as an established gain, regression or equivalence.

One setup attempt failed before its first inference request because the client
URL appended `/v1` twice. The zero-row failure and original logs are retained in
`failed-c2-attempt0/`; only that client URL was corrected before running the full
fixed screen in a fresh directory.

## Reproduction and disposition

See `README.md` for commands, `PLAN.md` and `confirmation/DESIGN.json` for the
protocol, `confirmation/analysis.json` and `phase-flags.csv` for all paired
results, and the raw per-start request/response/server/counter files.
`candidate/validation.json`, `functional/` and `pilot/` retain quality and pilot
evidence. Diagnostic setup compile failures are retained separately.

The source change is isolated on local branch `review/native-mtp-head-sharing`.
No PR publication, upstream mutation, driver/kernel/ROCm/boot change or live
candidate deployment. Each wrapper restores the original native MTP service.

## Committed evidence audit

[Raw archive](native-mtp-head-data/acquisition.zip) and
[hashed inventory](native-mtp-head-data/archive.json) retain the raw requests,
responses, protocol, source, compiler provenance, quality logs, timing flags,
transport counters and failed setup attempts. Large snapshot payloads, binaries,
objects/libraries and private control configurations are excluded. Artifact paths
in this report refer to the recorded isolated lab. The archived README records
commands and pinned dependencies; a fresh directory is required for reproduction.

Run `python3 docs/patches/native-mtp-head-data/check.py` to verify source and
archive identity, retained checks and failures, and recompute fixed-study
statistics offline. It does not independently rerun GPU quality.

The repository documentation checker retains the base branch's two broken links
in `src/models/qwen/README.md` (`dflash_weights.cpp` and `hip/dflash/`). This change
adds no new documentation errors; the full documentation check is not green.
