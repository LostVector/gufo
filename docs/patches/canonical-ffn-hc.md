# Canonical batched FFN HC fusion: draft qualification

Candidate [6ec3f75f](https://github.com/LostVector/gufo/commit/6ec3f75f338d9f2154ef60e1c14e2dd2c70ff53b)
is based on gufo-org `8ab834a5`. The production change routes the immediate,
eligible target-FFN combine/mix pair through the existing fused HC implementation.
It retains format, geometry and residual-state guards and the original fallback.
It reuses the existing producer's quantized activations too; this is more than a
launch-only change. There is no RDMA dependency, public API change or new switch.

This is the common-code port of [fork draft #15](https://github.com/LostVector/gufo/pull/15).
The maintained `tp2` branch has not adopted it. That draft's independent TP2 traces
show 384 fewer dispatches per eligible 3/8-row target pass, with a 9-row fallback
unchanged. Its 12-pair/24-start/624-request serving study was statistically
inconclusive at C2/C4/C8. Those results do not qualify canonical TP1 performance.

## Directly observed canonical results

- Fresh HIP GPU-test build: the complete existing HC fixture and 48 additive
  batch-reference conditions pass, assertions enabled, real GPU, no skips.
- Fixed-history state harness: 104 records, 64 matched replays, 32 complete
  materialized state comparisons at depths 0/32768, sampled and greedy. Full
  vocabulary scores, tokens, RNG, positions and acceptance/work agree, including
  borrowed, serialized and file-backed restore. Greedy plans alternate 1/0 in
  the diagnostic engine; sampled planning is unchanged. This is not a production
  greedy-controller replay or independent full-model teacher qualification.
- Both full fresh production builds pass. All 327 normalized compile commands
  match; `batch.cpp` is the only changed compiled production source. Canonical
  headers/API 0.1.15 are used throughout, with no fork API 0.1.16 library mixing.
- Normal-controller HTTP B/C/B: all 27 responses and cache work match, including
  coding prompts solo/concurrent/solo and actual MTP. Six timing entries and
  eight adaptive-work differences remain; see the next section.
- Standard fixed-token C1 benchmark: two repetitions per production binary,
  identical generated-token hashes. Synthetic pp2048 and tg128 are **separate**
  workloads; generation uses a 16-token prefix, depth 0, capacity 4096 and stock
  `stop_at_eos=false`. They are not HTTP chat or generation after a 2048-token
  history. These two-repetition figures do not establish a performance difference:

| Synthetic C1 workload | Stock 8ab834a5 | Candidate 6ec3f75f |
| --- | ---: | ---: |
| pp2048, tokens/s | 1767.00 ± 6.37 | 1763.99 ± 9.13 |
| tg128 with MTP, tokens/s | 44.88 ± 0.12 | 44.76 ± 0.02 |

The table preserves the stock CLI's summary notation. It is not a confidence
interval, equivalence result, significant change, or adoption recommendation.

## Accounting failure and timing flags remain visible

The HTTP requests set `max_tokens=128`, but all six pp2048 replies naturally end
at 96, identically in both controls and candidate, with server `finish=stop`.
The frozen original analyzer incorrectly required 128 generated tokens and
failed. Its protocol, source and failure remain unchanged. A separate explicitly
posthoc observed-EOS analysis checks those actual completions without claiming
HTTP pp2048/tg128. All coding replies do generate 128 tokens. The separate stock
CLI benchmark above generates the full fixed 128 tokens.

At context capacity 65536, two session slots, chunk 2048, 1 GiB RAM checkpoint
cache, UD-Q4_K_XL target/shared-Q8_0 MTP, greedy/thinking-off/cache-prompt-false,
the HTTP candidate's C2 actor0 first follow-up gap was 108.80 ms versus
71.51/70.35 ms in the controls. Maximum interval was 108.74 ms versus
80.48/81.32 ms. Six per-request/phase entries exceed both 5% and 10 ms; eight
draft/acceptance comparisons differ. No flag, request or work difference is
discarded, and no serving gain or no-regression claim follows from this screen.

Focused diagnostic v1 captured a 109.43 ms gap in **unchanged stock upstream**.
That gap ended 15.57 ms before its first target `ForwardBatch` began. Therefore
that particular baseline delay occurs before the patched batched-target route.
The candidate/later-control gaps were 69.31/66.00 ms. This reversal does not
attribute or exonerate the original uninstrumented candidate flag.

Diagnostic v2 adds serial `DecodeStep` and prefill `Feed` spans, with session
identity, position and token count. Its three gaps were 68.53/70.89/68.34 ms.
One baseline gap includes 64.10 ms of serial decode; candidate/later-baseline
gaps predominantly include a batched cycle. Requests may change order, so
`requests.front()` position alone does not identify the same actor across runs.

A separate baseline-only event capture stopped at its predeclared six-start cap,
without a >95 ms gap: 66.50, 68.82, 65.14, 63.86, 72.50 and 67.99 ms. All 54
responses remain. This was bad-state capture, not a throughput/significance study;
its conditional stop rule must never be pooled into comparative statistics.
Logging may perturb arrival and adaptive decisions. The original flag remains
unresolved, rather than waived or used to justify endless confidence repeats.

## Reproduction and evidence scope

[The data manifest](canonical-ffn-hc-data/manifest.json) lists 258 retained text
artifacts: exact source/build/run commands, raw requests and stage timestamps,
state hashes, operator logs, original failures and observed CI states. Models,
binaries, objects and large materialized snapshot blobs are excluded. Source
logging adds no GPU wait or forced draft plan; its timings are diagnostic, never
release serving measurements. Nested engine/batch times must not be summed, and
stream-completion waits include necessary queued GPU execution.

The lab used EVO-X2/gfx1151, private ROCm 7.14.1/Clang 23, kernel
`7.3.0-strix-test`, Qwen3.8-Flash-Next UD-Q4_K_XL/shared-Q8_0 MTP. Every run records
boot identity and `/proc/cmdline`. No driver, system ROCm, kernel, boot, transport,
NVMe power or Halogen setting changed. No profiler/admin sampler/full-table
warming was added to the release screen. Source and diagnostic builds retain
their exact roles; source identity does not imply binary interchangeability.

The retained lab scripts use the original absolute workspace/toolchain/model
paths. Their README files explain setup and scope. Reproduce in a new named
output directory; controllers refuse to overwrite a frozen protocol. For a
portable read-only integrity and bounded-result audit:

```sh
python3 docs/patches/canonical-ffn-hc-data/audit.py
```

An offline audit verifies the retained packet and recomputes the HTTP flag/work
comparisons. It cannot independently rerun GPU arithmetic, materialize omitted
state blobs, or validate past live PIDs/kernel behavior. The original diagnostic
sources and commands are included for those hardware runs.

Exact candidate-head [repository/Linux checks](https://github.com/LostVector/gufo/actions/runs/41495760924)
passed. Exact-head [Windows validation](https://github.com/LostVector/gufo/actions/runs/41495760751)
also passed: extension contracts, packaging and CPU checks. Its GPU model job
is skipped on the fork; no Windows GPU-model qualification is claimed.
The final observed results are in `CI-final.json`; earlier pending snapshots
remain as historical observations.

The next review decision is about the small common-code dispatch change and its
correctness. Confirmed end-to-end acceleration, resolved cold first-gap latency,
wider canonical concurrency and default adoption remain unproven.
