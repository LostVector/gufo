# TP2 DFlash verification head sharing

The guarded change preserves the numerical and state behavior tested here and
improves decode speed on the predeclared workload. This is a fork review candidate.
The live service remains on native MTP.

## Change and measured scope

Reuse the existing lossless two-rank vocabulary-head gather for four- and
eight-row external verification, subject to the existing transport capacity.
Previously both ranks computed the full vocabulary head for these calls.
Native two-row MTP retains its existing route. Other shapes fall back unchanged.
There is no new kernel, allocation, public option or API contract.

The measured production source is byte-identical to this PR's executor source.
The archive pins both binaries, the source patch and 338 normalized compiler
commands. Both builds use the same release preset, dependencies and compiler;
revision metadata differs. The production runs contain no sampler or profiler.

## Fixed confirmation

Forty fresh starts, 200 requests including 40 warmups, ten independent process
pairs per width, balanced randomized baseline/candidate order and held-out
sampled seeds 2001–2010. All output and non-timing usage/work identities match.
Tasks are greedy and seeded sampled generation at short and actual 2,048-token
prompts. Settings: UD-Q4_K_XL target, Q8_0 DFlash2, two Strix Halo machines,
B6 driver, one slot, 65,536 context capacity, prefill chunk 2,048, fixed draft
width 3 or 7, neutral penalties, thinking off, 1 GiB RAM cache and prompt cache
off. Pilot data are excluded.

| Draft / target rows | Decode change | Marginal 95% paired-process CI | Holm-adjusted exact p | Client wall throughput change |
| --- | ---: | --- | ---: | ---: |
| 3 / 4 | +1.836% | [+1.634, +2.044]% | 0.015873 | +1.452% |
| 7 / 8 | +0.868% | [+0.639, +1.081]% | 0.015873 | +0.719% |

Decode speed is the registered primary outcome. The four tasks are clustered
within each process pair. The exact two-sided randomization enumerates all 252
balanced order assignments; Holm adjustment covers the two widths. The 20,000-draw
bootstrap preserves the two order strata. Confidence intervals are marginal and
conditional on these tasks and seeds. Client wall throughput is secondary, with
no additional multiplicity-controlled claim. All ten four-row pair estimates
are positive; nine of ten eight-row estimates are positive.

| Draft width | Workload | Baseline decode tok/s | Candidate | Paired change |
| --- | --- | ---: | ---: | ---: |
| 3 | short, greedy | 63.23 | 64.32 | +1.729% |
| 3 | short, sampled | 65.17 | 66.43 | +1.933% |
| 3 | pp2048, greedy | 57.81 | 58.96 | +1.984% |
| 3 | pp2048, sampled | 51.94 | 52.83 | +1.698% |
| 7 | short, greedy | 62.49 | 63.01 | +0.844% |
| 7 | short, sampled | 64.20 | 65.03 | +1.297% |
| 7 | pp2048, greedy | 54.84 | 55.33 | +0.898% |
| 7 | pp2048, sampled | 53.32 | 53.55 | +0.434% |

All 14 correlated phase flags across 11 paired requests remain: six snapshot,
four prefill, three time-to-first-token and one maximum inter-token latency.
The largest prefill regression is 115 ms, with a 110 ms first-token regression
in the same eight-row pair. Sixteen starts have recovered transport errors
(nine baseline, seven candidate); none were excluded or replaced. Mean gains do
not establish improved tail latency, candidate-specific error causation, or an
adaptive-policy or long-context speedup.

## Numerical, state and serving checks

The quality probe records 294 replay records and 28 complete serialized
post-replay state comparisons **per node**, at short, 16K and 32K actual depths.
It covers greedy and compact sampled execution, plus short noncompact sampled
execution with penalties. Complete verification matrices and committed-frontier
logits are finite and bit-identical. Tokens, RNG including deferred draws,
history, work and positions match. There are 126 actual gather cycles and 107
sampled rejection cycles per node. Fallback rows 1/2/3/7 and changed rows 4/8
are explicitly asserted. The preceding real-weight component check covers 32
cases and 29,798,400 logits plus guards per node.

Same-path and cross-path borrowed-object, serialized and file restores pass.
Borrowed objects were materialized before mutation: this does not prove an
unmaterialized copy-on-write race safe. The diagnostic adds synchronization for
complete logit capture, so it is not asynchronous-lifetime or throughput evidence.

Uninstrumented HTTP checks cover both widths, stops, state edges, actual 24.6K
histories, cached replay, cancellation/resume and disk restarts. The original
width-three aggregate cache report remains **failed**. Its legacy token-range
filter attributes foreign-conversation checkpoint 2828 to the spacing test.
Every individual check passed, and the same unchanged spacing assertions pass
in isolated caches at both widths. Independent cancellation/persistence restart
checks pass. No assertion or tolerance was relaxed. See the retained
`functional/DISK-SPACING-FAILURE.md` and original report in the archive.

With identical inference work, production counters show 169–200 additional data
work requests per width-three start and 130–160 per width-seven start on both
nodes, consistent with the added head exchanges. These are aggregate counters,
not a direct per-call kernel trace.

## Separate tail/error diagnostic

Four separate source/counter diagnostic starts cover both nodes, without the
profiler SDK or added GPU waits. The counter sampler's median interval is about
1.055 ms, p95 about 1.07 ms, and it consumes 8–10% of **one CPU core**. Its actual
maximum intervals are retained. It was absent from the confidence study. One
failed setup attempt incorrectly treated the live queue-depth gauge as monotonic;
that attempt is retained, and only that gauge's monotonicity guard was removed.

CRC/bad-frame/loss/retransmit counter changes reproduce in both baseline and
candidate four-row starts, during the actual 2,041-token prefill forward call
(about 1.45/1.51 seconds), with detection brackets of about 1.1–1.3 ms. The
eight-row diagnostic starts have no error events. These observed detections occur
in shared prefill execution rather than the DFlash vocabulary-head call.
Detection time is not DMA arrival time. This does not establish the cause of the
errors or explain all 14 untraced timing flags. No cross-host timestamp subtraction
is used. Source timings and sampled queue activity remain available.

## Decision and reproduction

Retain both guarded shapes for fork review: both primary decode outcomes resolve
positively. Transport errors and tail latency remain separate open investigations.
No upstream publication or live binary replacement is part of this change.

[Raw archive](dflash-tp-head-data/acquisition.zip) and
[hashed inventory](dflash-tp-head-data/archive.json) retain the protocol/design,
source/build provenance, requests, responses, phase flags, counter snapshots,
quality source/logs, original failed aggregate report, isolated ownership proof
and separate diagnostics. Binaries, model weights, object/static libraries,
private control configurations and large snapshot payloads are excluded. Actual
state-byte comparisons ran in the recorded probe; their hashes and logs remain.

Reproduction scripts are pinned to recorded lab paths and dependencies and
require fresh directories. Follow the archived `README.md` and `PROTOCOL.md`.
Build the pinned production candidate before constructing its diagnostic archive;
do not reuse objects from a profiler or tracing build.

```sh
python3 docs/patches/dflash-tp-head-data/check.py
```

This offline check verifies hashes, retained coverage/failures and recomputes
statistics. It does not rerun GPUs or independently prove model quality.
