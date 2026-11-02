# Canonical Gufo intake: November 2, 2026

The maintained `tp2` branch includes gufo-org `773a16419282a3b9790d685ce6b4dd980b285943`
and the previously merged native TP fixes. The qualified production source is
`9c7157201e8aa30cbfc51ddf3595a3dea13b196a`, tree
`fe79bd3c8a32a82d5ea339ddce5125e3749154f6`.
Fork `main` mirrors that canonical snapshot; it does not contain RDMA.

This is a checkpoint-correctness and upstream-sync adoption. **Startup timing is
inconclusive/advisory, not qualified as equivalent or faster.** The original
14 phase flags remain in the evidence. The separate Q8 reuse
[fork draft14](https://github.com/LostVector/gufo/pull/14) and
[upstream draft958](https://github.com/gufo-org/gufo/pull/958) are not included.
Earlier performance tables retain their original source and binary attribution.

## What changed

Upstream's checkpoint allocator counts the arena's existing reservation correctly
and can capture the live session at a split pass boundary when device headroom is
unavailable. The continuation adapter tracks that boundary explicitly. The local
prefill scratch default returns from 16,384 to 4,096 rows, matching canonical
Gufo's memory tradeoff. Native TP already capped each trunk batch at 4,096 rows to
fit its exchange window; its paired scheduling and transport limits are retained.

Distributed HTTP still mirrors separate bounded prefill/capture operations. It
does not advertise in-pass checkpoint admission or have an in-pass capture RPC.
The private reservation helper returns false under TP so an internal split path
cannot depend on unequal per-rank headroom; this does not enable a new advertised
TP capture capability. Rank-specific recurrent-state sizes are retained.

No driver, transport, model arithmetic, boot flag, system ROCm installation or
service setting changed. The existing PLE sharing and earlier request-End dispatch
remain included. No full n-gram table warming was used.

## Exact build and checks

Official CMake `release`: RelWithDebInfo, `BUILD_TESTING=OFF`,
`GUFO_ENABLE_TP2_RDMA=ON`, gfx1151, private ROCm7.14.1/Clang23.
Qualification executable SHA256:

```text
de7409c8e8c93d8fd7655ba1da87a45824e682ae47ec6f8403338997b262759a
```

[Repository CI](https://github.com/LostVector/gufo/actions/runs/41453594920)
and [Windows validation](https://github.com/LostVector/gufo/actions/runs/41453598440)
both pass on exact source9c. Fork Windows GPU model execution is skipped; this is
packaging/CPU validation, not Windows TP numerical qualification.

Locally completed checks:

- Four actual CPU tests; shared formatting895files, API0.1.16, generated target
  graph and docs278files/1,292links pass.
- Six HIP fixtures compile; real HC/projection GPU operators pass without skips.
  GPU-test assertions remain enabled; these are correctness artifacts.
- Official single-node snapshot/context8,192 and adapter/context16,384 in-pass
  tests pass, including AR/MTP with zero reported device headroom and normal
  checkpoint-arena capture.
- Supported native mirrored captures pass eight cases per rank at short/resumed
  2,047/4,095 boundaries, AR/MTP, with normal/asymmetric reported headroom.
- Actual32K native continuation passes four cases per rank:32,773-token prompts,
  checkpoint at32,768, context capacity65,536, AR/MTP; fresh/normal and resumed
  from1,999/asymmetric reported headroom. Complete payload/logit comparisons,
  sampled continuation, reset and borrowed/serialized restore assertions pass.
- Fresh production TP2 smoke12, cache4 and cancellation9 records pass, including
  live survivors and actual MTP.
- Matched94/9c/94 C3/C4/ragged60 records pass exact outputs/prompt/cache work,
  actual admissions/widths and runtime/counter/kernel/boot/topology/idle checks.
  Aggregate draft-work fields match in this untraced comparison.

The native reported-free-memory interposer changes one process's returned free
byte count only. It does not alter system memory limits or device policy.
The inherited helper prints "in-pass checkpoint"; the native tests use supported
split-prefill and separate captures, not a new distributed in-pass feature.

## Timing investigation and limits

The untraced wider comparison retains14strict phase entries, chiefly overlapping
first-C3-gap metrics about8–9ms above the first baseline. The later baseline also
slows, but candidate second-content arrival is still about6.3–6.6ms later than
that control. This was not dismissed as a reporting offset or waived by exact
outputs/throughput.

A fixed33-request B/C/B boundary trace preserves the eight preceding warmup/solo
requests before C3; only subsequent unaffected histories are omitted. It retains
14phase entries and6adaptive-work comparison fields. It reproduces the gap, but
aggregate work does not establish identical first-cycle computation.

A second fixed33-request trace records the first four real three-request decode
cycles on both nodes. It uses each build's own headers/libraries and the same
logging-only observer, with fixed-array timestamps emitted after the existing
policy observation. All16ordered-work comparisons match: plans, complete token
chains, prepared widths, lookup/emission counts and actual packed target rows.
The first cycle sets a shared three-proposal ceiling per request and packs
**nine verification rows**;
three resident requests do not imply three GPU rows.

| First C3 cycle, rank0 | Baseline before | Integration | Baseline after |
| --- | ---: | ---: | ---: |
| Complete instrumented cycle, ms | 63.902 | 64.923 | 67.378 |
| Target-forward stage, ms | 56.447 | 57.376 | 59.654 |

The integration lies between unchanged controls in this capture. That does not
clear the earlier flags or prove equivalence, especially with observer effects.
Target-forward includes host scheduling, GPU computation, communication and
synchronization; it is **not pure CPU dispatch or GPU time**. Reported snapshot
lifetimes also include completed-worker-to-consumption lag and must not be read
as memory-copy durations. The detailed trace has zero new phase entries and two
aggregate adaptive-work differences beyond its first four fully matched cycles;
all remain retained.

Under the repository policy that keeps inconclusive timing advisory for ordinary
and release PRs, this canonical correctness sync is adopted with strict numerical,
lifecycle, identity and CI gates passed. No speedup or unchanged-tail-latency claim
is made. Future performance profiling should separate actual target-forward GPU
work and communication/wait costs with real packed widths. These measurements do
not justify another SSD, driver or queue-policy patch by themselves.

## Reproduce and audit

Build the pinned source on both peers using the [TP2 instructions](../../models/qwen3.8-flash-next/TP2.md).
Keep the release preset and identical executable/provider/model files on both.
The tested profile remains UD-Q4_K_XL/shared-Q8_0 MTP, context65,536, slots2 (4for
wider tests), prefill chunk2,048 and1GiB RAM cache; TX32, active pollingoff,
four rails over crossed40+20Gb/s links. Context capacity is not prompt depth.

[Raw acquisition](upstream-773a1641-data/acquisition.zip),
[manifest](upstream-773a1641-data/archive.json) and
[offline audit](upstream-773a1641-data/check.py) retain sources, exact commands,
raw results, qualification and promotion records. They exclude models, snapshots,
executables, objects and control-token files. Run the audit with Python3; it
checks integrity and recorded coverage, not GPU execution or independent model
quality. Controller and build paths need adjustment on another machine.

The first native fixture incorrectly used a local-only in-pass admission oracle
and failed at its explicit zero-admission boundary. The supported-path fixture
retains the original numerical/payload assertions. Initial long-test preparation
also stopped before compilation because the old extraction contained unused
adjacent helpers; the corrected probe removes only those unused definitions.
An eight-character revision substitution corrupted one canonical full-hash label;
original acquired provenance is retained with an explicit erratum and verified
parent ancestry. Runtime source9c and executable identity were always pinned.
