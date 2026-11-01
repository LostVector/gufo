# Two-row MTP target-head gathering

This experimental change partitions the ordinary two-row MTP target vocabulary
projection between two ranks. Each rank computes half of the original Q8 output
matrix, exchanges its packed rows and reconstructs every original logit. Local
rows expand in descending order before peer placement, preserving unread input
and the original logit bits.

The measured profile is Qwen3.8-Flash-Next **UD-Q4_K_XL**, with the shared Q8 MTP
model. Its target output matrix is Q8_0, 248320 by 2560; the optimization currently
requires that shape and type. Full output weights remain allocated on both ranks.
AR, prefill, true batches and wider verification retain their existing routes.
The original prefill mode is captured before the head selects decode arithmetic.
Both peers must run a matching build; this path negotiates capability bit 8 and
wire version 14. The snapshot payload remains version 19.

## Measured single-stream decode

Final measured binary `6c4f1403d57549e712d70b3c18b36585388b8198712bec0a84c04083afd533ae`
is compared with the experimental confidence-policy baseline
`1caf9512d974e7f4b93b34ef7f23586d31d61e03efcb3c2003edd64bc686c932`.
The review base pins that baseline's source at `a42f9b5a` on
`lab/tp2-first-cap-baseline`; it contains preceding experimental work and is not
a release/adoption claim.

Two Ryzen AI Max+395 hosts, 128 GiB UMA each, private ROCm 7.14.1/Clang 23,
gfx1151, kernel 7.3.0-strix-test, approved Thunderbolt module sourceversion
`81C6FF2FBAC473C7AB21844`, TX window 32, four native rails per host and crossed
40+20 Gb/s links. Both hosts use `ttm.pages_limit=32505856 amd_iommu=off`.

Both arms enable MTP, configure 65536-token capacity/two slots/chunk 2048 and
bypass request caching. Each fresh start runs a 19-token/8-output warmup followed
by two 2048-token prompts, max128 output, actual113-token EOS replies. This is
single-stream cluster throughput after a 2K prompt, not a filled65K benchmark.
All replies retain110draft proposals,77accepted and36rounds.

| Request | Baseline tok/s | Final candidate tok/s | Paired rate gain | Paired bootstrap95% |
|---|---:|---:|---:|---:|
| First 2K, primary | 73.5692 | 74.2058 | +0.86541% | +0.72484% to +1.01322% |
| Repeat 2K, secondary | 73.4262 | 74.0114 | +0.79696% | +0.61208% to +0.97880% |

Eight independent fresh-start pairs, four B/C and four C/B, were shuffled before
acquisition with seed 2026110103. All eight primary pairs improved. Exact two-sided
balanced-assignment p=2/70=0.0285714. Bootstrap uses20,000 paired resamples with
seed 2026110104. The separately reported p=2/256=0.0078125 assumes pairwise sign
symmetry. The repeated request is secondary, not an independent replication.
No interim statistics, exclusions, replacement runs, added pairs or pooling of
previous pilots. All 48 outputs and speculative/prompt/cache work match, all 32
transport windows have zero checked error/retry/retransmission deltas, and
frozen-source/binary/options/PID/boot/topology/kernel/final-idle checks pass.

The data, exact prompt/settings and pinned model-file checksums are in
[tp2-target-head-data](tp2-target-head-data). Target shard hashes come from the
pinned download manifest previously verified on both hosts; the shared MTP
artifact has its own checksum in workload.json. The CPU-only statistics check is:

```bash
python3 docs/models/qwen3.8-flash-next/tp2-target-head-data/check.py
```

The earlier head-only12c9study and the12c9/6c4f healthy error-fix pilot remain
separate experiments. These figures are an increment over an already optimized
MTP baseline in the 72–77tok/s range.

## Retained startup issues: draft review only

Thirteen strict >5% AND >3ms comparisons concern three warmups:

- Pair 4 prefill +34.315ms and pair 8 prefill +34.718ms, with corresponding first-token
  and wall aliases (12 comparisons).
- Pair 6 `raw.cache_snapshot_ms` +3.176ms (11.724vs8.547ms), not yet localized.

No 2K request has a strict timing flag. Separate pre-exec application captures
localized the shared~33ms initial-prefill mode inside rank1n-gram `pread` calls,
including on an unchanged baseline. That does not attribute these particular
untraced requests, prove equal tail frequency or establish native NVMe service
time. The snapshot metric may include waiting; it is not a pure copy timer.
All flags and the individual request metrics are included. Startup qualification
remains open; this draft does not claim blanket latency improvement or adoption.

## GPU execution failure propagation

A deliberately injected invalid-value return at the new peer-copy API exposed
a 180-second failure path: rank 1 recorded the decode exception and waited for another
command, while rank 0 continued into MtpForward and waited for exchange 617 that the
peer would never issue. Existing180/240second transport/GPU timeouts were retained.

A failed distributed `Executor::Run` now invokes terminal communicator abort.
The native transport poisons itself, releases queued GPU waits and shuts down
its bootstrap socket to wake the peer's exchange thread. Arrival publication is
serialized with abort so a successful in-flight transfer cannot overwrite the
terminal wakeup value. Recoverable cache and sampling handling outside Run is
unchanged. This is a terminal pair failure; restart both model processes.

| Injected failure on final candidate | Full failed request duration |
|---|---:|
| Rank0new-head copy API error | 0.449s |
| Rank1new-head copy API error | 0.755s |
| Rank1process exit at copy dispatch | 0.794s |

These include inference and test bookkeeping, not pure error/wire latency.
Each fault was matched by binarySHA/PID/callerPC and496640-byte width/height2;
every response was explicitly non-normal and non-timeout. All 9 fresh recovery
replies and MTP work matched; all 12 transport windows were clean. Actual GPU
hardware faults, physical cable loss and arbitrary exceptions are not covered.
All 8 header-dependent host objects were rebuilt; GPU kernels and other objects
were preserved. Healthy12c9/6c4f B/C/C/B control had12 exact replies/work and zero
strict flags, without a gain/equivalence claim from that two-pair pilot.

### Reproducing the exact API boundary

Use private benchmark servers, the same unstripped binary on both nodes and an
already configured native transport. No driver/root changes are made by this
tool. On the selected fault rank, prepare a fresh directory:

```bash
python3 tests/functional/tp_head_fault.py prepare \
  --binary /absolute/path/to/gufo --hip /absolute/path/to/rocm \
  --output /tmp/head-copy-rank1 --rank 1 --mode copy-error
```

Start the other rank normally. On the selected rank, replace the binary prefix
of its ordinary `gufo serve llm ... --tp-world-size 2 --tp-rank 1 ...` command with:

```bash
/tmp/head-copy-rank1/launch.py serve llm ... --tp-world-size 2 --tp-rank 1 ...
```

The launcher verifies the binary/library and role, records its PID/start time,
arms once and confines LD_PRELOAD to that exec. Send a greedy thinking-off
streaming request on rank0, max512tokens, such as “Write a detailed numbered list
of one hundred practical Linux performance debugging techniques.” Save its SSE
response and enforce an absolute client deadline (for example `curl --max-time
10 --no-buffer`). After shutdown, copy that response file to the fault host and
verify both the exact injection and explicit non-timeout error:

```bash
python3 tests/functional/tp_head_fault.py verify \
  --output /tmp/head-copy-rank1 --response /path/to/response.sse
```

Repeat with a fresh directory and `--rank 0` for the other origin; prepare/start
the launcher only on the chosen rank. `--mode peer-exit` deliberately exits the
task-owned process with status86at that host dispatch point. The metadata-only
verification does not prove a prompt, completion deadline or fresh recovery;
retain the response/client timing and restart both private servers. Stripped
binaries or an unrecognized callback symbol are rejected rather than guessing
an address. The injector requires the guarded model shape above.

## Snapshot/cache qualification

A diagnostic host counter on the final6c4fobjects observed56 actual two-row gathers
per rank. Greedy andtemperature 0.7 / seed 73 continuations matched direct execution
across borrowed snapshots (same/other session), serialized bytes and a file byte
roundtrip. All committed tokens, positions and full 248320-score frontiers match
an independent autoregressive teacher; replay RNG agrees when initialized from
the same saved external sampler state. Prefill/AR fixture gather counts are zero.
This fixture has a32-token prompt/short continuation, not a filled65K history.

Production HTTP qualification passes13 case records: greedy/sampled interrupted
assistant kept/discarded,317 cached tokens, both live C2 cancellation victim roles,
exact survivors versus solo controls and post-cancellation replays. A separate
real two-host process-pair restart restores2287 of 2294 tokens from disk, re-prefills 7,
and produces the same 20-token answer as cold/RAM retry with actualMTP. Both rank-local
cache files persist. The private disk budget is2GiBper rank/RAM1GiB; initial disk
startup normally fingerprints artifacts. No manual table warming or system
changes were used. Two expected worker-control-close messages after intentional
head shutdown are retained, with no unknown server errors.

These are selected cases, not sampled-disk-replay, cancellation-during-disk-restore,
full-cache-capacity or universal model-quality guarantees. The RDMA communicator header used here was absent from upstream main at
`src/models/qwen38_flash_next/kernels/rocm/communicator.hpp` as checked on
2026-11-01. Upstream integration requires the transport prerequisite. This draft targets the fork's pinned
experimental baseline.
