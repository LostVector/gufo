# Ragged decode latency: request-identified round counters

Two fixed diagnostic B/C/B replays retain the preceding20-request four-session
history and the same production binariesfbacf9cc/2669e07c. Only the raggedC4 wave
is traced, after the original15 controls/requests and before the post-wave replay.
UD-Q4_K_XL/shared-Q8_0MTP,thinkingoff,65Kcapacity,4sessions,chunk2048,1GiBcache,
unchanged81C6/TX32 native driver/crossed40+20Gb/s. No GPU/SDK/NVMe/admin sampler,
production edit, new idle/priming or full-table warming. Necessary probe activation
and inherited10s post-stop cleanup are recorded. Diagnostic times are not added
to the original untraced confidence study or used to claim a new speedup.

## Directly measured

The final trace reads each Session's cumulative rounds/proposals/accepted counters
at FinishDecode entry and return, in scalar **and true batched MTP**. Session::Sync
this pointers join to their containing head StepPrefill's managed-request identity;
publication uses that same opaque request pointer. There is no order-only session
attribution. Per-request sums equal all server counters, and publications plus
speculative/plain completion accounting match all243 tokens per ragged wave.

| 33-token actor1 | Baseline before | PLE candidate | Baseline after |
| --- | ---: | ---: | ---: |
| Speculative rounds |12|13|10|
| Accepted proposals |15|15|21|
| Plain completions |6|5|2|
| Batch steps containing request |18|18|12|
| First-to-last publication span,ms |1219.785|1221.838|978.971|
| Shared batch-engine host spans,ms |1045.821|1049.186|805.057|
| Peer prefill host spans,ms |126.203|125.535|126.051|
| Other scalar decode host spans,ms |46.188|46.137|46.863|
| Outside those host spans,ms |1.572|0.980|1.000|

[Measured partition](host-partition.png) and [per-round CSV](rounds.csv) show that
the candidate-vs-fast-control publication difference242.867ms is accounted for
by244.129ms more inside shared batch calls, offset by slightly less work in the
other categories. It is not an unexplained quarter-second gap between engine
calls. There are six more completed batch steps: three more speculative rounds
and three more plain completions, with fewer accepted proposals. Actual proposal
counts for this request's first three steps are2/2/2 in the candidate versus3/5/5
in the fast baseline; these are observed proposals, not recorded planner caps.

Host spans include GPU execution, synchronization, launch work and transport.
**This does not prove HIP launch/wait overhead inside them is small, or label
244ms as GPU computation.** Less than1ms outside these spans is not an upper bound
on all avoidable host overhead. Shared batch time is not added once per member;
the partition clips disjoint head-thread enclosures to each publication window.

All60 outputs/prompt/cache-work identities match and widths3/4/actualMTP remain
verified. Both ranks' six checked native error windows, frozen source/binary/
harness/settings/boot/topology/PIDs/kernel and approved final idle checks pass.
Peer per-session counter triples also match the head as multisets, without
cross-host clock or ordinal-actor assumptions. Read-error/ring/map/recursion
counters and remaining maps are clean, and every
recorded completion has paired entry/return counters. An independent kernel
uprobe missed counter is unavailable; exact counter/publication totals provide
additional coverage checks without claiming universal losslessness.

The six new strict phase entries all concern actor1 against the faster **later**
baseline. They remain alongside every earlier flag. Candidate complete ragged-wave
throughput67.599tok/s versus67.089/65.872 controls improves in this observation,
while its33-token reply is slower than the latter control. One B/C/B is not
statistical evidence of either gain or equivalence. The128-token actor3 publication
span is2895.622ms versus2981.191ms in the fast-actor1 baseline. Forcing the policy
that helped actor1 is not established as an improvement for all clients.

## Code-level explanation and remaining hypothesis

TP2 here already exposes batched MTP. The first diagnostic confirmed that ordinary
pool AdvanceBatch was not called in the ragged wave; scalar-only attribution was
insufficient. The final recorder covers Session::FinishDecode for all members of
TpMirroredRunner::DecodeBatch as well as scalar calls. No patch enabling an
already-present batched path is proposed.

In source2669e07c, `MtpBatchController::Choose/Observe` in
`src/models/qwen38_flash_next/mtp_policy.hpp` learns costs by concurrency/context,
retains measured cells across requests, skips transition costs and periodically
explores widths/headless passes. `Session::PlanBatch` chooses on rank0, and
TpMirroredRunner serializes the shared/per-member draft plans to rank1. This is
a code-supported explanation for timing-dependent greedy work histories while
outputs stay exact. The particular learned-cell/choice inputs causing this
triplet's divergence were **not** captured, so that last step remains a hypothesis.
The already-fixed C2→C1 stale-backoff mechanism is a separate issue; do not
reopen it, initial-cohort timers or first-block experiments from this result.

Next: observe the actual batch-controller selected widths and cost cells in
this history, then consider one isolated policy change if those observations
justify it. Preserve per-client latency and whole-wave throughput as separate
outcomes. Do not change mathematical outputs or remove dependencies to make a
latency graph flatter. All timing claims remain scoped; no default adoption.

## Reproduce and inspect

[Acquisition archive](acquisition.zip) contains both diagnostic attempts,120raw
JSONL records,12peer traces, protocols/harness, analyses and the first trace's
scope-label erratum. V1 lacked per-request batch completion counters and its
inherited analyzer description incorrectly said untraced; the protocol and
erratum explicitly identify it as traced. V2 corrects that label before freezing.
No original report was overwritten or discarded.

[Recorder archive](recorder-v5.zip) contains source, exact artifact hashes,
reproduction build commands, real-header ABI fixture, both-host positive/negative
CPU controls and binary-DWARF layout checks. Both pinned ELF layouts put cycles/
drafted/accepted at304/312/320bytes; the real Stats type is48bytes. BPF copies the
first24bytes at entry/return and never changes engine state. The fake fixture's
known16byte offset tests nested/speculative/plain deltas and invalid-pointer
rejection; it is a recorder sensitivity control, not an inference substitute.
The earlier failed GPU hardware-counter shim remains disabled.

From the existing lab project root, use the source-frozen2669e07c worktree and
private runtime environment, then run the study`run.py`, `analyze.py`, `rounds.py`;
plot via`.venv/bin/python .../plot.py`. Both hosts need the compiled recorder and
existing read-only root inventory/recording helper. CPU selftests refuse to
replace old result directories: copy the recorder to a fresh lab path before
repeating them. Paths/SSH/model locations require the established lab setup;
these archives are evidence/reproduction inputs, not an installer.

Audit the exported final trace without a GPU or root access:

```sh
python3 docs/patches/tp2-ple-owner-data/ragged-round-trace/check.py
```

This independently rebuilds identity joins, all376 head completion deltas,
peer counter multisets, disjoint raw host partitions, and every strict phase
flag/adaptive work difference from the archives. It also verifies recorder
artifact hashes and pins both-host CPU controls. It does not rerun inference,
certify hardware or establish timing equivalence.
